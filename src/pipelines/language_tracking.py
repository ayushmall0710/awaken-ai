"""
Language Tracking Pipeline for ENG-05.

This module provides the LanguageTrackingAnalysis class to isolate language trials,
apply specific filtering, and select optimal electrodes for language tracking analysis.
It calculates Inter-Trial Phase Coherence (ITPC) using Morlet wavelets and DFT approaches.

References:
    - docs/language_tracking.md: "The Language Tracking Paradigm" & "Optimization Strategies"
    - tasks/ENG-05.md: Pipeline design and implementation details.
"""

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import mne
import numpy as np
import pandas as pd

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.pipelines.base import BasePipeline
from src.utils.signal_processing import normalize_channel_names

logger = logging.getLogger(__name__)


class LanguageTrackingAnalysis(BasePipeline):
    """
    Pipeline for Language Tracking Paradigm data.

    Coordinates data loading, channel selection, filtering, and ITPC computation.

    Attributes:
        loader (UnifiedDataLoader): Instance of UnifiedDataLoader.
        CLINICAL_20 (List[str]): Standard 10-20 system electrodes.
        LH_FOCUS_CHANNELS (List[str]): Left Hemisphere channels for language tracking.
        RH_FOCUS_CHANNELS (List[str]): Right Hemisphere channels for language tracking.
    """

    # Backward compatibility pointing to centralized configs
    CLINICAL_20 = config.CLINICAL_20
    LH_FOCUS_CHANNELS = config.LH_FOCUS_CHANNELS
    RH_FOCUS_CHANNELS = config.RH_FOCUS_CHANNELS

    # Filter constants
    HIGHPASS_FREQ = 0.5
    LOWPASS_FREQ = 30.0

    # Downsampling target
    TARGET_SFREQ = 256.0

    # ITPC Constants
    # Target frequencies based on Sokoliuk 2021 methodology:
    # Sentence-rate (~0.065 Hz) and Word-rate (~0.77 Hz)
    TARGET_SENTENCE_FREQ = 0.065
    TARGET_WORD_FREQ = 0.77

    # Frequency bands for band-averaged ITPC extraction.
    # Band-averaging across the entrainment band (rather than a single bin) is
    # more robust to small ICA-induced power shifts between adjacent bins and
    # reflects the finite bandwidth of neural entrainment responses.
    # Bands are taken from the stimulus design:
    #   Sentence: 12 words over ~15.5s -> ~0.065 Hz, nominal band 0.05-0.08 Hz
    #   Word: ~1.3s per word -> ~0.77 Hz, nominal band 0.70-0.90 Hz
    SENTENCE_BAND: tuple = (0.05, 0.08)
    WORD_BAND: tuple = (0.70, 0.90)

    ITPC_FREQS = np.logspace(np.log10(0.05), np.log10(2.0), num=40)
    ITPC_CYCLES = np.array([max(0.5, f * 2.0) for f in ITPC_FREQS])

    # Target frequency resolution for the zero-padded DFT.
    # Padding to 0.001 Hz resolution ensures the nearest DFT bin is within
    # 0.0005 Hz of the target frequencies, eliminating bin-mismatch artifacts
    # from short epoch lengths (1/16s = 0.0625 Hz raw resolution).
    DFT_FREQ_RESOLUTION = 0.001

    def __init__(
        self,
        loader: Optional[UnifiedDataLoader] = None,
        focus: Union[str, Iterable[str]] = "LH",
        filter_signal: bool = True,
    ):
        """
        Initialize the LanguageTrackingAnalysis.

        Args:
            loader: Optional UnifiedDataLoader instance.
            focus: Hemisphere focus ('LH', 'RH', or 'Clinical') or a custom iterable of channels.
            filter_signal: Whether to apply bandpass filtering.
        """
        super().__init__(loader=loader)
        self.focus = focus
        self.filter_signal = filter_signal
        self.epochs: Optional[mne.Epochs] = None

    def load(self) -> None:
        """Load and concatenate all language epochs for the patient."""
        sessions = self.loader.get_patient(self.patient_id).list_sessions()
        all_epochs = []

        for date in sessions:
            try:
                epochs = self.loader.load_clean_epochs(self.patient_id, date, trial_type="language")
                all_epochs.append(epochs)
            except FileNotFoundError:
                logger.warning(f"No clean language epochs found for {self.patient_id} on {date}. Skipping session.")
                continue

        if not all_epochs:
            raise ValueError(f"No clean epochs found for {self.patient_id}. Run 'awakenai preprocess' first.")

        self.epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]

    def preprocess(self) -> None:
        """Apply optimization steps: channel selection and bandpass filtering."""
        if self.epochs is None:
            raise ValueError("Epochs not loaded. Call load() first.")

        self.epochs = self.select_optimal_channels(self.epochs, focus=self.focus)
        if self.filter_signal:
            self.epochs = self.preprocess_signal(self.epochs)

        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            self.epochs.set_montage(montage, on_missing="warn")
        except Exception as e:
            logger.warning(f"Montage error for {self.patient_id}: {e}")

    def analyze(self, **kwargs) -> pd.DataFrame:
        """
        Compute ITPC (Morlet and DFT) and return results.
        """
        if self.epochs is None:
            raise ValueError("Epochs not preprocessed. Call preprocess() first.")

        focus_name = self.focus if isinstance(self.focus, str) else "Custom"

        # 2. Compute Morlet ITPC
        logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
        itpc_data, itc_obj = self.compute_itpc(self.epochs)
        morlet_metrics = self.extract_itpc_metrics(itpc_data)

        # Generate output plots
        output_dir = config.LOCAL_DATA_ROOT / "outputs" / self.patient_id
        self.plot_itpc_results(itc_obj, self.patient_id, str(output_dir), morlet_metrics)

        # 3. Compute DFT ITPC (Sokoliuk Method)
        logger.info(f"[{self.patient_id}] Computing DFT ITPC...")
        itpc_spectrum, dft_freqs = self.compute_itpc_dft(self.epochs)
        dft_metrics = self.extract_itpc_metrics_dft(itpc_spectrum, dft_freqs)

        # 4. Permutation test for statistical significance
        n_permutations = kwargs.get("n_permutations", 1000)
        logger.info(f"[{self.patient_id}] Running permutation test ({n_permutations} surrogates)...")
        null_sentence = self.compute_itpc_permutation_null(self.epochs, n_permutations, band="sentence")
        null_word = self.compute_itpc_permutation_null(self.epochs, n_permutations, band="word")
        p_sentence = self.compute_permutation_pvalue(dft_metrics["itpc_sentence"], null_sentence)
        p_word = self.compute_permutation_pvalue(dft_metrics["itpc_word"], null_word)

        # Combine results
        result_dict = {
            "patient_id": self.patient_id,
            "n_trials": len(self.epochs),
            "sfreq": self.epochs.info["sfreq"],
            "focus": focus_name,
            # Morlet
            "morlet_itpc_sentence": morlet_metrics["itpc_sentence"],
            "morlet_itpc_word": morlet_metrics["itpc_word"],
            "morlet_ratio_sent_word": morlet_metrics["ratio_sent_word"],
            "morlet_freq_sentence_hz": morlet_metrics["freq_sentence_hz"],
            "morlet_freq_word_hz": morlet_metrics["freq_word_hz"],
            # DFT
            "dft_itpc_sentence": dft_metrics["itpc_sentence"],
            "dft_itpc_word": dft_metrics["itpc_word"],
            "dft_ratio_sent_word": dft_metrics["ratio_sent_word"],
            "dft_freq_sentence_hz": dft_metrics["freq_sentence_hz"],
            "dft_freq_word_hz": dft_metrics["freq_word_hz"],
            # Permutation test
            "dft_p_sentence": p_sentence,
            "dft_p_word": p_word,
            "dft_n_permutations": n_permutations,
        }

        self.results = pd.DataFrame([result_dict])

        logger.info(
            f"Pipeline complete for {self.patient_id} ({focus_name}). "
            f"Morlet Ratio: {result_dict['morlet_ratio_sent_word']:.2f}"
        )
        return self.results

    def generate_summary(self) -> Any:
        """Generate summary of language tracking results."""
        if self.results is None or self.results.empty:
            return {}
        row = self.results.iloc[0]
        return {
            "patient_id": row["patient_id"],
            "focus": row["focus"],
            "morlet_ratio": row["morlet_ratio_sent_word"],
            "dft_ratio": row["dft_ratio_sent_word"],
        }

    def select_optimal_channels(self, epochs: mne.Epochs, focus: Union[str, Iterable[str]] = "LH") -> mne.Epochs:
        """
        Select subset of channels based on focus strategy.

        Args:
            epochs: MNE Epochs object.
            focus: Hemisphere focus ('LH', 'RH', or 'Clinical') or a custom iterable of channels.

        Returns:
            New Epochs object (copied) with picked channels.
        """
        available_chs = epochs.ch_names

        # NOTE: We DO NOT add CLINICAL_20 here to prevent signal dilution during global averaging
        # in downstream ITPC calculations. Only the targeted hemisphere channels are used.
        if isinstance(focus, str):
            if focus == "LH":
                target_chs = set(config.LH_FOCUS_CHANNELS)
            elif focus == "RH":
                target_chs = set(config.RH_FOCUS_CHANNELS)
            elif focus == "Clinical":
                target_chs = set(config.CLINICAL_20)
            else:
                raise ValueError(
                    f"Invalid focus string: '{focus}'. "
                    "Valid options are 'LH', 'RH', or 'Clinical'. "
                    "Alternatively, provide an iterable of channel names (e.g., ['F7', 'T7'])."
                )
        else:
            target_chs = set(focus)

        normalized_names = normalize_channel_names(available_chs)
        clean_map = {clean.upper(): orig for orig, clean in zip(available_chs, normalized_names)}

        picks = []
        missing = []

        for target in target_chs:
            target_upper = target.upper()
            if target in available_chs:
                picks.append(target)
            elif target_upper in clean_map:
                picks.append(clean_map[target_upper])
            else:
                missing.append(target)

        if missing:
            focus_name = focus if isinstance(focus, str) else "Custom"
            logger.warning(f"Missing channels for {focus_name} montage: {missing}")

        if not picks:
            logger.error("No valid channels found from target set. Returning original epochs.")
            return epochs

        focus_name = focus if isinstance(focus, str) else "Custom"
        logger.info(f"Selected {len(picks)} channels for {focus_name} focus.")
        return epochs.copy().pick(picks)

    def preprocess_signal(self, epochs: mne.Epochs, target_sfreq: Optional[float] = None) -> mne.Epochs:
        """
        Apply bandpass filtering and downsample.

        Applies 0.5-30 Hz bandpass filter (preserving Delta band for sentence-rate
        analysis) and downsamples to TARGET_SFREQ (default 256 Hz).

        Source EDFs record at 512 Hz. Downsampling to 256 Hz halves TFR computation
        time while preserving all frequencies of interest with large margin
        (Nyquist = 128 Hz >> 30 Hz low-pass). Matches Sokoliuk 2021 methodology.

        Args:
            epochs: MNE Epochs object.
            target_sfreq: Target sampling frequency after downsampling. Defaults to
                TARGET_SFREQ (256.0 Hz). Set to None to skip downsampling.

        Returns:
            New processed Epochs object.
        """
        if target_sfreq is None:
            target_sfreq = self.TARGET_SFREQ

        epochs_processed = epochs.copy()
        if not epochs_processed.preload:
            epochs_processed.load_data()

        logger.info(f"Applying bandpass filter: {self.HIGHPASS_FREQ}-{self.LOWPASS_FREQ} Hz")
        epochs_processed.filter(l_freq=self.HIGHPASS_FREQ, h_freq=self.LOWPASS_FREQ, method="iir", verbose=False)

        current_sfreq = epochs_processed.info["sfreq"]
        if target_sfreq is not None and current_sfreq > target_sfreq:
            logger.info(f"Downsampling from {current_sfreq:.0f} Hz to {target_sfreq:.0f} Hz")
            epochs_processed.resample(target_sfreq, verbose=False)

        return epochs_processed

    def compute_itpc(
        self,
        epochs: mne.Epochs,
        freqs: Optional[np.ndarray] = None,
        n_cycles: Optional[np.ndarray] = None,
    ):
        """
        Compute Inter-Trial Phase Coherence (ITPC) using Morlet wavelets.

        Args:
            epochs (mne.Epochs): Preprocessed epochs.
            freqs (np.array, optional): Frequencies of interest. Defaults to class ITPC_FREQS.
            n_cycles (np.array or int, optional): Number of cycles. Defaults to class ITPC_CYCLES.

        Returns:
            itpc (np.ndarray): ITPC data (n_channels, n_freqs, n_times).
            tfr (mne.time_frequency.AverageTFR): TFR object containing ITPC.
        """
        from mne.time_frequency import tfr_morlet

        if freqs is None:
            freqs = self.ITPC_FREQS
        if n_cycles is None:
            n_cycles = self.ITPC_CYCLES

        logger.info(f"Computing TFR and ITPC ({freqs[0]:.2f}-{freqs[-1]:.2f} Hz)...")
        power, itc = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            decim=1,
            n_jobs=1,
            average=True,
        )
        return itc.data, itc

    def compute_itpc_dft(self, epochs: mne.Epochs):
        """
        Compute ITPC using the Discrete Fourier Transform (Sokoliuk 2021 method).

        For each trial and electrode, zero-pads the time series to achieve
        DFT_FREQ_RESOLUTION (0.001 Hz) frequency resolution, then computes the FFT,
        extracts per-bin phase, and averages unit phase vectors across trials.

        Zero-padding interpolates the DFT spectrum (Sinc interpolation) so that
        np.argmin can resolve the exact target frequencies (0.065 Hz sentence,
        0.77 Hz word) rather than snapping to the nearest integer multiple of
        1/epoch_duration_s. Without padding, 16s epochs yield 0.0625 Hz resolution
        and the sentence-rate bin is 4% below the true target.

        Zero-padding does not add spectral information; it only improves bin
        alignment. The underlying phase estimates remain determined by the recorded
        data length.

        Args:
            epochs: Preprocessed MNE Epochs object.

        Returns:
            itpc_spectrum (np.ndarray): ITPC values, shape (n_channels, n_freqs).
            freqs (np.ndarray): Frequency axis in Hz.
        """
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]

        # Zero-pad to achieve DFT_FREQ_RESOLUTION.
        # n_pad = sfreq / resolution gives samples per cycle of the coarsest bin.
        n_pad = int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION))
        n_fft = max(n_pad, n_times)  # never truncate real data

        spectra = np.fft.rfft(data, n=n_fft, axis=2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        unit_vectors = np.exp(1j * np.angle(spectra))
        itpc_spectrum = np.abs(np.mean(unit_vectors, axis=0))

        logger.info(
            f"DFT ITPC computed: {n_trials} trials, {n_channels} channels, "
            f"n_fft={n_fft} ({freqs[1] - freqs[0]:.4f} Hz resolution)"
        )
        return itpc_spectrum, freqs

    def extract_itpc_metrics_dft(self, itpc_spectrum: np.ndarray, freqs: np.ndarray) -> dict:
        """
        Extract sentence-rate and word-rate ITPC from a DFT ITPC spectrum.

        Averages ITPC across all bins within SENTENCE_BAND and WORD_BAND rather
        than extracting a single nearest bin. This matches the gridsearch design
        and is more robust to ICA-induced bin-level power shifts.

        Args:
            itpc_spectrum: DFT ITPC array, shape (n_channels, n_freqs).
            freqs: Frequency axis from compute_itpc_dft.

        Returns:
            dict with same keys as extract_itpc_metrics for direct comparison.
        """
        sent_mask = (freqs >= self.SENTENCE_BAND[0]) & (freqs <= self.SENTENCE_BAND[1])
        word_mask = (freqs >= self.WORD_BAND[0]) & (freqs <= self.WORD_BAND[1])

        itpc_sent_val = float(np.mean(itpc_spectrum[:, sent_mask]))
        itpc_word_val = float(np.mean(itpc_spectrum[:, word_mask]))
        ratio = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0

        # Report the frequency of peak ITPC within each band (mean over channels)
        mean_sent_spec = np.mean(itpc_spectrum[:, sent_mask], axis=0)
        mean_word_spec = np.mean(itpc_spectrum[:, word_mask], axis=0)
        peak_sent_hz = (
            float(freqs[sent_mask][np.argmax(mean_sent_spec)]) if sent_mask.any() else self.TARGET_SENTENCE_FREQ
        )
        peak_word_hz = float(freqs[word_mask][np.argmax(mean_word_spec)]) if word_mask.any() else self.TARGET_WORD_FREQ

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_word": itpc_word_val,
            "ratio_sent_word": ratio,
            "freq_sentence_hz": peak_sent_hz,
            "freq_word_hz": peak_word_hz,
        }

    def extract_itpc_metrics(self, itpc_data: np.ndarray, freqs: Optional[np.ndarray] = None) -> dict:
        """
        Extract band-averaged ITPC metrics for sentence-rate and word-rate bands.

        Averages ITPC across all frequency bins within SENTENCE_BAND (0.05-0.08 Hz)
        and WORD_BAND (0.70-0.90 Hz), then averages across channels and time. This
        is more robust than single-bin extraction when ICA shifts power between
        adjacent bins, and matches the frequency band optimization strategy from
        the original research design.

        Args:
            itpc_data: Morlet ITPC array, shape (n_channels, n_freqs, n_times).
            freqs: Frequency axis. Defaults to class ITPC_FREQS.

        Returns:
            dict with itpc_sentence, itpc_word, ratio_sent_word, freq_sentence_hz,
            freq_word_hz (peak frequency within each band).
        """
        if freqs is None:
            freqs = self.ITPC_FREQS

        sent_mask = (freqs >= self.SENTENCE_BAND[0]) & (freqs <= self.SENTENCE_BAND[1])
        word_mask = (freqs >= self.WORD_BAND[0]) & (freqs <= self.WORD_BAND[1])

        itpc_sent_val = float(np.mean(itpc_data[:, sent_mask, :]))
        itpc_word_val = float(np.mean(itpc_data[:, word_mask, :]))
        ratio = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0

        # Report frequency of peak mean ITPC within each band
        mean_sent = np.mean(itpc_data[:, sent_mask, :], axis=(0, 2))  # (n_sent_bins,)
        mean_word = np.mean(itpc_data[:, word_mask, :], axis=(0, 2))  # (n_word_bins,)
        peak_sent_hz = float(freqs[sent_mask][np.argmax(mean_sent)]) if sent_mask.any() else self.TARGET_SENTENCE_FREQ
        peak_word_hz = float(freqs[word_mask][np.argmax(mean_word)]) if word_mask.any() else self.TARGET_WORD_FREQ

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_word": itpc_word_val,
            "ratio_sent_word": ratio,
            "freq_sentence_hz": peak_sent_hz,
            "freq_word_hz": peak_word_hz,
        }

    def compute_itpc_permutation_null(
        self,
        epochs: mne.Epochs,
        n_permutations: int = 1000,
        band: str = "sentence",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate null ITPC distribution via random-phase scrambling.

        For each surrogate, replaces DFT phases with independent uniform random
        draws while preserving power spectra. This destroys cross-trial phase
        consistency, providing the correct null for ITPC.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs (same as passed to compute_itpc_dft).
        n_permutations : int
            Number of surrogates.
        band : str
            "sentence" or "word" -- which band to average over.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        null_values : np.ndarray, shape (n_permutations,)
            Null ITPC values (one per surrogate).
        """
        band_limits = self.SENTENCE_BAND if band == "sentence" else self.WORD_BAND

        data = epochs.get_data()
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]

        n_fft = max(int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION)), n_times)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        band_mask = (freqs >= band_limits[0]) & (freqs <= band_limits[1])

        magnitudes = np.abs(np.fft.rfft(data, n=n_fft, axis=2))

        rng = np.random.default_rng(seed)
        null_values = np.empty(n_permutations)

        for i in range(n_permutations):
            random_phases = rng.uniform(0, 2 * np.pi, size=magnitudes.shape)
            scrambled = magnitudes * np.exp(1j * random_phases)
            unit_vecs = np.exp(1j * np.angle(scrambled))
            itpc_null = np.abs(np.mean(unit_vecs, axis=0))
            null_values[i] = float(np.mean(itpc_null[:, band_mask]))

        return null_values

    @staticmethod
    def compute_permutation_pvalue(observed: float, null_distribution: np.ndarray) -> float:
        """
        Compute one-sided p-value: proportion of null >= observed.

        Parameters
        ----------
        observed : float
            Observed ITPC value.
        null_distribution : np.ndarray
            Null ITPC values from compute_itpc_permutation_null.

        Returns
        -------
        float
            p-value in [0, 1].
        """
        return float(np.mean(null_distribution >= observed))

    def plot_itpc_results(self, itc, patient_id: str, output_dir: str, metrics: dict):
        """
        Generate and save enhanced ITPC plots (Topomap and TFR).

        Args:
            itc: MNE AverageTFR object.
            patient_id: Patient ID string.
            output_dir: Path to save outputs.
            metrics: Metrics dictionary from extract_itpc_metrics.
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_freq = metrics["freq_sentence_hz"]
        word_freq = metrics["freq_word_hz"]

        fig_topo, ax_topo = plt.subplots(1, 1, figsize=(10, 8))
        itc.plot_topomap(
            tmin=0,
            tmax=None,
            fmin=target_freq - 0.01,
            fmax=target_freq + 0.01,
            baseline=None,
            mode=None,
            axes=ax_topo,
            show=False,
            cmap="viridis",
            colorbar=True,
            vlim=(0, 0.3),
        )
        ax_topo.set_title(f"ITPC Topomap @ {target_freq:.3f} Hz\n{patient_id}", fontsize=14, fontweight="bold")
        topo_path = output_dir / f"{patient_id}_language_ITPC_topomap.png"
        fig_topo.savefig(topo_path, dpi=300, bbox_inches="tight")
        plt.close(fig_topo)

        fig_tfr, ax_tfr = plt.subplots(1, 1, figsize=(14, 8))
        itc.plot(
            baseline=None,
            mode=None,
            axes=ax_tfr,
            show=False,
            combine="mean",
            cmap="viridis",
            vlim=(0, 0.3),
            colorbar=True,
        )
        ax_tfr.axhline(
            y=target_freq, color="white", linestyle="--", linewidth=2, label=f"Sentence ({target_freq:.3f} Hz)"
        )
        ax_tfr.text(
            itc.times[0], target_freq, " Sentence", color="white", verticalalignment="bottom", fontweight="bold"
        )
        ax_tfr.axhline(y=word_freq, color="white", linestyle=":", linewidth=2, label=f"Word ({word_freq:.3f} Hz)")
        ax_tfr.text(itc.times[0], word_freq, " Word", color="white", verticalalignment="bottom", fontweight="bold")
        ax_tfr.set_title(f"ITPC Time-Frequency ({patient_id}) - Hemisphere Mean", fontsize=16)
        ax_tfr.set_xlabel("Time (s)", fontsize=12)
        ax_tfr.set_ylabel("Frequency (Hz)", fontsize=12)

        tfr_path = output_dir / f"{patient_id}_language_ITPC_tfr.png"
        fig_tfr.savefig(tfr_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tfr)
