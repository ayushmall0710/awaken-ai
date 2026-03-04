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
    # 0.02 Hz highpass provides a 2.5x safety margin below the SENTENCE_BAND
    # lower edge (0.05 Hz). An IIR filter attenuates -3 dB at its cutoff
    # frequency; setting HIGHPASS_FREQ == SENTENCE_BAND[0] would place the
    # 3 dB point at the first measurement bin. At 0.02 Hz cutoff, attenuation
    # at 0.05 Hz is < 0.1 dB. 5.0 Hz lowpass keeps only the slow neural
    # envelope relevant for sentence-rate (0.065 Hz), phrase-rate (0.78 Hz), and
    # word-rate (3.1 Hz) ITPC analysis. Matches docs/language_tracking.md.
    HIGHPASS_FREQ = 0.02
    LOWPASS_FREQ = 5.0

    # Downsampling target
    TARGET_SFREQ = 256.0

    # ITPC Constants
    # Target frequencies based on Sokoliuk 2021 methodology:
    # Sentence-rate (~0.065 Hz), Phrase-rate (~0.78 Hz), and Word-rate (~3.1 Hz)
    TARGET_SENTENCE_FREQ = 0.065
    TARGET_PHRASE_FREQ = 0.78
    TARGET_WORD_FREQ = 3.1

    # Frequency bands for band-averaged ITPC extraction.
    # Band-averaging across the entrainment band (rather than a single bin) is
    # more robust to small ICA-induced power shifts between adjacent bins and
    # reflects the finite bandwidth of neural entrainment responses.
    # Bands are taken from the stimulus design:
    #   Sentence: 12 audio files over ~15.5s -> ~0.065 Hz, nominal band 0.05-0.08 Hz
    #   Phrase: 1 audio file every ~1.28s -> ~0.78 Hz, nominal band 0.70-0.85 Hz
    SENTENCE_BAND: tuple = (0.75, 0.81)
    PHRASE_BAND: tuple = (1.52, 1.60)
    WORD_BAND: tuple = (3.05, 3.20)
    SENTENCE_BAND_WIDTH_HZ: float = SENTENCE_BAND[1] - SENTENCE_BAND[0]
    PHRASE_BAND_WIDTH_HZ: float = PHRASE_BAND[1] - PHRASE_BAND[0]
    WORD_BAND_WIDTH_HZ: float = WORD_BAND[1] - WORD_BAND[0]

    ITPC_FREQS = np.logspace(np.log10(0.5), np.log10(5.0), num=60)
    ITPC_CYCLES = np.array([max(0.5, f * 2.0) for f in ITPC_FREQS])

    # Target frequency resolution for the zero-padded DFT.
    # Padding to 0.01 Hz resolution ensures the nearest DFT bin is within
    # 0.005 Hz of the target frequencies, eliminating the 4% bin-mismatch
    # artifact from raw epoch resolution (1/16s = 0.0625 Hz). This is
    # sufficient because ITPC is extracted by band-averaging across SENTENCE_BAND
    # and WORD_BAND, not by isolating a single bin.
    DFT_FREQ_RESOLUTION = 0.01

    def __init__(
        self,
        loader: Optional[UnifiedDataLoader] = None,
        focus: Union[str, Iterable[str]] = "LH",
        filter_signal: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the LanguageTrackingAnalysis.

        Args:
            loader: Optional UnifiedDataLoader instance.
            focus: Hemisphere focus ('LH', 'RH', or 'Clinical') or a custom iterable of channels.
            filter_signal: Whether to apply bandpass filtering.
            session_id: Optional specific session ID to load. If None, loads all sessions.
        """
        super().__init__(loader=loader)
        self.focus = focus
        self.filter_signal = filter_signal
        self.session_id = session_id
        self.epochs: Optional[mne.Epochs] = None
        self._epochs_filtered: Optional[mne.Epochs] = None

    def load(self) -> None:
        """Load and concatenate all language epochs for the patient."""
        sessions = [self.session_id] if self.session_id else self.loader.get_patient(self.patient_id).list_sessions()
        all_epochs = []

        for session_id in sessions:
            try:
                epochs = self.loader.load_clean_epochs(self.patient_id, session_id, trial_type="language")
                all_epochs.append(epochs)
            except FileNotFoundError:
                logger.warning(
                    f"No clean language epochs found for {self.patient_id} on {session_id}. Skipping session."
                )
                continue

        if not all_epochs:
            raise ValueError(f"No clean epochs found for {self.patient_id}. Run 'awakenai preprocess' first.")

        self.epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]

    def preprocess(self) -> None:
        """Apply optimization steps: bandpass filtering then channel selection."""
        if self.epochs is None:
            raise ValueError("Epochs not loaded. Call load() first.")

        if self.filter_signal:
            self.epochs = self.preprocess_signal(self.epochs)

        # Store pre-channel-selected epochs for lateralization analysis
        self._epochs_filtered = self.epochs.copy()
        self.epochs = self.select_optimal_channels(self.epochs, focus=self.focus)

        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            self.epochs.set_montage(montage, on_missing="warn")
        except Exception as e:
            logger.warning(f"Montage error for {self.patient_id}: {e}")

    def analyze(self, **kwargs) -> pd.DataFrame:
        """
        Compute ITPC and return results matching exactly the requested feature table.
        """
        if self.epochs is None:
            logger.info(f"[{self.patient_id}] Epochs not loaded. Calling load() and preprocess()...")
            self.load()
            self.preprocess()

        # NOTE: self._epochs_filtered contains all valid channels before LH/RH isolation
        if self._epochs_filtered is None:
            self._epochs_filtered = self.epochs.copy()

        # Determine target hemisphere based on focus
        focus_name = self.focus if isinstance(self.focus, str) else "Custom"
        lateral_focus = "RH" if focus_name == "RH" else "LH"

        # 1. Compute Global ITPC (using all filtered channels)
        logger.info(f"[{self.patient_id}] Computing Global DFT ITPC...")
        itpc_spectrum_global, dft_freqs = self.compute_itpc_dft(self._epochs_filtered)
        global_metrics = self.extract_itpc_metrics_dft(itpc_spectrum_global, dft_freqs)

        # 2. Compute Morlet ITPC
        logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
        itpc_data_morlet, _ = self.compute_itpc(self._epochs_filtered)
        morlet_metrics = self.extract_itpc_metrics(itpc_data_morlet)

        # 3. Compute Lateral Hemisphere ITPC
        logger.info(f"[{self.patient_id}] Computing {lateral_focus} ITPC...")
        lateral_epochs = self.select_optimal_channels(self._epochs_filtered, focus=lateral_focus)
        itpc_spectrum_lateral, _ = self.compute_itpc_dft(lateral_epochs)
        lateral_metrics = self.extract_itpc_metrics_dft(itpc_spectrum_lateral, dft_freqs)

        # Combine exact requested results: patient_id, n_trials, itpc_sentence,
        # itpc_phrase, itpc_word, itpc_comprehension_combined, focused_hem_itpc
        result_dict = {
            "patient_id": self.patient_id,
            "n_trials": len(self.epochs),
            "focus": focus_name,
            "itpc_sentence": global_metrics["itpc_sentence"],
            "itpc_phrase": global_metrics["itpc_phrase"],
            "itpc_word": global_metrics["itpc_word"],
            "itpc_comprehension_combined": global_metrics["itpc_comprehension_combined"],
            "left_hem_itpc": lateral_metrics["itpc_sentence"] if lateral_focus == "LH" else None,
            "right_hem_itpc": lateral_metrics["itpc_sentence"] if lateral_focus == "RH" else None,
            "morlet_itpc_sentence": morlet_metrics.get("itpc_sentence"),
            "morlet_itpc_phrase": morlet_metrics.get("itpc_phrase"),
            "morlet_itpc_word": morlet_metrics.get("itpc_word"),
        }

        self.results = pd.DataFrame([result_dict])

        logger.info(
            f"Pipeline complete for {self.patient_id}. "
            f"Global Sentence ITPC: {result_dict['itpc_sentence']:.3f}, "
            f"Global Phrase: {result_dict['itpc_phrase']:.3f}, "
            f"{lateral_focus} Sentence ITPC: {lateral_metrics['itpc_sentence']:.3f}"
        )
        return self.results

    def run_per_session(self, patient_id: str) -> pd.DataFrame:
        """
        Run ITPC analysis independently per recording session.

        Returns a DataFrame with one row per session, suitable for
        longitudinal trajectory visualization.

        Parameters
        ----------
        patient_id : str
            Patient identifier.

        Returns
        -------
        pd.DataFrame
            Per-session ITPC metrics with 'session_date' column.
        """
        sessions = self.loader.get_patient(patient_id).list_sessions()

        rows = []
        original_patient_id = getattr(self, "patient_id", patient_id)

        for date in sessions:
            try:
                self.patient_id = patient_id
                self.epochs = self.loader.load_clean_epochs(patient_id, date, trial_type="language")
            except FileNotFoundError:
                logger.warning(f"No clean epochs for {patient_id} on {date}, skipping.")
                continue

            # Clear cache to force preprocess to refresh on the new epochs
            self._epochs_filtered = None

            # Run the same analysis that produces global, lateral, and morlet metrics
            self.analyze()

            metrics = self.results.iloc[0].to_dict()
            metrics["session_date"] = date
            rows.append(metrics)

        self.patient_id = original_patient_id
        return pd.DataFrame(rows)

    def generate_summary(self) -> Any:
        """Generate summary of language tracking results."""
        if self.results is None or self.results.empty:
            return {}
        row = self.results.iloc[0]
        return {
            "patient_id": row.get("patient_id", ""),
            "focus": row.get("focus", ""),
            "morlet_ratio": row.get("morlet_ratio_sent_word", None),
            "dft_ratio": row.get("dft_ratio_sent_word", None),
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

    def preprocess_signal(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply bandpass filtering, downsampling, and epoch cropping."""
        logger.info(
            f"[{self.patient_id}] Filtering {self.HIGHPASS_FREQ}-{self.LOWPASS_FREQ}Hz "
            f"and downsampling to {self.TARGET_SFREQ}Hz"
        )
        epochs = epochs.copy().filter(
            l_freq=self.HIGHPASS_FREQ,
            h_freq=self.LOWPASS_FREQ,
            method="iir",
            verbose=False,
        )
        if epochs.info["sfreq"] != self.TARGET_SFREQ:
            epochs = epochs.resample(self.TARGET_SFREQ, verbose=False)

        # Baseline correction (using the initial part of the cropped signal if needed)
        # Note: We do this before cropping just as standard practice, but
        # actual "pre-stimulus baseline" of -1s to 0s is not present in these epochs.
        epochs.apply_baseline((None, None), verbose=False)

        # Crop exactly 1.28s from the start to yield a 14.08s window
        tmin_crop = 1.28
        tmax_crop = 15.36
        logger.info(f"[{self.patient_id}] Cropping epochs to {tmin_crop} - {tmax_crop}s")
        try:
            epochs.crop(tmin=tmin_crop, tmax=tmax_crop, verbose=False)
        except ValueError as e:
            logger.error(f"Failed to crop epochs: {e}. Current times: {epochs.times[0]} to {epochs.times[-1]}")
            raise

        return epochs

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
            n_jobs=-1,
            average=True,
        )
        return itc.data, itc

    def compute_itpc_dft(self, epochs: mne.Epochs):
        """
        Compute ITPC using the Discrete Fourier Transform (Sokoliuk 2021 method).

        For each trial and electrode, zero-pads the time series to achieve
        DFT_FREQ_RESOLUTION (0.01 Hz) frequency resolution, then computes the FFT,
        extracts per-bin phase, and averages unit phase vectors across trials.

        Zero-padding interpolates the DFT spectrum (Sinc interpolation) so that
        band selection correctly includes bins near the target frequencies (0.065 Hz
        sentence, 0.78 Hz phrase, 3.1 Hz word) rather than snapping to the nearest integer multiple
        of 1/epoch_duration_s. Without padding, 16s epochs yield 0.0625 Hz resolution
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
        phrase_mask = (freqs >= self.PHRASE_BAND[0]) & (freqs <= self.PHRASE_BAND[1])
        word_mask = (freqs >= self.WORD_BAND[0]) & (freqs <= self.WORD_BAND[1])

        # Mean over channels, then peak within band
        mean_sent_spec = np.mean(itpc_spectrum[:, sent_mask], axis=0)
        mean_phrase_spec = np.mean(itpc_spectrum[:, phrase_mask], axis=0)
        mean_word_spec = np.mean(itpc_spectrum[:, word_mask], axis=0)
        itpc_sent_val = float(np.max(mean_sent_spec)) if sent_mask.any() else 0.0
        itpc_phrase_val = float(np.max(mean_phrase_spec)) if phrase_mask.any() else 0.0
        itpc_word_val = float(np.max(mean_word_spec)) if word_mask.any() else 0.0
        ratio_sw = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0
        ratio_sp = itpc_sent_val / itpc_phrase_val if itpc_phrase_val > 0 else 0.0

        # Bandwidth-normalized ratio: compare spectral density rather than raw values
        sent_density = itpc_sent_val / self.SENTENCE_BAND_WIDTH_HZ if self.SENTENCE_BAND_WIDTH_HZ > 0 else 0.0
        word_density = itpc_word_val / self.WORD_BAND_WIDTH_HZ if self.WORD_BAND_WIDTH_HZ > 0 else 0.0
        ratio_bw = sent_density / word_density if word_density > 0 else 0.0

        # Report the frequency of peak ITPC within each band
        peak_sent_hz = (
            float(freqs[sent_mask][np.argmax(mean_sent_spec)]) if sent_mask.any() else self.TARGET_SENTENCE_FREQ
        )
        peak_phrase_hz = (
            float(freqs[phrase_mask][np.argmax(mean_phrase_spec)]) if phrase_mask.any() else self.TARGET_PHRASE_FREQ
        )
        peak_word_hz = float(freqs[word_mask][np.argmax(mean_word_spec)]) if word_mask.any() else self.TARGET_WORD_FREQ

        itpc_comprehension_combined = (itpc_sent_val + itpc_phrase_val) / 2.0

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_phrase": itpc_phrase_val,
            "itpc_word": itpc_word_val,
            "itpc_comprehension_combined": itpc_comprehension_combined,
            "ratio_sent_word": ratio_sw,
            "ratio_sent_phrase": ratio_sp,
            "ratio_bw_normalized": ratio_bw,
            "freq_sentence_hz": peak_sent_hz,
            "freq_phrase_hz": peak_phrase_hz,
            "freq_word_hz": peak_word_hz,
        }

    def extract_itpc_metrics(self, itpc_data: np.ndarray, freqs: Optional[np.ndarray] = None) -> dict:
        """
        Extract band-averaged ITPC metrics for sentence-rate and word-rate bands.

        Averages ITPC across all frequency bins within SENTENCE_BAND (0.05-0.08 Hz)
        and WORD_BAND (0.70-0.85 Hz), then averages across channels and time. This
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
        phrase_mask = (freqs >= self.PHRASE_BAND[0]) & (freqs <= self.PHRASE_BAND[1])
        word_mask = (freqs >= self.WORD_BAND[0]) & (freqs <= self.WORD_BAND[1])

        itpc_sent_val = float(np.mean(itpc_data[:, sent_mask, :]))
        itpc_phrase_val = float(np.mean(itpc_data[:, phrase_mask, :]))
        itpc_word_val = float(np.mean(itpc_data[:, word_mask, :]))
        ratio_sw = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0
        ratio_sp = itpc_sent_val / itpc_phrase_val if itpc_phrase_val > 0 else 0.0

        # Bandwidth-normalized ratio
        sent_density = itpc_sent_val / self.SENTENCE_BAND_WIDTH_HZ if self.SENTENCE_BAND_WIDTH_HZ > 0 else 0.0
        word_density = itpc_word_val / self.WORD_BAND_WIDTH_HZ if self.WORD_BAND_WIDTH_HZ > 0 else 0.0
        ratio_bw = sent_density / word_density if word_density > 0 else 0.0

        # Report frequency of peak mean ITPC within each band
        mean_sent = np.mean(itpc_data[:, sent_mask, :], axis=(0, 2))
        mean_phrase = np.mean(itpc_data[:, phrase_mask, :], axis=(0, 2))
        mean_word = np.mean(itpc_data[:, word_mask, :], axis=(0, 2))
        peak_sent_hz = float(freqs[sent_mask][np.argmax(mean_sent)]) if sent_mask.any() else self.TARGET_SENTENCE_FREQ
        peak_phrase_hz = (
            float(freqs[phrase_mask][np.argmax(mean_phrase)]) if phrase_mask.any() else self.TARGET_PHRASE_FREQ
        )
        peak_word_hz = float(freqs[word_mask][np.argmax(mean_word)]) if word_mask.any() else self.TARGET_WORD_FREQ

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_phrase": itpc_phrase_val,
            "itpc_word": itpc_word_val,
            "ratio_sent_word": ratio_sw,
            "ratio_sent_phrase": ratio_sp,
            "ratio_bw_normalized": ratio_bw,
            "freq_sentence_hz": peak_sent_hz,
            "freq_phrase_hz": peak_phrase_hz,
            "freq_word_hz": peak_word_hz,
        }

    @staticmethod
    def compute_lateralization_index(lh_itpc: float, rh_itpc: float) -> float:
        """
        Compute Lateralization Index.

        LI = (LH - RH) / (LH + RH).
        Positive = left-lateralized (expected for language in right-handed patients).

        Parameters
        ----------
        lh_itpc : float
            Left hemisphere ITPC value.
        rh_itpc : float
            Right hemisphere ITPC value.

        Returns
        -------
        float
            Lateralization index in [-1, 1]. Returns 0.0 if both inputs are zero.
        """
        denom = lh_itpc + rh_itpc
        return (lh_itpc - rh_itpc) / denom if denom > 0 else 0.0

    def compute_hemisphere_itpc(self, focus: str) -> dict:
        """
        Compute DFT ITPC metrics for a given hemisphere using stored filtered epochs.

        Parameters
        ----------
        focus : str
            "LH" or "RH".

        Returns
        -------
        dict
            Same keys as extract_itpc_metrics_dft.
        """
        if self._epochs_filtered is None:
            raise ValueError("Call preprocess() before compute_hemisphere_itpc().")
        epochs_hemi = self.select_optimal_channels(self._epochs_filtered, focus=focus)
        itpc_spectrum, freqs = self.compute_itpc_dft(epochs_hemi)
        return self.extract_itpc_metrics_dft(itpc_spectrum, freqs)

    def compute_itpc_permutation_null(
        self,
        epochs: mne.Epochs,
        n_permutations: int = 1000,
        band: str = "sentence",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate null ITPC distribution via random-phase scrambling.

        For each surrogate, draws independent uniform random phases for each
        trial/channel/frequency-bin combination, which destroys cross-trial phase
        consistency and provides the correct null for ITPC.

        The unit vector of a complex number with random phase is simply
        exp(i * random_phase), so no real-data FFT or magnitude computation is
        needed. Random phases are generated only for the band of interest rather
        than the full spectrum, which reduces memory usage by the ratio of band
        bins to total FFT bins.

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
            Null ITPC values (one per surrogate). Each value is computed as
            mean ITPC over channels, then peak over band bins -- matching
            the estimator used in extract_itpc_metrics_dft.
        """
        if band == "sentence":
            band_limits = self.SENTENCE_BAND
        elif band == "phrase":
            band_limits = self.PHRASE_BAND
        else:
            band_limits = self.WORD_BAND

        data = epochs.get_data()
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]

        n_fft = max(int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION)), n_times)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        band_mask = (freqs >= band_limits[0]) & (freqs <= band_limits[1])
        n_band_bins = int(band_mask.sum())

        rng = np.random.default_rng(seed)

        # Vectorized: draw all permutations at once -- no Python loop needed.
        # Shape: (n_permutations, n_trials, n_channels, n_band_bins)
        random_phases = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, n_channels, n_band_bins))
        unit_vecs = np.exp(1j * random_phases)
        # Mean over trials (axis=1) -> (n_permutations, n_channels, n_band_bins)
        itpc_null = np.abs(np.mean(unit_vecs, axis=1))
        # Mean over channels (axis=1) -> (n_permutations, n_band_bins)
        channel_mean = np.mean(itpc_null, axis=1)
        # Peak over band bins (axis=1) -> (n_permutations,)
        return np.max(channel_mean, axis=1)

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
