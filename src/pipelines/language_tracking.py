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
    # 3 dB point at the first measurement bin.
    # We use a 25.0 Hz lowpass to safely cover up to beta rhythms before downsampling,
    # ensuring no high-frequency artifacts alias into the tracking bands.
    HIGHPASS_FREQ = 0.02
    LOWPASS_FREQ = 25.0

    # Downsampling target
    TARGET_SFREQ = 256.0

    # ITPC Constants
    # Target frequencies based on Sokoliuk 2021 methodology:
    TARGET_SENTENCE_FREQ = 0.78
    TARGET_PHRASE_FREQ = 1.56
    TARGET_WORD_FREQ = 3.125

    # Frequency bands for band-averaged ITPC extraction.
    # Band-averaging across the entrainment band (rather than a single bin) is
    # more robust to small ICA-induced power shifts between adjacent bins and
    # reflects the finite bandwidth of neural entrainment responses.
    # Note: 1 / 14.08s window length = 0.071 Hz bins exactly
    SENTENCE_BAND: tuple = (0.71, 0.85)
    PHRASE_BAND: tuple = (1.49, 1.63)
    WORD_BAND: tuple = (3.05, 3.20)
    SENTENCE_BAND_WIDTH_HZ: float = SENTENCE_BAND[1] - SENTENCE_BAND[0]
    PHRASE_BAND_WIDTH_HZ: float = PHRASE_BAND[1] - PHRASE_BAND[0]
    WORD_BAND_WIDTH_HZ: float = WORD_BAND[1] - WORD_BAND[0]

    # Epoch cropping: discard 2.28s from start to remove filter/ICA edge artifacts,
    # yielding a clean 13.08s analysis window.
    CROP_TMIN = 2.28
    CROP_TMAX = 15.36

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

        # 1. Compute Global ITPC (using all filtered channels for focus)
        logger.info(f"[{self.patient_id}] Computing Global DFT ITPC...")
        itpc_spectrum_global, dft_freqs = self.compute_itpc_dft(self.epochs)
        global_metrics = self.extract_itpc_metrics_dft(itpc_spectrum_global, dft_freqs)

        # 2. Compute Morlet ITPC
        logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
        itpc_data_morlet, _ = self.compute_itpc(self.epochs)
        morlet_metrics = self.extract_itpc_metrics(itpc_data_morlet)

        # 3. Compute Left Hemisphere ITPC
        logger.info(f"[{self.patient_id}] Computing LH ITPC...")
        lh_epochs = self.select_optimal_channels(self._epochs_filtered, focus="LH")
        itpc_spectrum_lh, _ = self.compute_itpc_dft(lh_epochs)
        lh_metrics = self.extract_itpc_metrics_dft(itpc_spectrum_lh, dft_freqs)

        # 4. Compute Right Hemisphere ITPC
        logger.info(f"[{self.patient_id}] Computing RH ITPC...")
        rh_epochs = self.select_optimal_channels(self._epochs_filtered, focus="RH")
        itpc_spectrum_rh, _ = self.compute_itpc_dft(rh_epochs)
        rh_metrics = self.extract_itpc_metrics_dft(itpc_spectrum_rh, dft_freqs)

        # 5. Chance-frequency bootstrap test for statistical significance
        n_permutations = kwargs.get("n_permutations", 1000)
        logger.info(
            f"[{self.patient_id}] \
                Running mathematical trial-level phase-scrambling permutation test ({n_permutations} surrogates)..."
        )
        null_sentence = self.compute_trial_shuffled_null_itpc(
            self._epochs_filtered, n_permutations, metric="sentence", seed=42
        )
        null_phrase = self.compute_trial_shuffled_null_itpc(
            self._epochs_filtered, n_permutations, metric="phrase", seed=43
        )
        null_word = self.compute_trial_shuffled_null_itpc(self._epochs_filtered, n_permutations, metric="word", seed=44)
        null_comp = self.compute_trial_shuffled_null_itpc(
            self._epochs_filtered, n_permutations, metric="comprehension", seed=45
        )

        p_sentence = self.compute_permutation_pvalue(global_metrics["itpc_sentence"], null_sentence)
        p_phrase = self.compute_permutation_pvalue(global_metrics["itpc_phrase"], null_phrase)
        p_word = self.compute_permutation_pvalue(global_metrics["itpc_word"], null_word)
        p_comprehension = self.compute_permutation_pvalue(global_metrics["itpc_comprehension_combined"], null_comp)

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
            "left_hem_itpc_sentence": lh_metrics["itpc_sentence"],
            "left_hem_itpc_phrase": lh_metrics["itpc_phrase"],
            "left_hem_itpc_word": lh_metrics["itpc_word"],
            "right_hem_itpc_sentence": rh_metrics["itpc_sentence"],
            "right_hem_itpc_phrase": rh_metrics["itpc_phrase"],
            "right_hem_itpc_word": rh_metrics["itpc_word"],
            "morlet_itpc_sentence": morlet_metrics.get("itpc_sentence"),
            "morlet_itpc_phrase": morlet_metrics.get("itpc_phrase"),
            "morlet_itpc_word": morlet_metrics.get("itpc_word"),
            "dft_p_sentence": p_sentence,
            "dft_p_phrase": p_phrase,
            "dft_p_word": p_word,
            "dft_p_comprehension": p_comprehension,
            "dft_n_permutations": n_permutations,
        }

        self.results = pd.DataFrame([result_dict])

        logger.info(
            f"Pipeline complete for {self.patient_id}. "
            f"Global Sentence ITPC: {result_dict['itpc_sentence']:.3f}, "
            f"LH Sentence ITPC: {lh_metrics['itpc_sentence']:.3f}, "
            f"RH Sentence ITPC: {rh_metrics['itpc_sentence']:.3f}"
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

            # Preprocess fresh epochs before analysis
            self._epochs_filtered = None
            self.preprocess()
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
        morlet_sent = row.get("morlet_itpc_sentence")
        morlet_word = row.get("morlet_itpc_word")
        morlet_ratio = morlet_sent / morlet_word if morlet_word else None
        dft_sent = row.get("itpc_sentence")
        dft_word = row.get("itpc_word")
        dft_ratio = dft_sent / dft_word if dft_word else None
        return {
            "patient_id": row.get("patient_id", ""),
            "focus": row.get("focus", ""),
            "morlet_ratio": morlet_ratio,
            "dft_ratio": dft_ratio,
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
        """Apply bandpass filtering, downsampling, and epoch cropping.
        Follows Sokoliuk et al. (2021) to eliminate edge artifacts by precise cropping.
        """
        logger.info(
            f"[{self.patient_id}] Filtering {self.HIGHPASS_FREQ}-{self.LOWPASS_FREQ}Hz "
            f"and downsampling to {self.TARGET_SFREQ}Hz"
        )

        # Determine actual sfreq safely
        current_sfreq = float(epochs.info["sfreq"])

        if current_sfreq > self.TARGET_SFREQ:
            epochs = epochs.copy().resample(self.TARGET_SFREQ, verbose=False)

        # Filter is applied across the full 17s extracted window
        epochs = epochs.copy().filter(
            l_freq=self.HIGHPASS_FREQ,
            h_freq=self.LOWPASS_FREQ,
            fir_design="firwin",
            phase="zero-double",  # No phase distortions
            verbose=False,
        )

        logger.info(f"[{self.patient_id}] Cropping epochs to {self.CROP_TMIN} - {self.CROP_TMAX}s")
        try:
            epochs.crop(tmin=self.CROP_TMIN, tmax=self.CROP_TMAX, verbose=False)
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

    def extract_itpc_metrics_dft(self, itpc_spectrum: np.ndarray, freqs: np.ndarray, channel_idx: int = None) -> dict:
        """
        Extract band-averaged ITPC for specific linguistic levels from the DFT spectrum.
        Matches the stimulus rates:
        - Sentence level: ~0.78 Hz
        - Phrase level: ~1.56 Hz
        - Word level: ~3.125 Hz

        Using a 14.08s window yields a frequency resolution of exactly 0.071 Hz bins.
        We extract the maximum response closest to our theoretical target.
        """
        if channel_idx is not None:
            spec = itpc_spectrum[channel_idx]
        else:
            # Average across channels first to find global peaks robustly
            spec = np.mean(itpc_spectrum, axis=0)

        # Helper function to find closest frequency bin and return ITPC
        def extract_closest_freq(target: float) -> tuple:
            idx = np.argmin(np.abs(freqs - target))
            return float(spec[idx]), float(freqs[idx])

        # Extract ITPC at specific linguistic response bins
        itpc_sent_val, peak_sent_hz = extract_closest_freq(self.TARGET_SENTENCE_FREQ)
        itpc_phrase_val, peak_phrase_hz = extract_closest_freq(self.TARGET_PHRASE_FREQ)
        itpc_word_val, peak_word_hz = extract_closest_freq(self.TARGET_WORD_FREQ)

        # Calculate combined metrics
        ratio_sw = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0
        ratio_sp = itpc_sent_val / itpc_phrase_val if itpc_phrase_val > 0 else 0.0
        itpc_comprehension_combined = (itpc_sent_val + itpc_phrase_val) / 2.0

        # Calculate bandpass-agnostic ratio metrics
        sent_density = itpc_sent_val / self.SENTENCE_BAND_WIDTH_HZ if self.SENTENCE_BAND_WIDTH_HZ > 0 else 0.0
        word_density = itpc_word_val / self.WORD_BAND_WIDTH_HZ if self.WORD_BAND_WIDTH_HZ > 0 else 0.0
        ratio_bw = sent_density / word_density if word_density > 0 else 0.0

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

        Averages ITPC across all frequency bins within SENTENCE_BAND, PHRASE_BAND,
        and WORD_BAND, then averages across channels and time. This
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

    def compute_trial_shuffled_null_itpc(
        self,
        epochs: mne.Epochs,
        n_permutations: int = 1000,
        metric: str = "word",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate null ITPC distribution via trial-level random phase scrambling.

        By adding a random phase offset (uniform [0, 2pi)) to each trial identically across
        all channels, we mathematically simulate circular-shifting the trials. This destroys
        stimulus-locked timing (true phase consistency) while preserving both the 1/f noise
        profile of the target frequency bin and the spatial covariance across electrodes.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs.
        n_permutations : int
            Number of surrogates.
        metric : str
            "sentence", "phrase", "word", or "comprehension"
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        null_values : np.ndarray, shape (n_permutations,)
            Null ITPC values.
        """
        rng = np.random.default_rng(seed)

        # Extract the true phase angles exactly as done in compute_itpc_dft
        data = epochs.get_data()
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]
        n_pad = int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION))
        n_fft = max(n_pad, n_times)

        # We only need the specific frequency indices.
        # rfftfreq matches what is used in compute_itpc_dft
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        def get_bin_idx(target_f):
            return np.argmin(np.abs(freqs - target_f))

        sent_idx = get_bin_idx(self.TARGET_SENTENCE_FREQ)
        phrase_idx = get_bin_idx(self.TARGET_PHRASE_FREQ)
        word_idx = get_bin_idx(self.TARGET_WORD_FREQ)

        # Compute DFT for the exact bins needed to save time
        # Unfortunately rfft doesn't let us pick bins, but doing it once per dataset is fast enough.
        spectra = np.fft.rfft(data, n=n_fft, axis=2)

        def get_surrogate_itpc(bin_idx):
            # Extract true phases for the specific bin
            # unit_vectors: shape (n_trials, n_channels)
            unit_vectors = np.exp(1j * np.angle(spectra[:, :, bin_idx]))

            # Generate random phase offsets per trial (identical across channels to preserve spatial covariance)
            # rand_phase: shape (n_permutations, n_trials, 1)
            rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))

            # Broadcast random phase across channels: (n_permutations, n_trials, n_channels)
            shifted_vectors = unit_vectors * np.exp(1j * rand_phase)

            # Calculate ITPC for each permutation
            # 1. Mean unit vector across trials: shape (n_permutations, n_channels)
            # 2. Magnitude (ITPC) per channel: np.abs
            # 3. Global average across channels: shape (n_permutations,)
            return np.mean(np.abs(np.mean(shifted_vectors, axis=1)), axis=1)

        if metric == "sentence":
            return get_surrogate_itpc(sent_idx)
        elif metric == "phrase":
            return get_surrogate_itpc(phrase_idx)
        elif metric == "word":
            return get_surrogate_itpc(word_idx)
        elif metric == "comprehension":
            # Comprehension is the unweighted average of sentence and phrase ITPC
            return (get_surrogate_itpc(sent_idx) + get_surrogate_itpc(phrase_idx)) / 2.0
        else:
            raise ValueError(f"Unknown metric '{metric}'")

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
