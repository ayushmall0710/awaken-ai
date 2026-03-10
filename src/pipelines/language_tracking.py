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
from typing import Optional

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
    # We use a proportional bandwidth (±10%) to naturally accommodate the logarithmic
    # widening of biological frequency responses at higher bands.
    BANDWIDTH_PERCENT = 0.10

    SENTENCE_BAND: tuple = (
        TARGET_SENTENCE_FREQ * (1 - BANDWIDTH_PERCENT),
        TARGET_SENTENCE_FREQ * (1 + BANDWIDTH_PERCENT),
    )
    PHRASE_BAND: tuple = (
        TARGET_PHRASE_FREQ * (1 - BANDWIDTH_PERCENT),
        TARGET_PHRASE_FREQ * (1 + BANDWIDTH_PERCENT),
    )
    WORD_BAND: tuple = (
        TARGET_WORD_FREQ * (1 - BANDWIDTH_PERCENT),
        TARGET_WORD_FREQ * (1 + BANDWIDTH_PERCENT),
    )

    SENTENCE_BAND_WIDTH_HZ: float = SENTENCE_BAND[1] - SENTENCE_BAND[0]
    PHRASE_BAND_WIDTH_HZ: float = PHRASE_BAND[1] - PHRASE_BAND[0]
    WORD_BAND_WIDTH_HZ: float = WORD_BAND[1] - WORD_BAND[0]

    # Epoch cropping: discard 2.28s from start to remove filter/ICA edge artifacts,
    # yielding a clean 13.08s analysis window.
    CROP_TMIN = 2.28
    CROP_TMAX = 16.36

    ITPC_FREQS = np.logspace(np.log10(0.5), np.log10(5.0), num=60)
    # n_cycles = 2f for the TFR visualisation pass: time resolution ≈ 2/f seconds.
    # At 0.78 Hz: window ≈ 2.56 s — enough temporal detail to see onset/offset structure.
    # At 3.125 Hz: window ≈ 0.64 s — resolves individual word-rate fluctuations.
    # The phase-extraction pass (_compute_morlet_target_phases) uses n_cycles = 5f independently
    # for better frequency selectivity, since it immediately time-averages and discards the time axis.
    ITPC_CYCLES = np.array([max(0.5, f * 2.0) for f in ITPC_FREQS])

    # Target frequency resolution for the zero-padded DFT.
    # Padding to 0.01 Hz resolution ensures the nearest DFT bin is within
    # 0.005 Hz of the target frequencies, eliminating the 4% bin-mismatch
    # artifact from raw epoch resolution (1/16s = 0.0625 Hz). This is
    # sufficient because ITPC is extracted by band-averaging across SENTENCE_BAND
    # and WORD_BAND, not by isolating a single bin.
    DFT_FREQ_RESOLUTION = 0.01

    # Morlet target frequency axis-2 index mapping (ascending frequency, matches ITPC_FREQS convention):
    # [0]=sentence (0.78 Hz), [1]=phrase (1.56 Hz), [2]=word (3.125 Hz)
    _MORLET_FREQ_IDX: dict = {"sentence": 0, "phrase": 1, "word": 2}

    def __init__(
        self,
        loader: Optional[UnifiedDataLoader] = None,
        filter_signal: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the LanguageTrackingAnalysis.

        Args:
            loader: Optional UnifiedDataLoader instance.
            filter_signal: Whether to apply bandpass filtering.
            session_id: Optional specific session ID to load. If None, loads all sessions.
        """
        super().__init__(loader=loader)
        self.filter_signal = filter_signal
        self.session_id = session_id
        self.epochs: Optional[mne.Epochs] = None
        self._epochs_filtered: Optional[mne.Epochs] = None
        self._dft_spectrum_full: Optional[np.ndarray] = None
        self._dft_freqs: Optional[np.ndarray] = None
        self._dft_ch_names: Optional[list] = None
        self._dft_info = None
        self._morlet_itc = None
        self._morlet_phases: Optional[np.ndarray] = None  # (n_trials, n_channels, 3) phases at [word, phrase, sentence]

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
        """Apply bandpass filtering and store filtered epochs."""
        if self.epochs is None:
            raise ValueError("Epochs not loaded. Call load() first.")
        if self.filter_signal:
            self.epochs = self._preprocess_signal(self.epochs)
        self._epochs_filtered = self.epochs.copy()
        try:
            montage = mne.channels.make_standard_montage("standard_1020")
            self._epochs_filtered.set_montage(montage, on_missing="warn")
        except Exception as e:
            logger.warning(f"Montage error for {self.patient_id}: {e}")

    def analyze(self, n_permutations: int = 1000) -> pd.DataFrame:
        """
        Compute per-focus ITPC and permutation p-values using a two-phase architecture.

        Phase 1 computes per-channel ITPC and null distributions for all available clinical
        channels. Phase 2 aggregates by focus channel subsets, running spatial cluster
        permutation to identify the optimal focus.

        Parameters
        ----------
        n_permutations : int, default 1000
            Surrogates for trial-shuffled null distributions.

        Returns
        -------
        pd.DataFrame
            Long-format, shape (4, 20): one row per focus ("clinical", "lh", "rh", "optimal").
            The "optimal" row contains NaN metrics when no significant spatial cluster is found.
        """
        if self.epochs is None:
            logger.info(f"[{self.patient_id}] Epochs not loaded. Calling load()...")
            self.load()
        if self._epochs_filtered is None:
            logger.info(f"[{self.patient_id}] Filtered epochs not ready. Calling preprocess()...")
            self.preprocess()

        # =========================================================================
        # Phase 1: Per-channel computation (shared foundation for all focuses)
        # =========================================================================

        # 1a. Pick available clinical channels
        clinical_epochs = self._pick_channel_subset(self._epochs_filtered, config.CLINICAL_20)
        clinical_ch_names = clinical_epochs.ch_names

        # 1b. Per-channel DFT ITPC — (n_ch, n_freqs)
        logger.info(f"[{self.patient_id}] Computing per-channel DFT ITPC (clinical channels)...")
        per_ch_itpc_dft, dft_freqs = self._compute_itpc_dft(clinical_epochs)
        self._dft_spectrum_full = per_ch_itpc_dft
        self._dft_freqs = dft_freqs
        self._dft_ch_names = clinical_ch_names
        self._dft_info = clinical_epochs.info

        # 1c. Per-channel Morlet ITPC — (n_ch, 3)
        logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
        _, itc_obj = self._compute_itpc(clinical_epochs)
        self._morlet_itc = itc_obj
        self._morlet_phases = self._compute_morlet_target_phases(clinical_epochs)
        per_ch_itpc_morlet = self._compute_per_channel_itpc_morlet(self._morlet_phases)

        # 1d. Per-channel null distributions — shared across all focuses
        logger.info(f"[{self.patient_id}] Computing per-channel null distributions ({n_permutations} surrogates)...")
        per_ch_null_dft = self._compute_per_channel_null_dft(clinical_epochs, n_permutations, seed=42)
        per_ch_null_morlet = self._compute_per_channel_null_morlet(n_permutations, seed=46)

        # =========================================================================
        # Phase 2: Focus aggregation
        # =========================================================================

        # 2a. Select optimal channels via spatial cluster permutation
        logger.info(f"[{self.patient_id}] Selecting optimal channels via spatial cluster permutation...")
        optimal_channels = self._select_optimal_channels(
            morlet_phases=self._morlet_phases,
            ch_names=clinical_ch_names,
            info=clinical_epochs.info,
            n_permutations=n_permutations,
        )
        if optimal_channels:
            logger.info(f"[{self.patient_id}] Optimal focus: {optimal_channels}")
        else:
            logger.info(f"[{self.patient_id}] No significant spatial cluster found; optimal focus is empty.")

        # 2b. Resolve all focus → channel-list mappings
        focuses = self._resolve_focuses(clinical_ch_names, optimal_channels)

        # 2c. Build one row per focus
        shared_args = dict(
            clinical_ch_names=clinical_ch_names,
            per_ch_itpc_dft=per_ch_itpc_dft,
            dft_freqs=dft_freqs,
            per_ch_itpc_morlet=per_ch_itpc_morlet,
            per_ch_null_dft=per_ch_null_dft,
            per_ch_null_morlet=per_ch_null_morlet,
        )
        rows = [self._build_focus_row(focus=name, channels=chs, **shared_args) for name, chs in focuses.items()]

        self.results = pd.DataFrame(rows)

        clinical_row = self.results[self.results["focus"] == "clinical"].iloc[0]
        logger.info(
            f"Pipeline complete for {self.patient_id}. "
            f"Clinical sentence ITPC: {clinical_row['itpc_sentence']:.3f}, "
            f"Optimal channels: {optimal_channels}"
        )
        return self.results

    def _run_per_session(self, patient_id: str, n_permutations: int = 1000) -> pd.DataFrame:
        """
        Run ITPC analysis independently per recording session.

        Returns a DataFrame with one row per session using the clinical focus metrics,
        suitable for longitudinal trajectory visualization.

        Parameters
        ----------
        patient_id : str
            Patient identifier.
        n_permutations : int, default 1000
            Surrogates passed to ``analyze()`` for null distributions.

        Returns
        -------
        pd.DataFrame
            Per-session clinical-focus ITPC metrics with "session_date" column.
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

            self._epochs_filtered = None
            self._morlet_phases = None
            self._morlet_itc = None
            self.preprocess()
            self.analyze(n_permutations=n_permutations)

            clinical_rows = self.results[self.results["focus"] == "clinical"]
            if len(clinical_rows) == 0:
                continue
            metrics = clinical_rows.iloc[0].to_dict()
            metrics["session_date"] = date
            rows.append(metrics)

        self.patient_id = original_patient_id
        return pd.DataFrame(rows)

    def generate_summary(self) -> dict:
        """
        Compute derived metrics from long-format results.

        Extracts lateralization indices from lh/rh focus rows and ratio_cognitive_acoustic
        from the clinical focus row.

        Returns
        -------
        dict
            Keys: patient_id, lateralization_index_word, lateralization_index_phrase,
            lateralization_index_sentence, lateralization_index_comprehension,
            ratio_cognitive_acoustic, morlet_ratio.
        """
        if self.results is None or self.results.empty:
            return {}

        def get_focus(name: str):
            rows = self.results[self.results["focus"] == name]
            return rows.iloc[0] if len(rows) > 0 else None

        lh_row = get_focus("lh")
        rh_row = get_focus("rh")
        clinical_row = get_focus("clinical")

        li_word = li_phrase = li_sentence = li_comp = None
        if lh_row is not None and rh_row is not None:
            li_word = self._compute_lateralization_index(lh_row["itpc_word"], rh_row["itpc_word"])
            li_phrase = self._compute_lateralization_index(lh_row["itpc_phrase"], rh_row["itpc_phrase"])
            li_sentence = self._compute_lateralization_index(lh_row["itpc_sentence"], rh_row["itpc_sentence"])
            lh_comp = (lh_row["itpc_sentence"] + lh_row["itpc_phrase"]) / 2.0
            rh_comp = (rh_row["itpc_sentence"] + rh_row["itpc_phrase"]) / 2.0
            li_comp = self._compute_lateralization_index(lh_comp, rh_comp)

        ratio_cog_ac = morlet_ratio = None
        if clinical_row is not None:
            word = clinical_row["itpc_word"]
            comp = clinical_row["itpc_comprehension"]
            ratio_cog_ac = comp / word if word != 0 else 0.0
            morlet_sent = clinical_row.get("morlet_itpc_sentence")
            morlet_word = clinical_row.get("morlet_itpc_word")
            morlet_ratio = morlet_sent / morlet_word if morlet_word is not None and morlet_word != 0.0 else None

        return {
            "patient_id": self.patient_id,
            "lateralization_index_word": li_word,
            "lateralization_index_phrase": li_phrase,
            "lateralization_index_sentence": li_sentence,
            "lateralization_index_comprehension": li_comp,
            "ratio_cognitive_acoustic": ratio_cog_ac,
            "morlet_ratio": morlet_ratio,
        }

    def _pick_channel_subset(self, epochs: mne.Epochs, channels: list) -> mne.Epochs:
        """
        Pick a named channel subset from epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Source epochs.
        channels : list of str
            Target channel names to pick.

        Returns
        -------
        mne.Epochs
            Copy with picked channels. Returns original if no valid channels found.
        """
        available = epochs.ch_names
        normalized_names = normalize_channel_names(available)
        clean_map = {clean.upper(): orig for orig, clean in zip(available, normalized_names)}

        picks = []
        missing = []
        for target in channels:
            target_upper = target.upper()
            if target in available:
                picks.append(target)
            elif target_upper in clean_map:
                picks.append(clean_map[target_upper])
            else:
                missing.append(target)

        if missing:
            logger.warning(f"Missing channels: {missing}")

        if not picks:
            logger.error("No valid channels found. Returning original epochs.")
            return epochs

        logger.info(f"Picked {len(picks)} channels.")
        return epochs.copy().pick(picks)

    def _resolve_focuses(self, available_ch_names: list, optimal_channels: list) -> dict:
        """
        Build focus-to-channel-list mapping for all four focuses.

        Each of clinical/lh/rh is intersected with available channels. Optimal is passed
        through as-is (already a subset of clinical channels from cluster permutation).

        Parameters
        ----------
        available_ch_names : list of str
            Channel names present in the clinical subset of the data.
        optimal_channels : list of str
            Channels selected by spatial cluster permutation. May be empty.

        Returns
        -------
        dict
            Keys "clinical", "lh", "rh", "optimal". Values are lists of channel names.
        """
        available = set(available_ch_names)

        def intersect(target_list: list) -> list:
            return [ch for ch in target_list if ch in available]

        return {
            "clinical": intersect(config.CLINICAL_20),
            "lh": intersect(config.LH_FOCUS_CHANNELS),
            "rh": intersect(config.RH_FOCUS_CHANNELS),
            "optimal": optimal_channels,
        }

    def _select_optimal_channels(
        self,
        morlet_phases: np.ndarray,
        ch_names: list,
        info: mne.Info,
        n_permutations: int = 1000,
        seed: int = 42,
        alpha: float = 0.05,
    ) -> list:
        """
        Select optimal channels via spatial cluster permutation on comprehension-frequency phase coherence.

        Projects per-trial phases onto their mean direction using cos(phase - mean_phase),
        yielding per-trial scalars testable against zero with a 1-sample t-test. Channels
        belonging to spatial clusters significant at alpha are returned as the optimal focus.

        Parameters
        ----------
        morlet_phases : np.ndarray, shape (n_trials, n_channels, 3)
            Phase angles at target frequencies. Axis-2: [0]=sentence, [1]=phrase, [2]=word.
        ch_names : list of str
            Channel names corresponding to axis-1.
        info : mne.Info
            MNE info with channel positions (needed for adjacency matrix).
        n_permutations : int
            Surrogates for cluster permutation.
        seed : int
            Random seed.
        alpha : float
            Significance threshold for cluster p-values.

        Returns
        -------
        list of str
            Channels belonging to significant clusters. Returns [] if none survive or
            if adjacency/permutation computation fails.
        """
        from mne.stats import permutation_cluster_1samp_test

        sent_idx = self._MORLET_FREQ_IDX["sentence"]
        phrase_idx = self._MORLET_FREQ_IDX["phrase"]

        sent_phases = morlet_phases[:, :, sent_idx]
        phrase_phases = morlet_phases[:, :, phrase_idx]

        mean_sent = np.angle(np.mean(np.exp(1j * sent_phases), axis=0))
        mean_phrase = np.angle(np.mean(np.exp(1j * phrase_phases), axis=0))

        X_sent = np.cos(sent_phases - mean_sent[np.newaxis, :])
        X_phrase = np.cos(phrase_phases - mean_phrase[np.newaxis, :])
        X = (X_sent + X_phrase) / 2.0

        try:
            adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
        except Exception as e:
            logger.warning(
                f"[{self.patient_id}] Could not compute channel adjacency: {e}. Returning empty optimal focus."
            )
            return []

        try:
            _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
                X,
                adjacency=adjacency,
                threshold={"start": 0, "step": 0.2},
                n_permutations=n_permutations,
                seed=seed,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"[{self.patient_id}] Cluster permutation failed: {e}. Returning empty optimal focus.")
            return []

        optimal = []
        for cluster, pv in zip(clusters, cluster_pv):
            if pv < alpha:
                ch_mask = np.asarray(cluster, dtype=bool).ravel()
                optimal.extend(ch_names[i] for i in np.where(ch_mask)[0])

        return list(dict.fromkeys(optimal))

    def _preprocess_signal(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply bandpass filtering, downsampling, and epoch cropping.
        Follows Sokoliuk et al. (2021) to eliminate edge artifacts by precise cropping.
        """
        logger.info(
            f"[{self.patient_id}] Filtering {self.HIGHPASS_FREQ}-{self.LOWPASS_FREQ}Hz "
            f"and downsampling to {self.TARGET_SFREQ}Hz"
        )

        epochs_processed = epochs.copy()
        if not epochs_processed.preload:
            epochs_processed.load_data()

        if current_sfreq > self.TARGET_SFREQ:
            epochs = epochs.copy().resample(self.TARGET_SFREQ, verbose=False)

        # Filter is applied across the full 17s extracted window
        epochs_processed.filter(
            l_freq=self.HIGHPASS_FREQ,
            h_freq=self.LOWPASS_FREQ,
            method="iir",
            iir_params=None,  # Defaults to Butterworth 4th order zero-phase
            verbose=False,
        )

        if current_sfreq > self.cfg.target_sfreq:
            epochs_processed.resample(self.cfg.target_sfreq, verbose=False)

        logger.info(f"[{self.patient_id}] Cropping epochs to {self.CROP_TMIN} - {self.CROP_TMAX}s")
        try:
            epochs.crop(tmin=self.CROP_TMIN, tmax=self.CROP_TMAX, verbose=False)
        except ValueError as e:
            logger.error(
                f"Failed to crop epochs: {e}. "
                f"Current times: {epochs_processed.times[0]} to {epochs_processed.times[-1]}"
            )
            raise

        return epochs_processed

    def _compute_itpc(
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

    def _compute_itpc_dft(self, epochs: mne.Epochs):
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

    def _extract_itpc_metrics_dft(
        self, itpc_spectrum: np.ndarray, freqs: np.ndarray, channel_idx: Optional[int] = None
    ) -> dict:
        """
        Extract ITPC for specific linguistic levels from the DFT spectrum.

        Note: This method currently extracts the ITPC from the single closest frequency
        bin to the theoretical target, whereas the Morlet path averages across a band.

        Matches the stimulus rates:
        - Sentence level: ~0.78 Hz
        - Phrase level: ~1.56 Hz
        - Word level: ~3.125 Hz
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

    def _extract_itpc_metrics(self, itpc_data: np.ndarray, freqs: Optional[np.ndarray] = None) -> dict:
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
    def _compute_lateralization_index(lh_itpc: float, rh_itpc: float) -> float:
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

    def _compute_hemisphere_itpc(self, focus: str) -> dict:
        """
        Compute DFT ITPC metrics for a given hemisphere using stored filtered epochs.

        Parameters
        ----------
        focus : str
            "LH" or "RH".

        Returns
        -------
        dict
            Same keys as _extract_itpc_metrics_dft.
        """
        if self._epochs_filtered is None:
            raise ValueError("Call preprocess() before _compute_hemisphere_itpc().")
        channels = config.LH_FOCUS_CHANNELS if focus == "LH" else config.RH_FOCUS_CHANNELS
        epochs_hemi = self._pick_channel_subset(self._epochs_filtered, channels)
        itpc_spectrum, freqs = self._compute_itpc_dft(epochs_hemi)
        return self._extract_itpc_metrics_dft(itpc_spectrum, freqs)

    def _compute_morlet_target_phases(self, epochs: mne.Epochs) -> np.ndarray:
        """
        Compute per-trial Morlet phase angles at the three target frequency bins.

        Runs a second tfr_morlet call with average=False, output='complex',
        restricted to only three frequencies (word, phrase, sentence) to avoid
        storing the full 60-frequency complex array.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs.

        Returns
        -------
        np.ndarray, shape (n_trials, n_channels, 3)
            Phase angles (radians) at target frequencies in ascending order.
            Axis-2 order: [0]=sentence (0.78 Hz), [1]=phrase (1.56 Hz), [2]=word (3.125 Hz).
        """
        from mne.time_frequency import tfr_morlet

        target_freqs = np.array([self.TARGET_SENTENCE_FREQ, self.TARGET_PHRASE_FREQ, self.TARGET_WORD_FREQ])
        # n_cycles = 5f (higher than ITPC_CYCLES = 2f used for TFR visualisation).
        # Frequency resolution Δf ≈ f/5 vs f/2 for the TFR pass.
        # Justified because this pass immediately time-averages the complex output — the time
        # axis is discarded — so temporal resolution is irrelevant and frequency selectivity
        # should be maximised to reduce cross-frequency contamination in the phase estimate.
        n_cycles = np.array([max(0.5, f * 5.0) for f in target_freqs])

        epoch_tfr = tfr_morlet(
            epochs,
            freqs=target_freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            output="complex",
            average=False,
            n_jobs=-1,
        )
        # epoch_tfr.data: (n_trials, n_channels, 3, n_times)
        complex_data = epoch_tfr.data
        # Time-average complex values before taking angle: equivalent to extracting the dominant phase
        # of the epoch, analogous to a single-bin DFT. This summarizes the per-trial phase for the
        # permutation test without requiring a reference time point.
        mean_complex = np.mean(complex_data, axis=-1)  # (n_trials, n_channels, 3)
        return np.angle(mean_complex)

    def _extract_morlet_observed_itpc(self) -> dict:
        """
        Compute observed Morlet ITPC from stored ``_morlet_phases``.

        Uses identical math to ``_compute_morlet_null_itpc`` / ``_compute_surrogate_itpc``
        (time-average complex first, take angle, then |mean_trials(exp(i·phase))|),
        so the observed statistic and its permutation null are computed on the same
        quantity and p-values are properly calibrated.

        This replaces the previous approach of band-averaging ``_morlet_itc.data``
        (an AverageTFR), which computed mean_t(ITPC(t,f)) — a different quantity
        from the null that caused slightly conservative, uncalibrated p-values.

        Returns
        -------
        dict
            Keys: ``itpc_word``, ``itpc_phrase``, ``itpc_sentence``.
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call analyze() first.")

        phases = self._morlet_phases  # (n_trials, n_channels, 3)

        def _itpc_at(freq_idx: int) -> float:
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])  # (n_trials, n_channels)
            # mean across trials per channel, magnitude, then mean across channels
            return float(np.mean(np.abs(np.mean(unit_vectors, axis=0))))

        return {
            "itpc_word": _itpc_at(self._MORLET_FREQ_IDX["word"]),
            "itpc_phrase": _itpc_at(self._MORLET_FREQ_IDX["phrase"]),
            "itpc_sentence": _itpc_at(self._MORLET_FREQ_IDX["sentence"]),
        }

    @staticmethod
    def _compute_per_channel_itpc_morlet(phases: np.ndarray) -> np.ndarray:
        """
        Compute per-channel Morlet ITPC from trial phase angles.

        Parameters
        ----------
        phases : np.ndarray, shape (n_trials, n_channels, 3)
            Phase angles at target frequencies.
            Axis-2 order: [0]=sentence, [1]=phrase, [2]=word.

        Returns
        -------
        np.ndarray, shape (n_channels, 3)
            ITPC per channel per frequency.
        """
        unit_vectors = np.exp(1j * phases)  # (n_trials, n_channels, 3)
        return np.abs(np.mean(unit_vectors, axis=0))  # (n_channels, 3)

    def _compute_trial_shuffled_null_itpc(
        self,
        epochs: Optional[mne.Epochs],
        n_permutations: int = 1000,
        metric: str = "word",
        seed: int = 42,
        method: str = "dft",
    ) -> np.ndarray:
        """
        Generate null ITPC distribution via trial-level random phase scrambling.

        By adding a random phase offset (uniform [0, 2pi)) to each trial identically across
        all channels, we mathematically simulate circular-shifting the trials. This destroys
        stimulus-locked timing (true phase consistency) while preserving both the 1/f noise
        profile of the target frequency bin and the spatial covariance across electrodes.

        Parameters
        ----------
        epochs : mne.Epochs or None
            Preprocessed epochs. Pass ``None`` when ``method="morlet"``.
        n_permutations : int
            Number of surrogates.
        metric : str
            "sentence", "phrase", "word", or "comprehension"
        seed : int
            Random seed for reproducibility.
        method : str, optional
            Analysis back-end: "dft" (default) or "morlet".
            When "morlet", ``epochs`` is ignored and ``_morlet_phases`` must
            already be populated (call ``analyze()`` first).

        Returns
        -------
        null_values : np.ndarray, shape (n_permutations,)
            Null ITPC values.
        """
        rng = np.random.default_rng(seed)

        if method == "morlet":
            return self._compute_morlet_null_itpc(n_permutations, metric, rng)
        elif method != "dft":
            raise ValueError(f"Unknown method '{method}'. Use 'dft' or 'morlet'.")

        # Extract the true phase angles exactly as done in _compute_itpc_dft
        data = epochs.get_data()
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]
        n_pad = int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION))
        n_fft = max(n_pad, n_times)

        # We only need the specific frequency indices.
        # rfftfreq matches what is used in _compute_itpc_dft
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
            unit_vectors = np.exp(1j * np.angle(spectra[:, :, bin_idx]))
            return self._compute_surrogate_itpc(unit_vectors, n_permutations, rng)

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

    def _compute_per_channel_null_dft(
        self,
        epochs: mne.Epochs,
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> dict:
        """
        Generate per-channel DFT null distributions via trial-level random phase scrambling.

        Produces one null array per target frequency at per-channel granularity, so
        Phase 2 can subset by any focus channel combination without re-running surrogates.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed clinical-channel epochs.
        n_permutations : int
            Number of surrogates.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys "sentence", "phrase", "word". Each value shape (n_permutations, n_channels).
        """
        rng = np.random.default_rng(seed)
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        n_trials, _n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]
        n_pad = int(np.ceil(sfreq / self.DFT_FREQ_RESOLUTION))
        n_fft = max(n_pad, n_times)
        spectra = np.fft.rfft(data, n=n_fft, axis=2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        # Same random phase offsets reused across all 3 target frequencies.
        # Shape (n_permutations, n_trials, 1): one offset per trial, broadcast over channels.
        rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))

        result = {}
        for freq_name, target_f in (
            ("sentence", self.TARGET_SENTENCE_FREQ),
            ("phrase", self.TARGET_PHRASE_FREQ),
            ("word", self.TARGET_WORD_FREQ),
        ):
            bin_idx = int(np.argmin(np.abs(freqs - target_f)))
            unit_vectors = np.exp(1j * np.angle(spectra[:, :, bin_idx]))  # (n_trials, n_channels)
            shifted = unit_vectors * np.exp(1j * rand_phase)  # (n_permutations, n_trials, n_channels)
            result[freq_name] = np.abs(np.mean(shifted, axis=1))  # (n_permutations, n_channels)

        return result

    def _compute_per_channel_null_morlet(
        self,
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> dict:
        """
        Generate per-channel Morlet null distributions via trial-level random phase scrambling.

        Uses stored ``_morlet_phases``. Returns per-channel null ITPC for each target frequency.

        Parameters
        ----------
        n_permutations : int
            Number of surrogates.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Keys "sentence", "phrase", "word". Each value shape (n_permutations, n_channels).
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call _compute_morlet_target_phases() first.")

        rng = np.random.default_rng(seed)
        phases = self._morlet_phases  # (n_trials, n_channels, 3)
        n_trials = phases.shape[0]
        rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))

        result = {}
        for freq_name, freq_idx in self._MORLET_FREQ_IDX.items():
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])  # (n_trials, n_channels)
            shifted = unit_vectors * np.exp(1j * rand_phase)  # (n_permutations, n_trials, n_channels)
            result[freq_name] = np.abs(np.mean(shifted, axis=1))  # (n_permutations, n_channels)

        return result

    def _compute_morlet_null_itpc(
        self,
        n_permutations: int,
        metric: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate null Morlet ITPC distribution via trial-level random phase scrambling.

        Uses stored ``_morlet_phases`` (set by ``_compute_morlet_target_phases``).
        Axis-2 order: [0]=sentence, [1]=phrase, [2]=word.

        Parameters
        ----------
        n_permutations : int
            Number of surrogates.
        metric : str
            "word", "phrase", "sentence", or "comprehension".
        rng : np.random.Generator
            Seeded random generator.

        Returns
        -------
        np.ndarray, shape (n_permutations,)
            Null ITPC values.
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call analyze() before running Morlet permutation tests.")

        phases = self._morlet_phases  # (n_trials, n_channels, 3)

        def surrogate_itpc(freq_idx: int) -> np.ndarray:
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])  # (n_trials, n_channels)
            return self._compute_surrogate_itpc(unit_vectors, n_permutations, rng)

        if metric in self._MORLET_FREQ_IDX:
            return surrogate_itpc(self._MORLET_FREQ_IDX[metric])
        elif metric == "comprehension":
            return (
                surrogate_itpc(self._MORLET_FREQ_IDX["sentence"]) + surrogate_itpc(self._MORLET_FREQ_IDX["phrase"])
            ) / 2.0
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    @staticmethod
    def _compute_surrogate_itpc(
        unit_vectors: np.ndarray,
        n_permutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate null ITPC surrogates from unit vectors via random phase shuffling.

        Parameters
        ----------
        unit_vectors : np.ndarray, shape (n_trials, n_channels)
            Complex unit vectors on the unit circle (exp(i*phase)).
        n_permutations : int
            Number of surrogates.
        rng : np.random.Generator
            Seeded random generator.

        Returns
        -------
        np.ndarray, shape (n_permutations,)
        """
        n_trials = unit_vectors.shape[0]
        # Identical random phase offset per trial across channels preserves spatial covariance
        rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))
        shifted = unit_vectors * np.exp(1j * rand_phase)
        return np.mean(np.abs(np.mean(shifted, axis=1)), axis=1)

    @staticmethod
    def _compute_permutation_pvalue(observed: float, null_distribution: np.ndarray) -> float:
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

    def _compute_focus_pvalue(
        self,
        per_ch_null: dict,
        ch_indices: list,
        observed: dict,
    ) -> dict:
        """
        Compute permutation p-values for a focus by subsetting per-channel null distributions.

        Parameters
        ----------
        per_ch_null : dict
            Keys "sentence", "phrase", "word". Each value shape (n_surrogates, n_channels).
        ch_indices : list of int
            Indices of the focus channels within the clinical channel array.
        observed : dict
            Observed ITPC values with keys "sentence", "phrase", "word".

        Returns
        -------
        dict
            Keys "p_sentence", "p_phrase", "p_word", "p_comprehension".
        """
        null_sent = np.mean(per_ch_null["sentence"][:, ch_indices], axis=-1)  # (n_surr,)
        null_phrase = np.mean(per_ch_null["phrase"][:, ch_indices], axis=-1)
        null_word = np.mean(per_ch_null["word"][:, ch_indices], axis=-1)
        null_comp = (null_sent + null_phrase) / 2.0
        obs_comp = (observed["sentence"] + observed["phrase"]) / 2.0

        return {
            "p_sentence": float(np.mean(null_sent >= observed["sentence"])),
            "p_phrase": float(np.mean(null_phrase >= observed["phrase"])),
            "p_word": float(np.mean(null_word >= observed["word"])),
            "p_comprehension": float(np.mean(null_comp >= obs_comp)),
        }

    def _build_focus_row(
        self,
        focus: str,
        channels: list,
        clinical_ch_names: list,
        per_ch_itpc_dft: np.ndarray,
        dft_freqs: np.ndarray,
        per_ch_itpc_morlet: np.ndarray,
        per_ch_null_dft: dict,
        per_ch_null_morlet: dict,
    ) -> dict:
        """
        Assemble one output row for a given focus.

        Parameters
        ----------
        focus : str
            Focus name: "clinical", "lh", "rh", or "optimal".
        channels : list of str
            Channel names for this focus. Empty list produces NaN metrics.
        clinical_ch_names : list of str
            Ordered names of the clinical channel array (used for index mapping).
        per_ch_itpc_dft : np.ndarray, shape (n_clinical_ch, n_freqs)
        dft_freqs : np.ndarray
        per_ch_itpc_morlet : np.ndarray, shape (n_clinical_ch, 3)
            Axis-1 order: [0]=sentence, [1]=phrase, [2]=word (matches _MORLET_FREQ_IDX).
        per_ch_null_dft : dict
            "sentence"/"phrase"/"word" -> (n_surrogates, n_clinical_ch).
        per_ch_null_morlet : dict
            Same structure as per_ch_null_dft.

        Returns
        -------
        dict
            One row with all output schema columns.
        """
        nan = float("nan")
        base = {
            "patient_id": self.patient_id,
            "n_trials": len(self._epochs_filtered),
            "focus": focus,
            "channels": channels,
        }
        nan_metrics = {
            "itpc_word": nan,
            "itpc_phrase": nan,
            "itpc_sentence": nan,
            "itpc_comprehension": nan,
            "morlet_itpc_word": nan,
            "morlet_itpc_phrase": nan,
            "morlet_itpc_sentence": nan,
            "morlet_itpc_comprehension": nan,
            "dft_p_word": nan,
            "dft_p_phrase": nan,
            "dft_p_sentence": nan,
            "dft_p_comprehension": nan,
            "morlet_p_word": nan,
            "morlet_p_phrase": nan,
            "morlet_p_sentence": nan,
            "morlet_p_comprehension": nan,
        }

        if not channels:
            return {**base, **nan_metrics}

        ch_to_idx = {ch: i for i, ch in enumerate(clinical_ch_names)}
        ch_indices = [ch_to_idx[ch] for ch in channels if ch in ch_to_idx]

        # Guard: _resolve_focuses guarantees focus channels are always in clinical_ch_names,
        # but protect against empty index lists for callers that bypass _resolve_focuses.
        if not ch_indices:
            logger.warning(f"Focus '{focus}' channels not found in clinical array; returning NaN row.")
            return {**base, **nan_metrics}

        # DFT ITPC: _extract_itpc_metrics_dft averages across channels (axis 0)
        dft_metrics = self._extract_itpc_metrics_dft(per_ch_itpc_dft[ch_indices, :], dft_freqs)
        obs_dft = {
            "sentence": dft_metrics["itpc_sentence"],
            "phrase": dft_metrics["itpc_phrase"],
            "word": dft_metrics["itpc_word"],
        }

        # Morlet ITPC: average focus channels then index by frequency
        focus_morlet = np.mean(per_ch_itpc_morlet[ch_indices, :], axis=0)  # (3,)
        obs_morlet = {
            "sentence": float(focus_morlet[self._MORLET_FREQ_IDX["sentence"]]),
            "phrase": float(focus_morlet[self._MORLET_FREQ_IDX["phrase"]]),
            "word": float(focus_morlet[self._MORLET_FREQ_IDX["word"]]),
        }

        dft_pvals = self._compute_focus_pvalue(per_ch_null_dft, ch_indices, obs_dft)
        morlet_pvals = self._compute_focus_pvalue(per_ch_null_morlet, ch_indices, obs_morlet)

        return {
            **base,
            "itpc_word": dft_metrics["itpc_word"],
            "itpc_phrase": dft_metrics["itpc_phrase"],
            "itpc_sentence": dft_metrics["itpc_sentence"],
            "itpc_comprehension": (dft_metrics["itpc_sentence"] + dft_metrics["itpc_phrase"]) / 2.0,
            "morlet_itpc_word": obs_morlet["word"],
            "morlet_itpc_phrase": obs_morlet["phrase"],
            "morlet_itpc_sentence": obs_morlet["sentence"],
            "morlet_itpc_comprehension": (obs_morlet["sentence"] + obs_morlet["phrase"]) / 2.0,
            "dft_p_word": dft_pvals["p_word"],
            "dft_p_phrase": dft_pvals["p_phrase"],
            "dft_p_sentence": dft_pvals["p_sentence"],
            "dft_p_comprehension": dft_pvals["p_comprehension"],
            "morlet_p_word": morlet_pvals["p_word"],
            "morlet_p_phrase": morlet_pvals["p_phrase"],
            "morlet_p_sentence": morlet_pvals["p_sentence"],
            "morlet_p_comprehension": morlet_pvals["p_comprehension"],
        }
