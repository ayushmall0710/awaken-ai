"""
Language Tracking Pipeline for ENG-05.

This module provides the LanguageTrackingAnalysis class to isolate language
trials, apply specific filtering, and select optimal electrodes for language
tracking analysis. It calculates Inter-Trial Phase Coherence (ITPC) using
Morlet wavelets and DFT approaches.

References:
    - docs/language_tracking.md: "The Language Tracking Paradigm"
      & "Optimization Strategies"
    - tasks/ENG-05.md: Pipeline design and implementation details.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.pipelines.base import BasePipeline
from src.utils.signal_processing import normalize_channel_names, select_optimal_channels

logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig:
    """Configuration for Language Tracking Paradigm."""

    # Filter constants
    highpass_freq: float = 0.02
    lowpass_freq: float = 25.0
    target_sfreq: float = 256.0

    # ITPC target frequencies
    target_sentence_freq: float = 0.78
    target_phrase_freq: float = 1.56
    target_word_freq: float = 3.125
    bandwidth_percent: float = 0.10

    # Derived bands (calculated at init)
    sentence_band: Tuple[float, float] = field(init=False)
    phrase_band: Tuple[float, float] = field(init=False)
    word_band: Tuple[float, float] = field(init=False)

    # Epoch cropping
    crop_tmin: float = 2.28
    crop_tmax: float = 16.36

    # Analysis settings
    dft_freq_resolution: float = 0.01
    itpc_freqs: np.ndarray = field(default_factory=lambda: np.logspace(np.log10(0.5), np.log10(5.0), num=60))

    def __post_init__(self):
        def get_band(target):
            return (target * (1 - self.bandwidth_percent), target * (1 + self.bandwidth_percent))

        self.sentence_band = get_band(self.target_sentence_freq)
        self.phrase_band = get_band(self.target_phrase_freq)
        self.word_band = get_band(self.target_word_freq)

    @property
    def sentence_band_width_hz(self) -> float:
        """
        Width of the sentence frequency band in Hz.

        Returns
        -------
        float
            Bandwidth in Hz.
        """
        return self.sentence_band[1] - self.sentence_band[0]

    @property
    def phrase_band_width_hz(self) -> float:
        """
        Width of the phrase frequency band in Hz.

        Returns
        -------
        float
            Bandwidth in Hz.
        """
        return self.phrase_band[1] - self.phrase_band[0]

    @property
    def word_band_width_hz(self) -> float:
        """
        Width of the word frequency band in Hz.

        Returns
        -------
        float
            Bandwidth in Hz.
        """
        return self.word_band[1] - self.word_band[0]


class ITPCProcessor:
    """Service layer for ITPC mathematical operations."""

    @staticmethod
    def compute_dft_itpc(data: np.ndarray, sfreq: float, resolution: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ITPC using Discrete Fourier Transform.

        Parameters
        ----------
        data : np.ndarray
            Epoch data in shape (n_trials, n_channels, n_times).
        sfreq : float
            Sampling frequency in Hz.
        resolution : float, optional
            Frequency resolution for DFT in Hz. Defaults to 0.01.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (itpc_spectrum, freqs). itpc_spectrum shape: (n_channels, n_freqs).
        """
        n_trials, n_channels, n_times = data.shape
        n_pad = int(np.ceil(sfreq / resolution))
        n_fft = max(n_pad, n_times)

        spectra = np.fft.rfft(data, n=n_fft, axis=2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        unit_vectors = np.exp(1j * np.angle(spectra))
        itpc_spectrum = np.abs(np.mean(unit_vectors, axis=0))
        return itpc_spectrum, freqs

    @staticmethod
    def compute_morlet_itpc(
        epochs: mne.Epochs,
        freqs: np.ndarray,
        n_cycles: np.ndarray,
        average: bool = True,
    ) -> Tuple[np.ndarray, Any]:
        """
        Compute ITPC using Morlet wavelets.

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs to analyze.
        freqs : np.ndarray
            Frequencies of interest.
        n_cycles : np.ndarray
            Number of cycles per frequency.
        average : bool, optional
            Whether to average across trials. Defaults to True.

        Returns
        -------
        Tuple[np.ndarray, Any]
            (itpc_data, itc_obj). itpc_data shape depends on average.
        """
        from mne.time_frequency import tfr_morlet

        itc_obj = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            decim=1,
            n_jobs=-1,
            average=average,
        )
        # When average=True, itc_obj is an AverageTFR
        itpc_data = itc_obj[1].data if isinstance(itc_obj, (list, tuple)) else itc_obj.data
        return itpc_data, itc_obj


class PermutationEngine:
    """Utility for generating trial-shuffled null distributions."""

    @staticmethod
    def generate_null_distribution(
        unit_vectors: np.ndarray,
        n_permutations: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate null ITPC surrogates from unit vectors via random phase
        shuffling. Preserves spatial covariance by adding identical phase
        offset per trial across channels.

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
        np.ndarray, shape (n_permutations, n_channels)
            Null ITPC distribution.
        """
        n_trials, n_channels = unit_vectors.shape
        # (n_permutations, n_trials, 1) broadcast over channels
        rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))
        shifted = unit_vectors * np.exp(1j * rand_phase)
        return np.abs(np.mean(shifted, axis=1))


class LanguageTrackingAnalysis(BasePipeline):
    """
    Pipeline for Language Tracking Paradigm data.

    Coordinates data loading, channel selection, filtering, and ITPC computation.

    Attributes:
        loader (UnifiedDataLoader): Instance of UnifiedDataLoader.
        cfg (LanguageConfig): Configuration for the pipeline.
    """

    # Focus channel definitions
    CLINICAL_20 = config.CLINICAL_20
    LH_FOCUS_CHANNELS = config.LH_FOCUS_CHANNELS
    RH_FOCUS_CHANNELS = config.RH_FOCUS_CHANNELS

    # Morlet target frequency axis-2 index mapping (ascending frequency,
    # matches itpc_freqs convention):
    # [0]=sentence (0.78 Hz), [1]=phrase (1.56 Hz), [2]=word (3.125 Hz)
    _MORLET_FREQ_IDX: dict = {"sentence": 0, "phrase": 1, "word": 2}

    def __init__(
        self,
        loader: Optional[UnifiedDataLoader] = None,
        filter_signal: bool = True,
        session_id: Optional[str] = None,
        config: Optional[LanguageConfig] = None,
    ):
        """
        Initialize the LanguageTrackingAnalysis.

        Parameters
        ----------
        loader : Optional[UnifiedDataLoader], optional
            Optional UnifiedDataLoader instance. Defaults to None.
        filter_signal : bool, optional
            Whether to apply bandpass filtering. Defaults to True.
        session_id : Optional[str], optional
            Optional specific session ID to load. If None, loads all sessions.
            Defaults to None.
        config : Optional[LanguageConfig], optional
            Optional LanguageConfig instance. Defaults to None.
        """
        super().__init__(loader=loader)
        self.cfg = config or LanguageConfig()
        self.filter_signal = filter_signal
        self.session_id = session_id
        self.epochs: Optional[mne.Epochs] = None
        self._epochs_filtered: Optional[mne.Epochs] = None
        self._dft_spectrum_full: Optional[np.ndarray] = None
        self._dft_freqs: Optional[np.ndarray] = None
        self._dft_ch_names: Optional[list] = None
        self._dft_info = None
        self._morlet_itc = None
        # (n_trials, n_channels, 3) phases at [word, phrase, sentence]
        self._morlet_phases: Optional[np.ndarray] = None

    def load(self) -> None:
        """
        Load and concatenate all language epochs for the patient.

        Returns
        -------
        None
        """
        sessions = [self.session_id] if self.session_id else self.loader.get_patient(self.patient_id).list_session_ids()
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
            raise ValueError(
                f"No clean epochs found for {self.patient_id}. Run 'awakenai setup {self.patient_id}' first."
            )

        self.epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]

    def preprocess(self) -> None:
        """
        Apply bandpass filtering and store filtered epochs.

        Returns
        -------
        None
        """
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
        Compute per-focus ITPC and permutation p-values using a two-phase
        architecture.

        Phase 1 computes per-channel ITPC and null distributions for all
        available clinical channels. Phase 2 aggregates by focus channel
        subsets, running spatial cluster permutation to identify the
        optimal focus.

        Parameters
        ----------
        n_permutations : int, optional
            Number of permutations for null distribution. Defaults to 1000.

        Returns
        -------
        pd.DataFrame
            ITPC results for each focus.
        """
        # Ensure data is ready (BasePipeline.run() should handle this)
        if self.epochs is None:
            self.load()
        if self._epochs_filtered is None:
            self.preprocess()

        # Phase 1: Per-channel computation (shared foundation for all focuses)
        # 1a. Pick clinical subset and compute core metrics
        clinical_epochs = self._pick_channel_subset(self._epochs_filtered, config.CLINICAL_20)
        clinical_ch_names = clinical_epochs.ch_names

        # 1b. Per-channel ITPC
        per_ch_itpc_dft, dft_freqs = self._compute_itpc_dft(clinical_epochs)
        self._dft_spectrum_full = per_ch_itpc_dft
        self._dft_freqs = dft_freqs
        self._dft_ch_names = clinical_ch_names
        self._dft_info = clinical_epochs.info

        _, itc_obj = self._compute_itpc(clinical_epochs)
        self._morlet_itc = itc_obj
        self._morlet_phases = self._compute_morlet_target_phases(clinical_epochs)
        per_ch_itpc_morlet = self._compute_per_channel_itpc_morlet(self._morlet_phases)

        # 1c. Per-channel null distributions
        per_ch_null_dft = self._compute_per_channel_null_dft(clinical_epochs, n_permutations, seed=42)
        per_ch_null_morlet = self._compute_per_channel_null_morlet(n_permutations, seed=46)

        # Phase 2: Focus aggregation
        # 2a. Select optimal channels via spatial cluster permutation
        optimal_channels = self._select_optimal_channels(
            morlet_phases=self._morlet_phases,
            ch_names=clinical_ch_names,
            info=clinical_epochs.info,
            n_permutations=n_permutations,
        )

        # 2b. Resolve focuses and build results
        focuses = self._resolve_focuses(clinical_ch_names, optimal_channels)
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
        return self.results

    def _run_per_session(self, patient_id: str, n_permutations: int = 1000) -> pd.DataFrame:
        """
        Run ITPC analysis independently per recording session.
        """
        sessions = self.loader.get_patient(patient_id).list_sessions()
        rows = []

        for date in sessions:
            try:
                # Use standard run() template for each session
                self.run(patient_id, session_id=None, n_permutations=n_permutations)

                clinical_rows = self.results[self.results["focus"] == "clinical"]
                if len(clinical_rows) == 0:
                    continue
                metrics = clinical_rows.iloc[0].to_dict()
                metrics["session_date"] = date
                rows.append(metrics)
            except Exception as e:
                logger.warning(f"Session {date} failed for {patient_id}: {e}")
                continue

        return pd.DataFrame(rows)

    def generate_summary(self) -> dict:
        """
        Compute derived metrics from long-format results.

        Extracts lateralization indices from lh/rh focus rows and
        ratio_cognitive_acoustic from the clinical focus row.

        Returns
        -------
        dict
            Keys: patient_id, lateralization_index_word,
            lateralization_index_phrase, lateralization_index_sentence,
            lateralization_index_comprehension, ratio_cognitive_acoustic,
            morlet_ratio.
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
        side_metrics = {}

        for side, row in (("lh", lh_row), ("rh", rh_row)):
            if row is None:
                continue
            for m in ["word", "phrase", "sentence", "comprehension"]:
                side_metrics[f"{side}_itpc_{m}"] = row.get(f"itpc_{m}")
                side_metrics[f"{side}_p_{m}"] = row.get(f"dft_p_{m}")
                side_metrics[f"{side}_morlet_itpc_{m}"] = row.get(f"morlet_itpc_{m}")
                side_metrics[f"{side}_morlet_p_{m}"] = row.get(f"morlet_p_{m}")

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
            **side_metrics,
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
            Copy with picked channels. Returns original if no valid channels
            found.
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

        Each of clinical/lh/rh is intersected with available channels.
        Optimal is passed through as-is (already a subset of clinical
        channels from cluster permutation).

        Parameters
        ----------
        available_ch_names : list of str
            Channel names present in the clinical subset of the data.
        optimal_channels : list of str
            Channels selected by spatial cluster permutation. May be empty.

        Returns
        -------
        dict
            Keys "clinical", "lh", "rh", "optimal". Values are lists of
            channel names.
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
        Select optimal channels via spatial cluster permutation on
        comprehension-frequency phase coherence.
        """
        # 1. Compute comprehension ITPC (average of sentence and phrase)
        # from already-computed DFT spectrum
        itpc_dft = self._dft_spectrum_full
        freqs = self._dft_freqs
        sent_f_idx = np.argmin(np.abs(freqs - self.cfg.target_sentence_freq))
        phrase_f_idx = np.argmin(np.abs(freqs - self.cfg.target_phrase_freq))
        itpc_comp = (itpc_dft[:, sent_f_idx] + itpc_dft[:, phrase_f_idx]) / 2.0

        # 2. Extract phases for sentence and phrase only
        # (first two indices of axis-2)
        target_phases = morlet_phases[:, :, :2]

        # 3. Call utility
        return select_optimal_channels(
            morlet_phases=target_phases,
            ch_names=ch_names,
            info=info,
            itpc_comp=itpc_comp,
            n_permutations=n_permutations,
            alpha=alpha,
            seed=seed,
        )

    def _preprocess_signal(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply bandpass filtering, downsampling, and epoch cropping.
        Follows Sokoliuk et al. (2021) to eliminate edge artifacts by
        precise cropping.
        """
        logger.info(
            f"[{self.patient_id}] Filtering "
            f"{self.cfg.highpass_freq}-{self.cfg.lowpass_freq}Hz "
            f"and downsampling to {self.cfg.target_sfreq}Hz"
        )

        epochs_processed = epochs.copy()
        if not epochs_processed.preload:
            epochs_processed.load_data()

        # Determine actual sfreq safely
        current_sfreq = float(epochs_processed.info["sfreq"])

        # Filter is applied across the full 17s extracted window
        epochs_processed.filter(
            l_freq=self.cfg.highpass_freq,
            h_freq=self.cfg.lowpass_freq,
            method="iir",
            iir_params=None,  # Defaults to Butterworth 4th order zero-phase
            verbose=False,
        )

        if current_sfreq > self.cfg.target_sfreq:
            epochs_processed.resample(self.cfg.target_sfreq, verbose=False)

        logger.info(f"[{self.patient_id}] Cropping epochs to {self.cfg.crop_tmin} - {self.cfg.crop_tmax}s")
        try:
            epochs_processed.crop(tmin=self.cfg.crop_tmin, tmax=self.cfg.crop_tmax, verbose=False)
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
        """
        if freqs is None:
            freqs = self.cfg.itpc_freqs
        if n_cycles is None:
            # n_cycles = 2f for the TFR visualisation pass: time resolution ≈ 2/f seconds.
            n_cycles = np.array([max(0.5, f * 2.0) for f in freqs])

        logger.info(f"Computing TFR and ITPC ({freqs[0]:.2f}-{freqs[-1]:.2f} Hz)...")
        itpc_data, itc_obj = ITPCProcessor.compute_morlet_itpc(epochs, freqs=freqs, n_cycles=n_cycles)
        return itpc_data, itc_obj

    def _compute_itpc_dft(self, epochs: mne.Epochs):
        """
        Compute ITPC using the Discrete Fourier Transform (Sokoliuk 2021 method).
        """
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        sfreq = epochs.info["sfreq"]

        itpc_spectrum, freqs = ITPCProcessor.compute_dft_itpc(data, sfreq, resolution=self.cfg.dft_freq_resolution)

        logger.info(
            f"DFT ITPC computed: {data.shape[0]} trials, {data.shape[1]} channels, "
            f"resolution={self.cfg.dft_freq_resolution} Hz"
        )
        return itpc_spectrum, freqs

    def _extract_itpc_metrics_dft(
        self, itpc_spectrum: np.ndarray, freqs: np.ndarray, channel_idx: Optional[int] = None
    ) -> dict:
        """
        Extract ITPC for specific linguistic levels from the DFT spectrum.
        """
        if channel_idx is not None:
            spec = itpc_spectrum[channel_idx]
        else:
            spec = np.mean(itpc_spectrum, axis=0)

        def extract_closest_freq(target: float) -> tuple:
            idx = np.argmin(np.abs(freqs - target))
            return float(spec[idx]), float(freqs[idx])

        itpc_sent_val, peak_sent_hz = extract_closest_freq(self.cfg.target_sentence_freq)
        itpc_phrase_val, peak_phrase_hz = extract_closest_freq(self.cfg.target_phrase_freq)
        itpc_word_val, peak_word_hz = extract_closest_freq(self.cfg.target_word_freq)

        ratio_sw = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0
        ratio_sp = itpc_sent_val / itpc_phrase_val if itpc_phrase_val > 0 else 0.0
        itpc_comprehension = (itpc_sent_val + itpc_phrase_val) / 2.0

        sent_density = itpc_sent_val / self.cfg.sentence_band_width_hz if self.cfg.sentence_band_width_hz > 0 else 0.0
        word_density = itpc_word_val / self.cfg.word_band_width_hz if self.cfg.word_band_width_hz > 0 else 0.0
        ratio_bw = sent_density / word_density if word_density > 0 else 0.0

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_phrase": itpc_phrase_val,
            "itpc_word": itpc_word_val,
            "itpc_comprehension": itpc_comprehension,
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
        """
        if freqs is None:
            freqs = self.cfg.itpc_freqs

        sent_mask = (freqs >= self.cfg.sentence_band[0]) & (freqs <= self.cfg.sentence_band[1])
        phrase_mask = (freqs >= self.cfg.phrase_band[0]) & (freqs <= self.cfg.phrase_band[1])
        word_mask = (freqs >= self.cfg.word_band[0]) & (freqs <= self.cfg.word_band[1])

        itpc_sent_val = float(np.mean(itpc_data[:, sent_mask, :]))
        itpc_phrase_val = float(np.mean(itpc_data[:, phrase_mask, :]))
        itpc_word_val = float(np.mean(itpc_data[:, word_mask, :]))
        ratio_sw = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0
        ratio_sp = itpc_sent_val / itpc_phrase_val if itpc_phrase_val > 0 else 0.0

        sent_density = itpc_sent_val / self.cfg.sentence_band_width_hz if self.cfg.sentence_band_width_hz > 0 else 0.0
        word_density = itpc_word_val / self.cfg.word_band_width_hz if self.cfg.word_band_width_hz > 0 else 0.0
        ratio_bw = sent_density / word_density if word_density > 0 else 0.0

        mean_sent = np.mean(itpc_data[:, sent_mask, :], axis=(0, 2))
        mean_phrase = np.mean(itpc_data[:, phrase_mask, :], axis=(0, 2))
        mean_word = np.mean(itpc_data[:, word_mask, :], axis=(0, 2))
        peak_sent_hz = (
            float(freqs[sent_mask][np.argmax(mean_sent)]) if sent_mask.any() else self.cfg.target_sentence_freq
        )
        peak_phrase_hz = (
            float(freqs[phrase_mask][np.argmax(mean_phrase)]) if phrase_mask.any() else self.cfg.target_phrase_freq
        )
        peak_word_hz = float(freqs[word_mask][np.argmax(mean_word)]) if word_mask.any() else self.cfg.target_word_freq

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
        """
        from mne.time_frequency import tfr_morlet

        target_freqs = np.array([self.cfg.target_sentence_freq, self.cfg.target_phrase_freq, self.cfg.target_word_freq])
        # n_cycles = 5f (higher than ITPC_CYCLES = 2f used for TFR visualisation).
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
        mean_complex = np.mean(complex_data, axis=-1)  # (n_trials, n_channels, 3)
        return np.angle(mean_complex)

    def _extract_morlet_observed_itpc(self) -> dict:
        """
        Compute observed Morlet ITPC from stored ``_morlet_phases``.

        Uses identical math to ``_compute_morlet_null_itpc`` /
        ``_compute_surrogate_itpc`` (time-average complex first, take angle,
        then |mean_trials(exp(i·phase))|), so the observed statistic and
        its permutation null are computed on the same quantity and p-values
        are properly calibrated.

        This replaces the previous approach of band-averaging
        ``_morlet_itc.data`` (an AverageTFR), which computed mean_t(ITPC(t,f))
        — a different quantity from the null that caused slightly
        conservative, uncalibrated p-values.

        Returns
        -------
        dict
            Keys: ``itpc_word``, ``itpc_phrase``, ``itpc_sentence``.
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call analyze() first.")

        phases = self._morlet_phases  # (n_trials, n_channels, 3)

        def _itpc_at(freq_idx: int) -> float:
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])
            # mean across trials per channel, magnitude, then mean channels
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
        """
        rng = np.random.default_rng(seed)

        if method == "morlet":
            return self._compute_morlet_null_itpc(n_permutations, metric, rng)
        elif method != "dft":
            raise ValueError(f"Unknown method '{method}'. Use 'dft' or 'morlet'.")

        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
        n_pad = int(np.ceil(sfreq / self.cfg.dft_freq_resolution))
        n_fft = max(n_pad, data.shape[2])
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        def get_bin_idx(target_f):
            return np.argmin(np.abs(freqs - target_f))

        sent_idx = get_bin_idx(self.cfg.target_sentence_freq)
        phrase_idx = get_bin_idx(self.cfg.target_phrase_freq)
        word_idx = get_bin_idx(self.cfg.target_word_freq)

        spectra = np.fft.rfft(data, n=n_fft, axis=2)

        def get_surrogate_itpc(bin_idx):
            unit_vectors = np.exp(1j * np.angle(spectra[:, :, bin_idx]))
            null_per_ch = PermutationEngine.generate_null_distribution(unit_vectors, n_permutations, rng)
            return np.mean(null_per_ch, axis=1)

        if metric == "sentence":
            return get_surrogate_itpc(sent_idx)
        elif metric == "phrase":
            return get_surrogate_itpc(phrase_idx)
        elif metric == "word":
            return get_surrogate_itpc(word_idx)
        elif metric == "comprehension":
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
        Generate per-channel DFT null distributions via trial-level random
        phase scrambling.
        """
        rng = np.random.default_rng(seed)
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        sfreq = epochs.info["sfreq"]
        n_pad = int(np.ceil(sfreq / self.cfg.dft_freq_resolution))
        n_fft = max(n_pad, data.shape[2])
        spectra = np.fft.rfft(data, n=n_fft, axis=2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

        result = {}
        for freq_name, target_f in (
            ("sentence", self.cfg.target_sentence_freq),
            ("phrase", self.cfg.target_phrase_freq),
            ("word", self.cfg.target_word_freq),
        ):
            bin_idx = int(np.argmin(np.abs(freqs - target_f)))
            # unit_vectors: (n_trials, n_channels)
            unit_vectors = np.exp(1j * np.angle(spectra[:, :, bin_idx]))
            result[freq_name] = PermutationEngine.generate_null_distribution(unit_vectors, n_permutations, rng)

        return result

    def _compute_per_channel_null_morlet(
        self,
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> dict:
        """
        Generate per-channel Morlet null distributions via trial-level random
        phase scrambling.
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call _compute_morlet_target_phases() first.")

        rng = np.random.default_rng(seed)
        phases = self._morlet_phases  # (n_trials, n_channels, 3)

        result = {}
        for freq_name, freq_idx in self._MORLET_FREQ_IDX.items():
            # unit_vectors: (n_trials, n_channels)
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])
            result[freq_name] = PermutationEngine.generate_null_distribution(unit_vectors, n_permutations, rng)

        return result

    def _compute_morlet_null_itpc(
        self,
        n_permutations: int,
        metric: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate null Morlet ITPC distribution via trial-level random phase
        scrambling.
        """
        if self._morlet_phases is None:
            raise ValueError("_morlet_phases not set. Call analyze() before running Morlet permutation tests.")

        phases = self._morlet_phases  # (n_trials, n_channels, 3)

        def surrogate_itpc(freq_idx: int) -> np.ndarray:
            unit_vectors = np.exp(1j * phases[:, :, freq_idx])
            null_per_ch = PermutationEngine.generate_null_distribution(unit_vectors, n_permutations, rng)
            return np.mean(null_per_ch, axis=1)

        if metric in self._MORLET_FREQ_IDX:
            return surrogate_itpc(self._MORLET_FREQ_IDX[metric])
        elif metric == "comprehension":
            return (
                surrogate_itpc(self._MORLET_FREQ_IDX["sentence"]) + surrogate_itpc(self._MORLET_FREQ_IDX["phrase"])
            ) / 2.0
        else:
            raise ValueError(f"Unknown metric '{metric}'")

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
        Compute permutation p-values for a focus by subsetting
        per-channel null distributions.

        Parameters
        ----------
        per_ch_null : dict
            Keys "sentence", "phrase", "word". Each value shape
            (n_surrogates, n_channels).
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
            Ordered names of the clinical channel array.
        per_ch_itpc_dft : np.ndarray, shape (n_clinical_ch, n_freqs)
        dft_freqs : np.ndarray
        per_ch_itpc_morlet : np.ndarray, shape (n_clinical_ch, 3)
            Axis-1 order: [0]=sentence, [1]=phrase, [2]=word
            (matches _MORLET_FREQ_IDX).
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
            "ratio_sent_word": nan,
            "ratio_sent_phrase": nan,
            "ratio_bw_normalized": nan,
            "freq_sentence_hz": nan,
            "freq_phrase_hz": nan,
            "freq_word_hz": nan,
        }

        if not channels:
            return {**base, **nan_metrics}

        ch_to_idx = {ch: i for i, ch in enumerate(clinical_ch_names)}
        ch_indices = [ch_to_idx[ch] for ch in channels if ch in ch_to_idx]

        # Guard: _resolve_focuses guarantees focus channels are in
        # clinical_ch_names, but protect against empty index lists.
        if not ch_indices:
            logger.warning(f"Focus '{focus}' channels not found in clinical array; returning NaN row.")
            return {**base, **nan_metrics}

        # DFT ITPC: _extract_itpc_metrics_dft averages across channels
        dft_metrics = self._extract_itpc_metrics_dft(per_ch_itpc_dft[ch_indices, :], dft_freqs)
        obs_dft = {
            "sentence": dft_metrics["itpc_sentence"],
            "phrase": dft_metrics["itpc_phrase"],
            "word": dft_metrics["itpc_word"],
        }

        # Morlet ITPC: average focus channels then index by frequency
        focus_morlet = np.mean(per_ch_itpc_morlet[ch_indices, :], axis=0)
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
            "itpc_comprehension": dft_metrics["itpc_comprehension"],
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
            "ratio_sent_word": dft_metrics["ratio_sent_word"],
            "ratio_sent_phrase": dft_metrics["ratio_sent_phrase"],
            "ratio_bw_normalized": dft_metrics["ratio_bw_normalized"],
            "freq_sentence_hz": dft_metrics["freq_sentence_hz"],
            "freq_phrase_hz": dft_metrics["freq_phrase_hz"],
            "freq_word_hz": dft_metrics["freq_word_hz"],
        }
