"""
Language Optimization Module for ENG-05.

This module provides the LanguageProcessor class to isolate language trials,
apply specific filtering, and select optimal electrodes for language tracking analysis.

References:
    - docs/language_tracking.md: "The Language Tracking Paradigm" & "Optimization Strategies"
    - tasks/ENG-05.md: "language_optimization.py created"
"""

import logging
from typing import Optional

import mne
import numpy as np

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError
from src.utils.signal_processing import normalize_channel_names

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """
    Processor for Language Tracking Paradigm data.

    Attributes:
        loader (UnifiedDataLoader): Instance of UnifiedDataLoader.
        CLINICAL_20 (List[str]): Standard 10-20 system electrodes.
        LH_FOCUS_CHANNELS (List[str]): Left Hemisphere channels for language tracking.
    """

    # Backward compatibility pointing to centralized configs
    CLINICAL_20 = config.CLINICAL_20
    LH_FOCUS_CHANNELS = config.LH_FOCUS_CHANNELS

    # Filter constants
    HIGHPASS_FREQ = 0.5
    LOWPASS_FREQ = 30.0

    # Downsampling target. Source EDFs record at 512 Hz; for ITPC targeting
    # 0.05-2.0 Hz, 256 Hz is sufficient (Nyquist = 128 Hz >> 30 Hz low-pass)
    # and halves TFR computation time. Matches Sokoliuk 2021 methodology.
    TARGET_SFREQ = 256.0

    # ITPC Constants
    # Frequencies: 0.05 Hz (very slow) to 2.0 Hz (covering word rate ~0.77 Hz)
    ITPC_FREQS = np.logspace(np.log10(0.05), np.log10(2.0), num=40)
    # Adaptive cycles: 0.5 minimum for low freq
    ITPC_CYCLES = np.array([max(0.5, f * 2.0) for f in ITPC_FREQS])

    def __init__(self, loader: Optional[UnifiedDataLoader] = None):
        """
        Initialize the LanguageProcessor.

        Args:
            loader: Optional UnifiedDataLoader instance. If None, creates a new one.
        """
        self.loader = loader if loader else UnifiedDataLoader()

    def process_patient(
        self,
        patient_id: str,
        focus: str = "LH",
        filter_signal: bool = True,
    ) -> Optional[mne.Epochs]:
        """
        End-to-end processing for a single patient using pre-cleaned epochs (ENG-03).

        Args:
            patient_id: Patient ID (e.g., 'CON008').
            focus: Channel selection focus ('LH' or 'Clinical').
            filter_signal: Whether to apply bandpass filtering.

        Returns:
            mne.Epochs object containing processed language trial segments.

        Raises:
            UnifiedDataLoadingError: If data loading fails.
        """
        # Load cleaned epochs from ENG-03
        try:
            # Note: We load all sessions and concatenate them if needed
            # ENG-03 produces one .fif file per session per trial type
            sessions = self.loader.get_patient_sessions(patient_id)
            all_epochs = []

            for date in sessions:
                try:
                    epochs = self.loader.load_clean_epochs(patient_id, date, trial_type="language")
                    all_epochs.append(epochs)
                except FileNotFoundError:
                    logger.warning(f"No clean language epochs found for {patient_id} on {date}. Skipping session.")
                    continue

            if not all_epochs:
                logger.error(f"No clean epochs found for {patient_id}. Run ENG-03 (ArtifactRejector) first.")
                return None

            epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]

        except UnifiedDataLoadingError as e:
            logger.error(f"Failed to load data for {patient_id}: {e}")
            return None

        # Apply optimization steps
        # 1. Channel Selection
        # 2. Filtering
        epochs = self.select_optimal_channels(epochs, focus=focus)
        if filter_signal:
            epochs = self.preprocess_signal(epochs)

        return epochs

    def select_optimal_channels(self, epochs: mne.Epochs, focus: str = "LH") -> mne.Epochs:
        """
        Select subset of channels based on focus strategy.

        Args:
            epochs: MNE Epochs object.
            focus: 'LH' for Left Hemisphere focus, 'Clinical' for standard 20.

        Returns:
            New Epochs object (copied) with picked channels.
        """
        available_chs = epochs.ch_names

        if focus == "LH":
            target_chs = set(config.LH_FOCUS_CHANNELS + config.CLINICAL_20)
        elif focus == "RH":
            target_chs = set(config.RH_FOCUS_CHANNELS + config.CLINICAL_20)
        else:
            target_chs = set(config.CLINICAL_20)

        # Build map of clean_name -> original_name for robust matching
        # Uses shared logic from utils to strip prefixes (EEG-, etc)
        normalized_names = normalize_channel_names(available_chs)
        clean_map = {}
        for orig, clean in zip(available_chs, normalized_names):
            clean_map[clean.upper()] = orig

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
            logger.warning(f"Missing channels for {focus} montage: {missing}")

        if not picks:
            logger.error("No valid channels found from target set. Returning original epochs.")
            return epochs

        logger.info(f"Selected {len(picks)} channels for {focus} focus.")
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
        self, epochs: mne.Epochs, freqs: Optional[np.ndarray] = None, n_cycles: Optional[np.ndarray] = None
    ):
        """
        Compute Inter-Trial Phase Coherence (ITPC).

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
        # return_itc=True returns (power, itc). We strictly want ITC (ITPC).
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

        For each trial and electrode, computes the FFT, extracts the phase at each
        frequency bin, then averages unit phase vectors across trials. This provides
        a single ITPC spectrum (no time dimension) suitable for cross-validating the
        Morlet wavelet approach.

        Frequency resolution = 1 / epoch_duration Hz. For 16s epochs this yields
        ~0.0625 Hz resolution, comparable to Sokoliuk's 0.07 Hz.

        Args:
            epochs: Preprocessed MNE Epochs object.

        Returns:
            itpc_spectrum (np.ndarray): ITPC values, shape (n_channels, n_freqs).
            freqs (np.ndarray): Frequency axis in Hz.
        """
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        n_trials, n_channels, n_times = data.shape
        sfreq = epochs.info["sfreq"]

        # FFT of each trial and channel
        spectra = np.fft.rfft(data, axis=2)  # (n_trials, n_channels, n_freqs)
        freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

        # Extract unit phase vectors and average across trials
        unit_vectors = np.exp(1j * np.angle(spectra))  # (n_trials, n_channels, n_freqs)
        itpc_spectrum = np.abs(np.mean(unit_vectors, axis=0))  # (n_channels, n_freqs)

        logger.info(
            f"DFT ITPC computed: {n_trials} trials, {n_channels} channels, "
            f"freq resolution={freqs[1] - freqs[0]:.4f} Hz, max_freq={freqs[-1]:.1f} Hz"
        )
        return itpc_spectrum, freqs

    def extract_itpc_metrics_dft(self, itpc_spectrum: np.ndarray, freqs: np.ndarray) -> dict:
        """
        Extract sentence-rate and word-rate ITPC from a DFT ITPC spectrum.

        Args:
            itpc_spectrum: DFT ITPC array, shape (n_channels, n_freqs).
            freqs: Frequency axis from compute_itpc_dft.

        Returns:
            dict with same keys as extract_itpc_metrics for direct comparison.
        """
        target_sent = 0.065
        idx_sent = np.argmin(np.abs(freqs - target_sent))
        actual_sent = freqs[idx_sent]
        itpc_sent_val = float(np.mean(itpc_spectrum[:, idx_sent]))

        target_word = 0.77
        idx_word = np.argmin(np.abs(freqs - target_word))
        actual_word = freqs[idx_word]
        itpc_word_val = float(np.mean(itpc_spectrum[:, idx_word]))

        ratio = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_word": itpc_word_val,
            "ratio_sent_word": ratio,
            "freq_sentence_hz": actual_sent,
            "freq_word_hz": actual_word,
            "idx_sentence": idx_sent,
        }

    def extract_itpc_metrics(self, itpc_data: np.ndarray, freqs: Optional[np.ndarray] = None) -> dict:
        """
        Extract ITPC metrics at Sentence (0.065 Hz) and Word (0.77 Hz) rates.

        Args:
            itpc_data (np.ndarray): ITPC data array.
            freqs (np.ndarray, optional): Frequencies corresponding to ITPC data. Defaults to class ITPC_FREQS.

        Returns:
            dict: Dictionary containing sentence_mean, word_mean, ratio, and actual frequencies.
        """
        if freqs is None:
            freqs = self.ITPC_FREQS

        # A. Sentence Rate (~0.065 Hz)
        target_sent = 0.065
        idx_sent = np.argmin(np.abs(freqs - target_sent))
        actual_sent = freqs[idx_sent]
        itpc_sent_val = np.mean(itpc_data[:, idx_sent, :])

        # B. Word Rate (~0.77 Hz)
        target_word = 0.77
        idx_word = np.argmin(np.abs(freqs - target_word))
        actual_word = freqs[idx_word]
        itpc_word_val = np.mean(itpc_data[:, idx_word, :])

        # C. Ratio
        ratio = itpc_sent_val / itpc_word_val if itpc_word_val > 0 else 0.0

        return {
            "itpc_sentence": itpc_sent_val,
            "itpc_word": itpc_word_val,
            "ratio_sent_word": ratio,
            "freq_sentence_hz": actual_sent,
            "freq_word_hz": actual_word,
            "idx_sentence": idx_sent,
        }

    def plot_itpc_results(self, itc, patient_id: str, output_dir: str, metrics: dict):
        """
        Generate and save enhanced ITPC plots (Topomap and TFR).

        Args:
            itc: MNE AverageTFR object.
            patient_id: Patient ID string.
            output_dir: Path to save outputs.
            metrics: Metrics dictionary from extract_itpc_metrics.
        """
        from pathlib import Path

        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_freq = metrics["freq_sentence_hz"]
        word_freq = metrics["freq_word_hz"]

        # 1. Topomap at Sentence Rate
        fig_topo, ax_topo = plt.subplots(1, 1, figsize=(10, 8))
        itc.plot_topomap(
            tmin=0,
            tmax=None,
            fmin=target_freq - 0.01,
            fmax=target_freq + 0.01,
            baseline=None,
            mode=None,  # Plot raw ITPC values (0-1)
            axes=ax_topo,
            show=False,
            cmap="viridis",
            colorbar=True,
            vlim=(0, 0.3),  # Fix scale for comparability across subjects
        )
        ax_topo.set_title(
            f"ITPC Topomap @ {target_freq:.3f} Hz (Sentence Rate)\n{patient_id}",
            fontsize=14,
            fontweight="bold",
        )
        topo_path = output_dir / f"{patient_id}_language_ITPC_topomap.png"
        fig_topo.savefig(topo_path, dpi=300, bbox_inches="tight")
        plt.close(fig_topo)
        logger.info(f"Saved enhanced Topomap to {topo_path}")

        # 2. Time-Frequency Plot
        fig_tfr, ax_tfr = plt.subplots(1, 1, figsize=(14, 8))
        # Plot TFR
        itc.plot(
            baseline=None,
            mode=None,  # Plot raw ITPC
            axes=ax_tfr,
            show=False,
            combine="mean",
            cmap="viridis",
            vlim=(0, 0.3),  # Cap at 0.3 to see low-value variations clearly
            colorbar=True,
        )

        # Add markers for specific frequencies
        ax_tfr.axhline(
            y=target_freq,
            color="white",
            linestyle="--",
            linewidth=2,
            label=f"Sentence ({target_freq:.3f} Hz)",
        )
        ax_tfr.text(
            itc.times[0],
            target_freq,
            " Sentence",
            color="white",
            verticalalignment="bottom",
            fontweight="bold",
        )

        ax_tfr.axhline(
            y=word_freq,
            color="white",
            linestyle=":",
            linewidth=2,
            label=f"Word ({word_freq:.3f} Hz)",
        )
        ax_tfr.text(
            itc.times[0],
            word_freq,
            " Word",
            color="white",
            verticalalignment="bottom",
            fontweight="bold",
        )

        ax_tfr.set_title(f"ITPC Time-Frequency ({patient_id}) - Mean of Channels", fontsize=16)
        ax_tfr.set_xlabel("Time (s)", fontsize=12)
        ax_tfr.set_ylabel("Frequency (Hz)", fontsize=12)

        tfr_path = output_dir / f"{patient_id}_language_ITPC_tfr.png"
        fig_tfr.savefig(tfr_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tfr)
        logger.info(f"Saved enhanced TFR plot to {tfr_path}")
