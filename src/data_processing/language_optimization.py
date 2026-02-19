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

from src.data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError
from src.utils.signal_processing import normalize_channel_names

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """
    Processor for Language Tracking Paradigm data.

    Attributes:
        loader (UnifiedDataLoader): Instance of UnifiedDataLoader.
        CLINICAL_20 (List[str]): Standard 10-20 system electrodes (Ref: ENG-05.md).
        LH_FOCUS_CHANNELS (List[str]): Left Hemisphere channels for language tracking (Ref: docs/language_tracking.md).
    """

    # Ref: ENG-05.md "Recommended Electrode Set"
    CLINICAL_20 = [
        "Fp1",
        "Fp2",
        "Fz",
        "F3",
        "F4",
        "F7",
        "F8",
        "Cz",
        "C3",
        "C4",
        "T7",
        "T8",
        "Pz",
        "P3",
        "P4",
        "P7",
        "P8",
        "O1",
        "O2",
    ]

    # Ref: docs/language_tracking.md "Electrode Selection (Left-Hemisphere Focus)"
    LH_FOCUS_CHANNELS = ["F7", "T7", "P7", "F3", "C3", "P3"]

    # Filter constants
    HIGHPASS_FREQ = 0.5
    LOWPASS_FREQ = 30.0

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
            primary = set(self.LH_FOCUS_CHANNELS)
            remainder = set(self.CLINICAL_20) - primary
            target_chs = primary | remainder
        else:
            target_chs = set(self.CLINICAL_20)

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

    def preprocess_signal(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply signal processing filters.

        Ref: docs/language_tracking.md "High-pass: 0.5 Hz, Low-pass: ~30 Hz"

        Note: The upstream ArtifactRejector (ENG-03) has been updated to use a 0.5 Hz
        high-pass filter (previously 1 Hz) to preserve Delta band information.
        We apply the 0.5-30 Hz bandpass here to enforce the specific language analysis range.
        """
        epochs_filtered = epochs.copy()

        if not epochs_filtered.preload:
            epochs_filtered.load_data()

        logger.info(f"Applying bandpass filter: {self.HIGHPASS_FREQ}-{self.LOWPASS_FREQ} Hz")
        epochs_filtered.filter(l_freq=self.HIGHPASS_FREQ, h_freq=self.LOWPASS_FREQ, method="iir", verbose=False)

        return epochs_filtered

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
