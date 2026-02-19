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
