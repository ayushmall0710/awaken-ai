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
import pandas as pd

from src.data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError

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
        aligned_events: pd.DataFrame,
        focus: str = "LH",
        filter_signal: bool = True,
    ) -> Optional[mne.Epochs]:
        """
        End-to-end processing for a single patient using pre-aligned events.

        Args:
            patient_id: Patient ID (e.g., 'CON008').
            aligned_events: Pre-aligned events from TimestampAligner (REQUIRED).
            focus: Channel selection focus ('LH' or 'Clinical').
            filter_signal: Whether to apply bandpass filtering.

        Returns:
            mne.Epochs object containing processed language trial segments.
        """
        # Filter for language trials only
        lang_events = aligned_events[aligned_events["trial_type"] == "language"]
        if lang_events.empty:
            logger.warning(f"No language trials in aligned events for {patient_id}")
            return None

        # Load raw EDF
        try:
            raw_data = self.loader.load_edf(patient_id)
        except UnifiedDataLoadingError as e:
            logger.error(f"Failed to load EDF for {patient_id}: {e}")
            return None

        return self.create_epochs_from_events(raw_data, lang_events, focus, filter_signal)

    def select_optimal_channels(self, raw: mne.io.Raw, focus: str = "LH") -> mne.io.Raw:
        """
        Select subset of channels based on focus strategy.

        Args:
            raw: MNE Raw object.
            focus: 'LH' for Left Hemisphere focus, 'Clinical' for standard 20.

        Returns:
            New Raw object (copied) with picked channels.
        """
        available_chs = raw.ch_names

        # Normalize names (stripping EEG prefix if strictly needed)
        target_chs = []

        if focus == "LH":
            # Logic: Prioritize LH_FOCUS_CHANNELS, fill remainder with Clinical 20
            primary = self.LH_FOCUS_CHANNELS
            remainder = [ch for ch in self.CLINICAL_20 if ch not in primary]
            target_chs = primary + remainder
        else:
            target_chs = self.CLINICAL_20

        picks = []
        missing = []

        for target in target_chs:
            if target in available_chs:
                picks.append(target)
            else:
                # Try simple variations
                found = False
                for ch in available_chs:
                    clean_ch = ch.replace("EEG ", "").replace("-Ref", "").split("-")[0]
                    if clean_ch.upper() == target.upper():
                        picks.append(ch)
                        found = True
                        break
                if not found:
                    missing.append(target)

        if missing:
            logger.warning(f"Missing channels for {focus} montage: {missing}")

        if not picks:
            logger.error("No valid channels found from target set. Returning original raw.")
            return raw

        logger.info(f"Selected {len(picks)} channels for {focus} focus.")
        return raw.copy().pick(picks)

    def preprocess_signal(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply signal processing filters.

        Ref: docs/language_tracking.md "High-pass: 0.5 Hz, Low-pass: ~30 Hz"
        """
        raw_filtered = raw.copy()

        if not raw_filtered.preload:
            raw_filtered.load_data()

        l_freq = 0.5
        h_freq = 30.0

        logger.info(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
        raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)

        return raw_filtered

    def create_epochs_from_events(
        self,
        raw: mne.io.Raw,
        events_df: pd.DataFrame,
        focus: str = "LH",
        filter_signal: bool = True,
        tmax: float = 16.0,
    ) -> Optional[mne.Epochs]:
        """
        Create optimized epochs from aligned events DataFrame.

        This method is designed to consume output from TimestampAligner,
        providing consistent event timing across all analyses.

        Args:
            raw: MNE Raw object (can be multi-session dict or single Raw).
            events_df: DataFrame with 'event_start' column (EDF-relative seconds).
            focus: Channel selection strategy ('LH' or 'Clinical').
            filter_signal: Apply 0.5-30Hz bandpass filtering.
            tmax: Epoch duration in seconds (default: 16.0s per language_tracking spec).

        Returns:
            mne.Epochs object with optimized channels and filtering applied.
        """
        # Handle multi-session case
        if isinstance(raw, dict):
            all_epochs = []
            for session_date, session_raw in raw.items():
                # Filter events for this session if date column exists
                if "date" in events_df.columns:
                    session_events = events_df[events_df["date"] == session_date]
                else:
                    session_events = events_df

                if session_events.empty:
                    continue

                epochs = self._create_epochs_from_aligned(session_raw, session_events, focus, filter_signal, tmax)
                if epochs:
                    all_epochs.append(epochs)

            if not all_epochs:
                return None
            return mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]
        else:
            # Single session
            return self._create_epochs_from_aligned(raw, events_df, focus, filter_signal, tmax)

    def _create_epochs_from_aligned(
        self,
        raw: mne.io.Raw,
        events_df: pd.DataFrame,
        focus: str,
        filter_signal: bool,
        tmax: float,
    ) -> Optional[mne.Epochs]:
        """
        Internal helper to create epochs from aligned events for a single session.
        Handles both flat (one event per row) and nested (one trial per row) DataFrames.
        """
        # 1. Channel Selection
        raw_selected = self.select_optimal_channels(raw, focus=focus)

        # 2. Filtering
        if filter_signal:
            raw_selected = self.preprocess_signal(raw_selected)

        # 3. Convert aligned events to MNE events array
        # TimestampAligner provides event_start_edf (EDF-relative seconds) directly,
        # so no timezone conversion is needed here.
        sfreq = raw_selected.info["sfreq"]
        recording_end = raw_selected.times[-1]
        events = []
        event_id = {"language": 1}

        # Determine input structure (TimestampAligner returns nested 'sentences' column)
        is_nested = "sentences" in events_df.columns and "event_start_edf" not in events_df.columns

        if is_nested:
            # Flatten by iterating over trials and their nested events
            iterator = []
            for _, trial_row in events_df.iterrows():
                nested_items = trial_row.get("sentences", [])
                if isinstance(nested_items, list):
                    iterator.extend(nested_items)
        else:
            # Already flat
            if "event_start_edf" not in events_df.columns:
                logger.error("Aligner output missing 'event_start_edf' and 'sentences'. Cannot process.")
                return None
            iterator = [row.to_dict() for _, row in events_df.iterrows()]

        # Process flattened event stream
        for event_data in iterator:
            if not isinstance(event_data, dict):
                continue

            onset_sec = event_data.get("event_start_edf")

            # Skip if alignment failed (NaN or None)
            if onset_sec is None or pd.isna(onset_sec):
                continue

            # Validate onset is within recording bounds
            if onset_sec < 0 or onset_sec > recording_end:
                if abs(onset_sec) > 5.0 and abs(onset_sec - recording_end) > 5.0:
                    logger.debug(f"Event onset {onset_sec:.2f}s out of bounds.")
                continue

            sample = int(onset_sec * sfreq)
            events.append([sample, 0, 1])

        if not events:
            logger.warning("No valid events after alignment validation.")
            return None

        events_array = np.array(events)

        # 5. Create Epochs
        epochs = mne.Epochs(
            raw_selected,
            events_array,
            event_id=event_id,
            tmin=0,
            tmax=tmax,
            baseline=None,
            preload=True,
            reject=None,
            verbose=False,
            event_repeated="drop",
        )

        return epochs
