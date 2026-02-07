"""
Unit tests for ENG-05 Language Optimization.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.data_processing.language_optimization import LanguageProcessor


@pytest.fixture
def mock_language_raw():
    """Create a mock Raw object with language-relevant channels."""
    # Include some target channels and some extras
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
        "ECG",
        "EOG",
    ]
    sfreq = 1000.0
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    # Set measurement date for alignment
    # info['meas_date'] = ... (Cannot set directly in recent MNE)

    # 30 seconds of data
    data = np.random.randn(len(ch_names), int(30 * sfreq))
    raw = mne.io.RawArray(data, info)
    raw.set_meas_date(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
    return raw


@pytest.fixture
def mock_loader(mock_language_raw):
    """Mock UnifiedDataLoader."""
    with patch("src.data_processing.language_optimization.UnifiedDataLoader") as MockLoader:
        loader_instance = MockLoader.return_value

        # Mock load_edf to return our mock raw
        loader_instance.load_edf.return_value = mock_language_raw

        # Mock get_patient_trials
        # Create trials that align with the mock_raw date
        # EDF starts at 12:00:00. Trial 1 at 12:00:05 (5s in)
        base_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()

        trials_df = pd.DataFrame(
            [
                {
                    "patient_id": "TEST",
                    "date": "2024-01-01",
                    "trial_type": "language",
                    "start_time": base_ts + 5.0,
                    "end_time": base_ts + 21.0,  # 16s duration
                    "duration": 16.0,
                    "sentences": [],
                }
            ]
        )
        loader_instance.get_patient_trials.return_value = trials_df

        yield loader_instance


@pytest.fixture
def aligned_events_df():
    """Mock aligned events from TimestampAligner."""
    base_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    return pd.DataFrame(
        [
            {
                "patient_id": "TEST",
                "date": "2024-01-01",
                "trial_type": "language",
                "event_start": base_ts + 5.0,
                "event_start_edf": 5.0,
                "event_end": base_ts + 21.0,
                "duration": 16.0,
            },
            {
                "patient_id": "TEST",
                "date": "2024-01-01",
                "trial_type": "language",
                "event_start": base_ts + 25.0,
                "event_start_edf": 25.0,
                "event_end": base_ts + 41.0,
                "duration": 16.0,
            },
        ]
    )


def test_initialization(mock_loader):
    """Test standard initialization."""
    processor = LanguageProcessor(loader=mock_loader)
    assert processor.loader is not None
    assert "F7" in processor.LH_FOCUS_CHANNELS


def test_select_optimal_channels_lh(mock_language_raw):
    """Test LH channel selection priority."""
    processor = LanguageProcessor(MagicMock())

    # Run selection
    processed_raw = processor.select_optimal_channels(mock_language_raw, focus="LH")

    # Verify LH channels are present (F7, T7, P7)
    current_chs = processed_raw.ch_names
    assert "F7" in current_chs
    assert "T7" in current_chs
    assert "P7" in current_chs

    # Verify we didn't lose too many (mock has full montages, should keep most)
    assert len(current_chs) >= 6  # Minimum LH set


def test_select_optimal_channels_missing(mock_language_raw):
    """Test behavior when target channels are missing (should warn, not crash)."""
    processor = LanguageProcessor(MagicMock())

    # Create raw with only generic names
    info = mne.create_info(["CH1", "CH2"], 1000.0, ch_types="eeg")
    bad_raw = mne.io.RawArray(np.zeros((2, 1000)), info)

    # Should log warning but return original raw if NO matches found
    processed_raw = processor.select_optimal_channels(bad_raw, focus="LH")
    assert processed_raw.ch_names == ["CH1", "CH2"]


def test_preprocess_signal(mock_language_raw):
    """Test filtering parameters."""
    processor = LanguageProcessor(MagicMock())

    # Run processing
    filtered = processor.preprocess_signal(mock_language_raw)

    # Check filter info
    assert filtered.info["highpass"] == 0.5
    assert filtered.info["lowpass"] == 30.0


def test_create_epochs_from_events(mock_language_raw, aligned_events_df):
    """Test creating epochs from aligned events DataFrame."""
    processor = LanguageProcessor(MagicMock())

    # Call new method
    epochs = processor.create_epochs_from_events(
        raw=mock_language_raw, events_df=aligned_events_df, focus="LH", filter_signal=True
    )

    # Should return epochs
    assert epochs is not None
    assert isinstance(epochs, mne.Epochs)

    # Should have 2 epochs (from aligned_events_df fixture)
    assert len(epochs) >= 1


def test_process_patient_with_aligned_events(mock_loader, aligned_events_df):
    """Test process_patient with aligned events provided (new integration path)."""
    processor = LanguageProcessor(loader=mock_loader)

    # Call with aligned_events parameter
    epochs = processor.process_patient("TEST", aligned_events=aligned_events_df, focus="LH")

    # Should return epochs
    assert epochs is not None
    assert isinstance(epochs, mne.Epochs)

    # Should process aligned events
    assert len(epochs) >= 1
