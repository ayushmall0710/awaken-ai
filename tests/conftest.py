import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_processing.pipeline import process_stimulus_df


@pytest.fixture
def mock_loader():
    """Mock UnifiedDataLoader."""
    with patch("src.data_processing.timestamp_aligner.UnifiedDataLoader") as MockLoader:
        loader_instance = MockLoader.return_value
        # Ensure child methods are Mocks
        loader_instance.get_patient.return_value = MagicMock()
        loader_instance.get_stimulus_audio_path = MagicMock()
        loader_instance.load_stimulus_audio = MagicMock()
        yield loader_instance


@pytest.fixture
def mock_raw():
    """Mock mne.io.Raw object."""
    raw = MagicMock()
    raw.info = {
        "sfreq": 1000.0,
        "meas_date": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),  # 12:00 UTC
        "ch_names": ["DC", "EEG"],
    }
    raw.times = np.linspace(0, 10, 10000)  # 10 seconds of data
    raw.ch_names = ["DC", "EEG"]

    # Create a dummy DC signal (sine wave)
    # 10s at 1000Hz = 10000 samples
    dc_data = np.sin(2 * np.pi * 5 * np.linspace(0, 10, 10000))
    # Return 2 channels
    raw.get_data.return_value = np.array([dc_data, np.zeros(10000)])

    return raw


@pytest.fixture
def sample_trials_df():
    """Sample DataFrame resembling patient.trials_df, aligned with mock_raw dates."""
    # Trial starts at 12:00:01 UTC (timestamp=1704110401)
    # EDF starts at 12:00:00 UTC (timestamp=1704110400)
    # So relative trial start should be 1.0s
    base_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()

    return pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "trial_type": "language",
                "start_time": base_ts + 1.0,  # 1s after EDF start
                "end_time": base_ts + 2.0,  # 2s after EDF start
                "duration": 1.0,
                "sentences": [{"event": "1", "onset_time": 0.0}],
                "source_file": "test_log.csv",
            }
        ]
    )


@pytest.fixture
def sample_stimulus_data():
    """Standard stimulus data dictionary for testing."""
    return {
        "patient_id": ["P001", "P001"],
        "date": ["2024-01-01", "2024-01-01"],
        "trial_type": ["language", "lcmd+p"],
        "sentences": ["[1, 2, 3]", "[]"],
        "start_time": [1.0, 2.0],
        "end_time": [2.0, 3.0],
        "duration": [1.0, 1.0],
    }


@pytest.fixture
def sample_stimulus_df(sample_stimulus_data):
    """Standard stimulus DataFrame for testing."""
    return pd.DataFrame(sample_stimulus_data)


@pytest.fixture
def unified_df(raw_stimulus_df):
    """Create test DataFrame by actually processing raw data through pipeline logic."""
    # Use a source name that passes provenance checks (needs 'stimulus_results' or 'patient_df')
    source_name = "test_stimulus_results.csv"
    return process_stimulus_df(raw_stimulus_df, source_name)


@pytest.fixture
def raw_stimulus_df():
    """Simulates raw CSV data before processing - used for pipeline tests."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003", "P004"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "trial_type": ["lang_11", "lcmd+p", "lang_70", "oddball"],
            "sentences": ["", "[]", "", "{'event': 'standard'}"],
            "start_time": [1.0, 2.0, 3.0, 4.0],
            "end_time": [2.0, 3.0, 4.0, 5.0],
            "duration": [1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def temp_data_dir():
    """Temporary directory for pipeline tests that need file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
