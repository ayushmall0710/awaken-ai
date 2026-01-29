import pytest
import pandas as pd
import tempfile
from pathlib import Path
from data_processing.pipeline import process_stimulus_df


@pytest.fixture
def sample_stimulus_data():
    """Standard stimulus data dictionary for testing."""
    return {
        'patient_id': ['P001', 'P001'],
        'date': ['2024-01-01', '2024-01-01'],
        'trial_type': ['language', 'lcmd+p'],
        'sentences': ['[1, 2, 3]', '[]'],
        'start_time': [1.0, 2.0],
        'end_time': [2.0, 3.0],
        'duration': [1.0, 1.0],
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
    return pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
        'trial_type': ['lang_11', 'lcmd+p', 'lang_70', 'oddball'],
        'sentences': ['', '[]', '', "{'event': 'standard'}"],
        'start_time': [1.0, 2.0, 3.0, 4.0],
        'end_time': [2.0, 3.0, 4.0, 5.0],
        'duration': [1.0, 1.0, 1.0, 1.0],
    })


@pytest.fixture
def temp_data_dir():
    """Temporary directory for pipeline tests that need file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
