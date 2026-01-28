import pytest
import pandas as pd
import tempfile
from pathlib import Path


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
def unified_df():
    """Create test DataFrame representing unified/processed data - no external files needed."""
    return pd.DataFrame({
        'patient_id': ['P001', 'P001', 'P002', 'P003'],
        'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03'],
        'trial_type': ['language', 'left_command', 'language', 'oddball'],
        'sentences': [
            [{'event': '11', 'onset_time': None}],
            [],
            [{'event': '70', 'onset_time': None}],
            [{'event': 'standard', 'onset_time': None}]
        ],
        'start_time': [1.0, 2.0, 3.0, 4.0],
        'end_time': [2.0, 3.0, 4.0, 5.0],
        'duration': [1.0, 1.0, 1.0, 1.0],
        'source_file': [
            'P001_stimulus_results.csv',
            'patient_df_P001.csv',
            'P002_stimulus_results.csv',
            'patient_df_P003.csv'
        ]
    })


@pytest.fixture
def raw_stimulus_df():
    """Simulates raw CSV data before processing - used for pipeline tests."""
    return pd.DataFrame({
        'patient_id': ['P001', 'P002'],
        'date': ['2024-01-01', '2024-01-02'],
        'trial_type': ['lang_11', 'lcmd+p'],
        'sentences': ['', '[]'],
        'start_time': [1.0, 2.0],
        'end_time': [2.0, 3.0],
        'duration': [1.0, 1.0],
    })


@pytest.fixture
def temp_data_dir():
    """Temporary directory for pipeline tests that need file I/O."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
