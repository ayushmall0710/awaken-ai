"""
Test suite for UnifiedDataLoader and PatientData classes.

Run with: pytest tests/test_unified_data_loader.py -v

Uses fixture-based dummy data for CI compatibility.
"""

import pandas as pd
import pytest

from src.data_loading.unified_data_loader import (
    UnifiedDataLoader,
    UnifiedDataLoadingError,
)
from src.data_loading.patient_data import PatientData


@pytest.fixture
def sample_unified_data():
    """Sample unified stimulus data for testing."""
    return {
        "patient_id": ["CON001a", "CON001a", "CON005", "CON005", "CON005", "CON008"],
        "date": [
            "2025-01-15",
            "2025-01-15",
            "2025-02-14",
            "2025-02-14",
            "2025-05-06",
            "2025-03-10",
        ],
        "trial_type": [
            "language",
            "oddball",
            "language",
            "oddball",
            "language",
            "language",
        ],
        "sentences": [
            [{"text": "hello", "order": 1}],
            [{"text": "standard", "order": 1}],
            [{"text": "world", "order": 1}],
            [{"text": "rare", "order": 1}],
            [{"text": "test", "order": 1}],
            [{"text": "data", "order": 1}],
        ],
        "start_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "end_time": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "duration": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "source_file": [
            "test.csv",
            "test.csv",
            "test.csv",
            "test.csv",
            "test.csv",
            "test.csv",
        ],
    }


@pytest.fixture
def sample_unified_df(sample_unified_data):
    """Sample unified DataFrame for testing."""
    return pd.DataFrame(sample_unified_data)


@pytest.fixture
def temp_parquet_file(sample_unified_df, tmp_path):
    """Create temporary parquet file for testing."""
    parquet_file = tmp_path / "test_unified_stimulus_results.parquet"
    sample_unified_df.to_parquet(parquet_file, index=False)
    return parquet_file


@pytest.fixture
def loader(temp_parquet_file):
    """Create UnifiedDataLoader instance with test data."""
    return UnifiedDataLoader(temp_parquet_file)


class TestUnifiedDataLoader:
    """Test suite for UnifiedDataLoader class."""

    def test_initialization(self, loader):
        assert loader is not None
        assert len(loader.trials_df) > 0
        assert len(loader.get_patient_ids()) == 3  # CON001a, CON005, CON008

    def test_cross_patient_queries(self, loader):
        patient_ids = loader.get_patient_ids()
        assert isinstance(patient_ids, list)
        assert len(patient_ids) > 0

        trial_types = loader.get_trial_types()
        assert "language" in trial_types

        language_trials = loader.get_trials_by_type("language")
        assert len(language_trials) > 0

        filtered = loader.get_trials_by_type("language", patient_ids=patient_ids[:2])
        assert len(filtered) <= len(language_trials)

    def test_single_patient_access(self, loader):
        patient_ids = loader.get_patient_ids()
        test_patient = patient_ids[0]

        trials = loader.get_patient_trials(test_patient)
        assert len(trials) > 0

        patient = loader.get_patient(test_patient)
        assert isinstance(patient, PatientData)
        assert patient.patient_id == test_patient

    def test_multi_session_support(self, loader):
        sessions = loader.get_patient_sessions("CON005")
        assert len(sessions) == 2
        assert "2025-02-14" in sessions
        assert "2025-05-06" in sessions

        patient = loader.get_patient("CON005")
        patient_sessions = patient.list_sessions()
        assert len(patient_sessions) == 2

    def test_validation_schema(self, loader):
        validation = loader.validate_schema()
        assert validation["has_required_columns"]
        assert validation["has_data"]

    def test_validation_per_patient(self, loader):
        patient_ids = loader.get_patient_ids()
        validation_df = loader.validate_all_patients()

        assert len(validation_df) == len(patient_ids)
        assert "patient_id" in validation_df.columns
        assert "has_trials" in validation_df.columns

    def test_error_handling_invalid_patient(self, loader):
        with pytest.raises(UnifiedDataLoadingError):
            loader.get_patient_trials("INVALID_ID")

    def test_load_edf_filepath_exclusivity(self, loader):
        with pytest.raises(ValueError):
            loader.load_edf(patient_id="CON005", filepath="/some/path.EDF")

        with pytest.raises(ValueError):
            loader.load_edf()

    def test_error_handling_invalid_session(self, loader):
        with pytest.raises(UnifiedDataLoadingError):
            loader.load_edf("CON005", date="2099-01-01")

    def test_metadata_access(self, loader):
        info = loader.get_info()
        assert "total_trials" in info
        assert "total_patients" in info
        assert info["total_trials"] == len(loader.trials_df)


class TestPatientData:
    """Test suite for PatientData class."""

    def test_trial_filtering(self, loader):
        patient = loader.get_patient("CON001a")
        trial_types = patient.get_trial_types()
        assert len(trial_types) > 0

        if trial_types:
            trials = patient.get_trials_by_type(trial_types[0])
            assert len(trials) >= 0

    def test_trial_access(self, loader):
        patient = loader.get_patient("CON001a")
        if len(patient.trials_df) > 0:
            trial = patient.get_trial(0)
            assert trial is not None
            assert "trial_type" in trial.index

    def test_multi_session_patient(self, loader):
        patient_multi = loader.get_patient("CON005")

        sessions = patient_multi.list_sessions()
        assert len(sessions) == 2

    def test_get_all_trials(self, loader):
        all_trials = loader.get_all_trials()
        assert isinstance(all_trials, pd.DataFrame)
        assert len(all_trials) == 6

    def test_get_trial_summary(self, loader):
        summary = loader.get_trial_summary()
        assert isinstance(summary, pd.DataFrame)
        assert "patient_id" in summary.columns
        assert "trial_type" in summary.columns
        assert "count" in summary.columns
