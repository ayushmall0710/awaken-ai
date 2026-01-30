"""
Test suite for UnifiedDataLoader and PatientData classes.

Run with: pytest tests/test_unified_data_loader.py -v
"""

import unittest
import warnings
from pathlib import Path

import pandas as pd
import pytest
import mne

# Add src to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_loading.unified_data_loader import UnifiedDataLoader, UnifiedDataLoadingError
from data_loading.patient_data import PatientData


class TestUnifiedDataLoader(unittest.TestCase):
    """Test suite for UnifiedDataLoader class."""

    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        cls.parquet_path = (
            project_root / "data" / "EEG" / "unified_stimulus_results.parquet"
        )

        if not cls.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {cls.parquet_path}")

        cls.loader = UnifiedDataLoader(cls.parquet_path)

    def test_initialization(self):
        """Test loader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertGreater(len(self.loader.trials_df), 0)
        self.assertEqual(len(self.loader.get_patient_ids()), 14)

    def test_cross_patient_queries(self):
        patient_ids = self.loader.get_patient_ids()
        self.assertIsInstance(patient_ids, list)
        self.assertGreater(len(patient_ids), 0)

        trial_types = self.loader.get_trial_types()
        self.assertIn("language", trial_types)

        language_trials = self.loader.get_trials_by_type("language")
        self.assertGreater(len(language_trials), 0)

        filtered = self.loader.get_trials_by_type(
            "language", patient_ids=patient_ids[:2]
        )
        self.assertLessEqual(len(filtered), len(language_trials))

    def test_single_patient_access(self):
        patient_ids = self.loader.get_patient_ids()
        test_patient = patient_ids[0]

        trials = self.loader.get_patient_trials(test_patient)
        self.assertGreater(len(trials), 0)

        patient = self.loader.get_patient(test_patient)
        self.assertIsInstance(patient, PatientData)
        self.assertEqual(patient.patient_id, test_patient)

    def test_multi_session_support(self):
        sessions = self.loader.get_patient_sessions("CON005")
        self.assertEqual(len(sessions), 2)
        self.assertIn("2025-02-14", sessions)
        self.assertIn("2025-05-06", sessions)

        patient = self.loader.get_patient("CON005")
        patient_sessions = patient.list_sessions()
        self.assertEqual(len(patient_sessions), 2)

    def test_edf_loading_single_session(self):
        patient = self.loader.get_patient("CON001a")

        try:
            raw = patient.raw
            self.assertIsNotNone(raw)
            self.assertIsInstance(raw, mne.io.Raw)
        except UnifiedDataLoadingError as e:
            self.assertIn("Could not find EDF", str(e))

    def test_edf_loading_multi_session(self):
        try:
            edfs = self.loader.load_edf("CON005")
            self.assertIsNotNone(edfs)
            self.assertIsInstance(edfs, dict)
            self.assertEqual(len(edfs), 2)
            self.assertIn("2025-02-14", edfs)
            self.assertIn("2025-05-06", edfs)
            self.assertIsInstance(edfs["2025-02-14"], mne.io.Raw)

            raw = self.loader.load_edf("CON005", date="2025-02-14")
            self.assertIsNotNone(raw)
            self.assertIsInstance(raw, mne.io.Raw)
        except UnifiedDataLoadingError as e:
            self.assertIn("Could not find EDF", str(e))

    def test_validation_schema(self):
        validation = self.loader.validate_schema()
        self.assertTrue(validation["has_required_columns"])
        self.assertTrue(validation["has_data"])

    def test_validation_per_patient(self):
        patient_ids = self.loader.get_patient_ids()
        validation_df = self.loader.validate_all_patients()

        self.assertEqual(len(validation_df), len(patient_ids))
        self.assertIn("patient_id", validation_df.columns)
        self.assertIn("has_trials", validation_df.columns)

    def test_error_handling_invalid_patient(self):
        with self.assertRaises(UnifiedDataLoadingError):
            self.loader.get_patient_trials("INVALID_ID")

    def test_edf_filenames_single_session(self):
        patient = self.loader.get_patient("CON001a")

        try:
            paths = patient.edf_paths
            filenames = patient.edf_filenames

            self.assertIsInstance(paths, Path)
            self.assertIsInstance(filenames, str)
            self.assertTrue(filenames.endswith(".EDF"))
        except UnifiedDataLoadingError as e:
            self.assertIn("Could not find EDF", str(e))

    def test_edf_filenames_multi_session(self):
        patient = self.loader.get_patient("CON005")

        try:
            paths = patient.edf_paths
            filenames = patient.edf_filenames

            self.assertIsInstance(paths, dict)
            self.assertIsInstance(filenames, dict)
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(filenames), 2)
            self.assertIn("2025-02-14", filenames)
            self.assertIn("2025-05-06", filenames)
            self.assertTrue(filenames["2025-02-14"].endswith(".EDF"))
        except UnifiedDataLoadingError as e:
            self.assertIn("Could not find EDF", str(e))

    def test_load_edf_by_filepath(self):
        patient = self.loader.get_patient("CON001a")

        try:
            edf_path = patient.edf_paths
            raw = self.loader.load_edf(filepath=edf_path)
            self.assertIsNotNone(raw)
            self.assertIsInstance(raw, mne.io.Raw)
        except UnifiedDataLoadingError:
            pass

    def test_load_edf_filepath_exclusivity(self):
        with self.assertRaises(ValueError):
            self.loader.load_edf(patient_id="CON005", filepath="/some/path.EDF")

        with self.assertRaises(ValueError):
            self.loader.load_edf()

    def test_error_handling_invalid_session(self):
        with self.assertRaises(UnifiedDataLoadingError):
            self.loader.load_edf("CON005", date="2099-01-01")

    def test_metadata_access(self):
        info = self.loader.get_info()
        self.assertIn("total_trials", info)
        self.assertIn("total_patients", info)
        self.assertEqual(info["total_trials"], len(self.loader.trials_df))


class TestPatientData(unittest.TestCase):
    """Test suite for PatientData class."""

    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        parquet_path = (
            project_root / "data" / "EEG" / "unified_stimulus_results.parquet"
        )
        cls.loader = UnifiedDataLoader(parquet_path)
        cls.patient = cls.loader.get_patient("CON001a")

    def test_trial_filtering(self):
        trial_types = self.patient.get_trial_types()
        self.assertGreater(len(trial_types), 0)

        if trial_types:
            trials = self.patient.get_trials_by_type(trial_types[0])
            self.assertGreaterEqual(len(trials), 0)

    def test_trial_access(self):
        if len(self.patient.trials_df) > 0:
            trial = self.patient.get_trial(0)
            self.assertIsNotNone(trial)
            self.assertIn("trial_type", trial.index)

    def test_multi_session_patient(self):
        patient_multi = self.loader.get_patient("CON005")

        sessions = patient_multi.list_sessions()
        self.assertEqual(len(sessions), 2)

        try:
            edfs = patient_multi.get_raw()
            self.assertIsInstance(edfs, dict)
            self.assertEqual(len(edfs), 2)

            raw = patient_multi.get_raw("2025-02-14")
            self.assertIsInstance(raw, mne.io.Raw)
        except UnifiedDataLoadingError:
            pass


if __name__ == "__main__":
    unittest.main()
