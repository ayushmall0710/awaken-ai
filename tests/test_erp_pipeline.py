"""Tests for ERP Pipeline (ENG-02b)."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.data_processing.erp_pipeline import ERP_CONFIG, OddballERPPipeline

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_unified_loader():
    """Prevent UnifiedDataLoader from touching the filesystem in every test."""
    with patch("src.data_processing.erp_pipeline.UnifiedDataLoader") as mock:
        mock_instance = Mock()
        mock_instance.load_aligned_trials = Mock(return_value=pd.DataFrame())
        mock_instance.load_eeg = Mock()
        mock_instance.load_clean_epochs = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_output_dir():
    """Temporary directory for pipeline outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_aligned_events():
    """Mock aligned events DataFrame from ENG-02."""
    return pd.DataFrame(
        [
            {
                "patient_id": "TEST001",
                "date": "2024-01-01",
                "session_id": "1",
                "trial_type": "oddball",
                "start_time": 1704110400.0,
                "end_time": 1704110432.0,
                "duration": 32.0,
                "sentences": [
                    {"event": "standard", "event_start": 1704110401.0},
                    {
                        "event": "rare",
                        "event_start": 1704110405.0,
                        "correlation_score": 0.95,
                    },
                    {"event": "standard", "event_start": 1704110407.0},
                    {
                        "event": "rare",
                        "event_start": 1704110410.0,
                        "correlation_score": 0.92,
                    },
                    {"event": "standard", "event_start": 1704110412.0},
                    {
                        "event": "rare",
                        "event_start": 1704110415.0,
                        "correlation_score": 0.98,
                    },
                ],
                "dc_channel": "DC1",
                "alignment_method": "peak_detection",
            }
        ]
    )


@pytest.fixture
def mock_raw_eeg():
    """Create mock MNE Raw object with synthetic EEG data."""
    sfreq = 500.0
    duration = 60.0
    n_samples = int(sfreq * duration)
    n_channels = 10

    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "Pz", "O1", "O2", "DC1"]
    ch_types = ["eeg"] * 9 + ["stim"]

    data = np.random.randn(n_channels, n_samples) * 1e-5
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    return raw


@pytest.fixture
def mock_eng03_epochs():
    """Synthetic 35s ENG-03 oddball epochs with metadata (EEG-only channels)."""
    sfreq = 512.0
    window_sec = 35.0
    n_channels = 9
    n_samples = int(sfreq * window_sec) + 1
    n_epochs = 3

    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "Pz", "O1", "O2"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    data = np.random.randn(n_epochs, n_channels, n_samples) * 1e-5

    trial_start_unix = 1704110400.0
    metadata = pd.DataFrame(
        {
            "start_time_unix": [
                trial_start_unix,
                trial_start_unix + 40.0,
                trial_start_unix + 80.0,
            ],
            "end_time_unix": [
                trial_start_unix + window_sec,
                trial_start_unix + 40.0 + window_sec,
                trial_start_unix + 80.0 + window_sec,
            ],
            "trial_type": ["oddball"] * n_epochs,
            "patient_id": ["TEST001"] * n_epochs,
        }
    )

    epochs = mne.EpochsArray(data, info=info, tmin=0.0, baseline=None, verbose=False)
    epochs.metadata = metadata
    return epochs


@pytest.fixture
def mock_epochs(mock_raw_eeg):
    """Create mock MNE Epochs object."""
    sfreq = mock_raw_eeg.info["sfreq"]
    event_times = [5.0, 10.0, 15.0]
    events = np.array([[int(t * sfreq), 0, 1] for t in event_times])

    picks = mne.pick_types(mock_raw_eeg.info, eeg=True)
    epochs = mne.Epochs(
        mock_raw_eeg,
        events,
        event_id={"rare": 1},
        tmin=-0.2,
        tmax=0.7,
        baseline=(None, 0),
        picks=picks,
        preload=True,
        verbose=False,
    )
    return epochs


@pytest.fixture
def mock_evoked(mock_epochs):
    """Create mock MNE Evoked object (averaged ERP)."""
    return mock_epochs.average()


# ── TestOddballERPPipeline ───────────────────────────────────────────────────


class TestOddballERPPipeline:
    def test_initialization(self, temp_output_dir):
        """Test pipeline initialization."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        assert pipeline.output_dir == temp_output_dir
        assert not pipeline.verbose

        assert not (temp_output_dir / "epochs").exists(), "epochs dir should not be created"
        assert (temp_output_dir / "erps").exists()
        assert (temp_output_dir / "features").exists()
        assert (temp_output_dir / "plots" / "erp").exists()
        assert (temp_output_dir / "qc").exists()

    def test_extract_rare_events(self, temp_output_dir, mock_aligned_events):
        """Test extraction of rare events from aligned trials."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        rare_events = pipeline._extract_rare_events(mock_aligned_events)

        assert len(rare_events) == 3
        assert all("timestamp_unix" in e for e in rare_events)
        assert all("date" in e for e in rare_events)
        assert all("trial_idx" in e for e in rare_events)

        expected_timestamps = [1704110405.0, 1704110410.0, 1704110415.0]
        actual_timestamps = [e["timestamp_unix"] for e in rare_events]
        assert actual_timestamps == expected_timestamps

    def test_compute_erp(self, temp_output_dir, mock_epochs):
        """Test ERP computation (averaging) returns tuple of (erp, sem)."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        erp, erp_sem = pipeline._compute_erp(mock_epochs)

        assert isinstance(erp, mne.Evoked)
        assert isinstance(erp_sem, mne.Evoked)
        assert erp.data.ndim == 2
        assert erp.data.shape[0] == len(mock_epochs.ch_names)
        assert erp_sem.data.shape == erp.data.shape
        assert np.allclose(erp.times, mock_epochs.times)

    def test_detect_p300_peak(self, temp_output_dir, mock_evoked):
        """Test P300 peak detection."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        pz_idx = mock_evoked.ch_names.index("Pz")
        time_idx_400ms = np.argmin(np.abs(mock_evoked.times - 0.4))
        mock_evoked.data[pz_idx, time_idx_400ms] += 5e-6

        result = pipeline._detect_p300_peak(mock_evoked, "Pz")

        assert "amplitude" in result
        assert "latency" in result
        assert not np.isnan(result["amplitude"])
        assert not np.isnan(result["latency"])
        assert 300 <= result["latency"] <= 600

    def test_detect_p300_missing_electrode(self, temp_output_dir, mock_evoked):
        """Test P300 detection with missing electrode."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        result = pipeline._detect_p300_peak(mock_evoked, "MISSING")

        assert np.isnan(result["amplitude"])
        assert np.isnan(result["latency"])

    def test_extract_standard_events(self, temp_output_dir, mock_aligned_events):
        """Test extraction of standard (frequent) events."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        standard_events = pipeline._extract_standard_events(mock_aligned_events)

        assert len(standard_events) == 3
        assert all(e["timestamp_unix"] > 0 for e in standard_events)
        assert all("timestamp_unix" in e and "date" in e for e in standard_events)

    def test_compute_difference_erp(self, temp_output_dir, mock_evoked):
        """Test difference ERP computation (rare - standard)."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Create two ERPs with slight difference
        rare_erp = mock_evoked.copy()
        standard_erp = mock_evoked.copy()
        standard_erp.data = standard_erp.data * 0.5  # Standard is 50% amplitude

        diff_erp = pipeline._compute_difference_erp(rare_erp, standard_erp)

        assert isinstance(diff_erp, mne.Evoked)
        assert diff_erp.data.shape == rare_erp.data.shape
        # Difference should be ~50% of rare (rare - 0.5*rare = 0.5*rare)
        assert np.allclose(diff_erp.data, rare_erp.data * 0.5, rtol=0.01)

    def test_p3a_subtype_detection(self, temp_output_dir, mock_evoked):
        """Test P3a subtype when Fz amplitude > Pz amplitude."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Zero out midline channels to avoid random noise influencing subtype
        for elec in ["Fz", "Cz", "Pz"]:
            elec_idx = mock_evoked.ch_names.index(elec)
            mock_evoked.data[elec_idx, :] = 0.0

        # Boost Fz amplitude at P300 window relative to Pz
        fz_idx = mock_evoked.ch_names.index("Fz")
        pz_idx = mock_evoked.ch_names.index("Pz")
        time_idx_400ms = np.argmin(np.abs(mock_evoked.times - 0.4))

        mock_evoked.data[fz_idx, time_idx_400ms] = 8e-6  # Fz: +8µV
        mock_evoked.data[pz_idx, time_idx_400ms] = 2e-6  # Pz: +2µV

        composite = pipeline._compute_composite_p300(mock_evoked, "TEST001")

        assert composite.get("p300_subtype") == "P3a"
        assert composite["best_electrode"] == "Fz"

    def test_p300_absent_subtype_when_no_valid_electrodes(self, temp_output_dir, mock_evoked):
        """Test 'absent' subtype when n_valid_electrodes == 0."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Zero out all P300 amplitudes to make them invalid (negative)
        for elec in ["Fz", "Cz", "Pz"]:
            elec_idx = mock_evoked.ch_names.index(elec)
            mock_evoked.data[elec_idx, :] = -1e-6  # Force negative amplitude

        composite = pipeline._compute_composite_p300(mock_evoked, "TEST001")

        assert composite["n_valid_electrodes"] == 0
        assert composite.get("p300_subtype") == "absent"

    def test_qc_notes_with_subtype_separator(self, temp_output_dir, mock_evoked):
        """Test qc_notes includes subtype with semicolon separator."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        pz_idx = mock_evoked.ch_names.index("Pz")
        time_idx_400ms = np.argmin(np.abs(mock_evoked.times - 0.4))
        mock_evoked.data[pz_idx, time_idx_400ms] += 5e-6

        features = pipeline._quantify_p300(mock_evoked, "TEST001", "2024-01-01", n_epochs=3)

        notes = features["qc_notes"]
        assert ";" in notes
        # Subtype text should be present (case-insensitive)
        lowered = notes.lower()
        assert "p3b" in lowered or "p3a" in lowered or "mixed" in lowered

    def test_quantify_p300(self, temp_output_dir, mock_evoked):
        """Test P300 quantification feature schema."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = pipeline._quantify_p300(mock_evoked, "TEST001", "2024-01-01", n_epochs=3)

        assert features["patient_id"] == "TEST001"
        assert features["date"] == "2024-01-01"
        assert features["n_epochs"] == 3
        assert "processing_timestamp" in features
        assert "baseline_std_uV" in features

        for elec in ["Pz", "Cz", "Fz"]:
            assert f"p300_amplitude_{elec}_uV" in features
            assert f"p300_latency_{elec}_ms" in features

        assert "p300_composite_amplitude_uV" in features
        assert "p300_composite_latency_ms" in features
        assert "p300_best_electrode" in features
        assert "p300_n_valid_electrodes" in features
        assert "p300_n_flagged_electrodes" in features

        assert "qc_notes" in features
        assert isinstance(features["qc_notes"], str)

        assert "p300_amplitude_uV" in features
        assert "p300_latency_ms" in features

        assert "n_rejected" not in features
        assert "timezone_offset_seconds" not in features

    @patch("src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR")
    def test_load_aligned_trials(self, mock_aligned_dir, temp_output_dir, mock_aligned_events):
        """Test loading aligned trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            aligned_dir = Path(tmpdir)
            mock_aligned_dir.return_value = aligned_dir

            aligned_file = aligned_dir / "TEST001_events.parquet"
            mock_aligned_events.to_parquet(aligned_file)

            mock_aligned_dir.__truediv__ = lambda self, x: aligned_dir / x

            pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

            with patch(
                "src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR",
                aligned_dir,
            ):
                trials = pipeline._load_aligned_trials("TEST001")

            assert len(trials) == 1
            assert trials.iloc[0]["trial_type"] == "oddball"

    def test_save_outputs(self, temp_output_dir, mock_epochs, mock_evoked):
        """Test saving ERPs and three feature tables.

        900ms epochs are not saved (regenerated from ENG-03).
        Three master tables created: clinical, electrode_detail, mapping_qc.
        """
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = {
            "patient_id": "TEST001",
            "date": "2024-01-01",
            "n_epochs": 5,
            "n_standard_epochs": 8,
            "baseline_std_uV": 1.5,
            "p300_amplitude_uV": 4.5,
            "p300_latency_ms": 350,
            "p300_amplitude_Pz_uV": 5.0,
            "p300_latency_Pz_ms": 360,
            "p300_amplitude_Cz_uV": 2.0,
            "p300_latency_Cz_ms": 340,
            "p300_amplitude_Fz_uV": 1.0,
            "p300_latency_Fz_ms": 370,
            "diff_amplitude_Pz_uV": 3.0,
            "diff_latency_Pz_ms": 350,
            "diff_amplitude_Cz_uV": 1.5,
            "diff_latency_Cz_ms": 340,
            "diff_amplitude_Fz_uV": 0.8,
            "diff_latency_Fz_ms": 360,
            "p300_best_electrode": "Pz",
            "p300_subtype": "P3b",
            "p300_n_valid_electrodes": 3,
            "p300_n_flagged_electrodes": 0,
            "qc_notes": "All electrodes valid",
            "n_rare_events": 10,
            "n_mapped": 9,
            "n_unmapped": 1,
            "n_boundary_clipped": 0,
            "mapping_rate": 0.9,
            "processing_timestamp": "2024-01-01T00:00:00",
        }

        pipeline._save_outputs("TEST001", "2024-01-01", mock_epochs, mock_evoked, features)

        # Check ERPs saved
        erp_file = temp_output_dir / "erps" / "TEST001_2024-01-01_oddball-ave.fif"
        assert erp_file.exists()

        # Check three master tables created
        clinical_file = temp_output_dir / "features" / "p300_oddball_clinical.parquet"
        detail_file = temp_output_dir / "features" / "p300_oddball_electrode_detail.parquet"
        qc_file = temp_output_dir / "features" / "p300_oddball_mapping_qc.parquet"

        assert clinical_file.exists(), "Clinical table should exist"
        assert detail_file.exists(), "Electrode detail table should exist"
        assert qc_file.exists(), "Mapping QC table should exist"

        # Verify clinical table structure
        clinical = pd.read_parquet(clinical_file)
        assert len(clinical) == 1
        assert clinical.iloc[0]["patient_id"] == "TEST001"
        assert clinical.iloc[0]["session_date"] == "2024-01-01"
        assert "p300_diff_amplitude_Pz_uV" in clinical.columns
        assert "qc_pass" in clinical.columns

        # Verify electrode detail table structure
        detail = pd.read_parquet(detail_file)
        assert len(detail) == 3  # One row per electrode (Fz, Cz, Pz)
        assert set(detail["electrode"].unique()) == {"Fz", "Cz", "Pz"}
        assert "is_valid" in detail.columns
        assert "flagged_reason" in detail.columns

        # Verify mapping QC table structure
        qc = pd.read_parquet(qc_file)
        assert len(qc) == 1
        assert qc.iloc[0]["n_rare_mapped"] == 9
        assert "pipeline_version" in qc.columns

        loaded_erp = mne.read_evokeds(erp_file, verbose=False)[0]
        assert loaded_erp.data.shape == mock_evoked.data.shape

    def test_plot_individual_erp(self, temp_output_dir, mock_evoked):
        """Test individual ERP plotting."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        pipeline._plot_individual_erp(mock_evoked, "TEST001", "2024-01-01")

        plot_file = temp_output_dir / "plots" / "erp" / "TEST001_2024-01-01_oddball_erp.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_compute_grand_average(self, temp_output_dir, mock_evoked):
        """Test grand average computation."""
        erps_dir = temp_output_dir / "erps"
        erps_dir.mkdir(parents=True, exist_ok=True)

        mock_evoked.save(erps_dir / "TEST001_2024-01-01_oddball-ave.fif", overwrite=True)
        mock_evoked.save(erps_dir / "TEST002_2024-01-02_oddball-ave.fif", overwrite=True)
        mock_evoked.save(erps_dir / "grand_average_oddball-ave.fif", overwrite=True)

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        grand_avg = pipeline.compute_grand_average()

        assert isinstance(grand_avg, mne.Evoked)

        grand_avg_file = temp_output_dir / "erps" / "grand_average_oddball-ave.fif"
        assert grand_avg_file.exists()

        grand_avg_plot = temp_output_dir / "plots" / "erp" / "grand_average_oddball_erp.png"
        assert grand_avg_plot.exists()

        assert grand_avg.nave == 2

    def test_update_master_feature_tables_deduplicates_session(self, temp_output_dir):
        """Master tables should keep latest row for the same patient/session key."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Build clinical table (Table 1)
        clinical_df1 = pd.DataFrame(
            [{"patient_id": "TEST001", "session_date": "2024-01-01", "n_rare_epochs": 3, "p300_amplitude_uV": 4.0}]
        )
        clinical_df2 = pd.DataFrame(
            [{"patient_id": "TEST001", "session_date": "2024-01-01", "n_rare_epochs": 5, "p300_amplitude_uV": 6.0}]
        )

        # Build electrode detail table (Table 2)
        detail_df1 = pd.DataFrame(
            [
                {"patient_id": "TEST001", "session_date": "2024-01-01", "electrode": "Pz", "p300_amplitude_uV": 4.0},
            ]
        )
        detail_df2 = pd.DataFrame(
            [
                {"patient_id": "TEST001", "session_date": "2024-01-01", "electrode": "Pz", "p300_amplitude_uV": 6.0},
            ]
        )

        # Build mapping QC table (Table 3)
        qc_df1 = pd.DataFrame([{"patient_id": "TEST001", "session_date": "2024-01-01", "n_rare_mapped": 3}])
        qc_df2 = pd.DataFrame([{"patient_id": "TEST001", "session_date": "2024-01-01", "n_rare_mapped": 5}])

        # First batch
        pipeline._update_master_feature_tables(clinical_df1, detail_df1, qc_df1)

        # Second batch (should dedupe and keep latest)
        pipeline._update_master_feature_tables(clinical_df2, detail_df2, qc_df2)

        # Verify clinical table
        clinical_path = temp_output_dir / "features" / "p300_oddball_clinical.parquet"
        clinical = pd.read_parquet(clinical_path)
        assert len(clinical) == 1
        assert int(clinical.iloc[0]["n_rare_epochs"]) == 5
        assert float(clinical.iloc[0]["p300_amplitude_uV"]) == 6.0

        # Verify electrode detail table
        detail_path = temp_output_dir / "features" / "p300_oddball_electrode_detail.parquet"
        detail = pd.read_parquet(detail_path)
        assert len(detail) == 1
        assert float(detail.iloc[0]["p300_amplitude_uV"]) == 6.0

        # Verify mapping QC table
        qc_path = temp_output_dir / "features" / "p300_oddball_mapping_qc.parquet"
        qc = pd.read_parquet(qc_path)
        assert len(qc) == 1
        assert int(qc.iloc[0]["n_rare_mapped"]) == 5

    def test_generate_qc_report(self, temp_output_dir):
        """Test QC report generation."""
        features_dir = temp_output_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        features = pd.DataFrame(
            [
                {
                    "patient_id": "TEST001",
                    "date": "2024-01-01",
                    "n_epochs": 5,
                    "n_rejected": 0,
                    "p300_amplitude_uV": 4.5,
                    "p300_latency_ms": 387,
                    "baseline_std_uV": 0.8,
                },
                {
                    "patient_id": "TEST002",
                    "date": "2024-01-02",
                    "n_epochs": 3,
                    "n_rejected": 0,
                    "p300_amplitude_uV": 3.2,
                    "p300_latency_ms": 412,
                    "baseline_std_uV": 1.0,
                },
            ]
        )
        features.to_parquet(features_dir / "p300_features.parquet")

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        report = pipeline.generate_qc_report()

        assert report["total_patients"] == 2
        assert report["total_sessions"] == 2
        assert report["total_epochs"] == 8
        assert "p300_detection_rate" in report
        assert "mean_amplitude_uV" in report
        assert "mean_latency_ms" in report
        assert "by_patient" in report

        report_file = temp_output_dir / "qc" / "erp_qc_report.json"
        assert report_file.exists()

    def test_get_patients_with_oddball(self, temp_output_dir):
        """Test getting list of patients with oddball data."""
        aligned_dir = temp_output_dir / "aligned_events"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        df_with_oddball = pd.DataFrame([{"trial_type": "oddball"}])
        df_with_oddball.to_parquet(aligned_dir / "TEST001_events.parquet")

        df_without_oddball = pd.DataFrame([{"trial_type": "language"}])
        df_without_oddball.to_parquet(aligned_dir / "TEST002_events.parquet")

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        with patch("src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR", aligned_dir):
            patient_ids = pipeline._get_patients_with_oddball()

        assert len(patient_ids) == 1
        assert "TEST001" in patient_ids
        assert "TEST002" not in patient_ids

    def test_process_all_patients_empty(self, temp_output_dir):
        """Test batch processing with no patients."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        with patch.object(pipeline, "_get_patients_with_oddball", return_value=[]):
            result = pipeline.process_all_patients()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ── TestERPConfig ────────────────────────────────────────────────────────────


class TestERPConfig:
    def test_config_values(self):
        """Sanity-check ERP_CONFIG bounds."""
        assert ERP_CONFIG["tmin"] < 0
        assert ERP_CONFIG["tmax"] > 0
        assert ERP_CONFIG["baseline"][1] == 0
        assert ERP_CONFIG["p300_window"][0] >= 0.3
        assert ERP_CONFIG["p300_window"][1] <= 0.7
        assert ERP_CONFIG["min_epochs"] >= 2


# ── TestTrialMapping (new methods) ──────────────────────────────────────────


class TestTrialMapping:
    def test_build_trial_windows(self, temp_output_dir, mock_eng03_epochs):
        """Verify trial-windows DataFrame has expected schema and values."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        tw = pipeline._build_trial_windows(mock_eng03_epochs)

        assert set(tw.columns) == {"eng03_epoch_idx", "start_time_unix", "end_time_unix", "window_sec"}
        assert len(tw) == 3
        assert list(tw["eng03_epoch_idx"]) == [0, 1, 2]
        assert tw["window_sec"].iloc[0] == 35.0
        assert tw["end_time_unix"].iloc[0] > tw["start_time_unix"].iloc[0]

    def test_build_trial_windows_no_metadata_raises(self, temp_output_dir):
        """Epochs without metadata should raise ValueError."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        info = mne.create_info(ch_names=["Fz"], sfreq=256.0, ch_types="eeg")
        epochs = mne.EpochsArray(np.zeros((1, 1, 100)), info=info, verbose=False)

        with pytest.raises(ValueError, match="no metadata"):
            pipeline._build_trial_windows(epochs)

    def test_map_events_to_trials_all_mapped(self, temp_output_dir, mock_eng03_epochs):
        """All events inside trial windows should be mapped."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]

        rare_events = [
            {"timestamp_unix": trial_start + 5.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 15.0, "date": "2024-01-01", "trial_idx": 0},
        ]

        mapped_df, diag = pipeline._map_events_to_trials(
            rare_events,
            tw,
            sfreq=float(mock_eng03_epochs.info["sfreq"]),
        )

        assert diag["n_rare_events"] == 2
        assert diag["n_mapped"] == 2
        assert diag["n_unmapped"] == 0
        assert diag["mapping_rate"] == 1.0
        assert len(mapped_df) == 2

    def test_map_events_to_trials_some_unmapped(self, temp_output_dir, mock_eng03_epochs):
        """Events outside any trial window should be silently dropped."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]

        rare_events = [
            {"timestamp_unix": trial_start + 5.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 999.0, "date": "2024-01-01", "trial_idx": 0},
        ]

        mapped_df, diag = pipeline._map_events_to_trials(
            rare_events,
            tw,
            sfreq=float(mock_eng03_epochs.info["sfreq"]),
        )

        assert diag["n_mapped"] == 1
        assert diag["n_unmapped"] == 1
        assert len(mapped_df) == 1

    def test_map_events_to_trials_boundary_clip(self, temp_output_dir, mock_eng03_epochs):
        """Events whose sub-epoch crosses the trial edge should be excluded."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]

        rare_events = [
            {"timestamp_unix": trial_start + 0.05, "date": "2024-01-01", "trial_idx": 0},
        ]

        mapped_df, diag = pipeline._map_events_to_trials(
            rare_events,
            tw,
            sfreq=float(mock_eng03_epochs.info["sfreq"]),
        )

        assert diag["n_boundary_clipped"] == 1
        assert len(mapped_df) == 0

    def test_extract_subepochs_shape(self, temp_output_dir, mock_eng03_epochs):
        """Extracted sub-epochs should have correct shape and timing."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]

        rare_events = [
            {"timestamp_unix": trial_start + 10.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 20.0, "date": "2024-01-01", "trial_idx": 0},
        ]

        mapped_df, _ = pipeline._map_events_to_trials(
            rare_events,
            tw,
            sfreq=float(mock_eng03_epochs.info["sfreq"]),
        )

        sub = pipeline._extract_subepochs(mock_eng03_epochs, mapped_df)

        assert len(sub) == 2
        assert np.isclose(sub.tmin, ERP_CONFIG["tmin"], atol=1.0 / mock_eng03_epochs.info["sfreq"])
        n_expected_channels = len(mock_eng03_epochs.ch_names)
        assert sub.get_data().shape[1] == n_expected_channels

    def test_extract_subepochs_empty(self, temp_output_dir, mock_eng03_epochs):
        """Empty mapped_df should produce an Epochs object with zero good epochs."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        empty_mapped = pd.DataFrame(
            columns=["timestamp_unix", "eng03_epoch_idx", "offset_sec", "start_sample", "end_sample"]
        )
        sub = pipeline._extract_subepochs(mock_eng03_epochs, empty_mapped)

        assert len(sub) == 0
        assert hasattr(sub, "info")


# ── TestIntegration ──────────────────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.integration
    def test_full_pipeline_single_patient(self, temp_output_dir, mock_aligned_events, mock_eng03_epochs):
        """Test full pipeline on single patient using ENG-03 epochs."""
        aligned_dir = temp_output_dir / "aligned_events"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        trial_start = float(mock_eng03_epochs.metadata["start_time_unix"].iloc[0])
        mock_aligned_events = pd.DataFrame(
            [
                {
                    "patient_id": "TEST001",
                    "date": "2024-01-01",
                    "session_id": "1",
                    "trial_type": "oddball",
                    "start_time": trial_start,
                    "end_time": trial_start + 35.0,
                    "duration": 35.0,
                    "sentences": [
                        {"event": "standard", "event_start": trial_start + 1.0},
                        {"event": "rare", "event_start": trial_start + 5.0, "correlation_score": 0.95},
                        {"event": "rare", "event_start": trial_start + 15.0, "correlation_score": 0.92},
                        {"event": "rare", "event_start": trial_start + 25.0, "correlation_score": 0.98},
                    ],
                    "dc_channel": "DC1",
                    "alignment_method": "peak_detection",
                }
            ]
        )
        mock_aligned_events.to_parquet(aligned_dir / "TEST001_events.parquet")

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)
        pipeline.loader.load_clean_epochs = Mock(return_value=mock_eng03_epochs)

        with patch(
            "src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR",
            aligned_dir,
        ):
            result = pipeline.process_patient("TEST001")

        assert result["status"] == "success"
        assert "features" in result

        assert not (temp_output_dir / "epochs").exists(), "epochs dir should not be created"
        assert (temp_output_dir / "erps").exists()
        assert (temp_output_dir / "plots" / "erp").exists()

    def test_validate_p300_electrode_positive(self, temp_output_dir):
        """Test validation of expected P300."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        validation = pipeline._validate_p300_electrode("Pz", 6.5, 420, "TEST001")

        assert validation["is_valid"] is True
        assert validation["is_positive"] is True
        assert validation["is_on_time"] is True
        assert validation["is_expected_latency"] is True
        assert len(validation["issues"]) == 0

    def test_validate_p300_electrode_inverted(self, temp_output_dir):
        """Test detection of negative-polarity P300."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        validation = pipeline._validate_p300_electrode("Pz", -7.14, 537, "CON009")

        assert validation["is_valid"] is False
        assert validation["is_positive"] is False
        assert "negative_or_zero_amplitude" in validation["issues"]

    def test_validate_p300_electrode_late_latency(self, temp_output_dir):
        """Test detection of abnormal latency."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        validation = pipeline._validate_p300_electrode("Pz", 5.0, 650, "TEST001")

        assert validation["is_valid"] is False
        assert validation["is_on_time"] is False
        assert "latency_out_of_range" in validation["issues"]

    def test_validate_p300_electrode_atypical_latency(self, temp_output_dir):
        """Test detection of atypical but acceptable latency."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        validation = pipeline._validate_p300_electrode("Pz", 5.0, 280, "TEST001")

        assert validation["is_valid"] is True
        assert validation["is_expected_latency"] is False
        assert "latency_atypical" in validation["issues"]

    def test_validate_p300_electrode_nan(self, temp_output_dir):
        """Test handling of NaN values."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        validation = pipeline._validate_p300_electrode("Pz", np.nan, 420, "TEST001")

        assert validation["is_valid"] is False
        assert "missing_data" in validation["issues"]

    def test_compute_composite_p300_all_valid(self, temp_output_dir, mock_evoked):
        """Test composite computation with all electrodes valid."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        composite = pipeline._compute_composite_p300(mock_evoked, "TEST001")

        assert composite["n_valid_electrodes"] >= 1
        assert composite["n_flagged_electrodes"] <= 3
        assert composite["best_electrode"] in ["Fz", "Cz", "Pz", None]

        if composite["n_valid_electrodes"] > 0:
            assert not np.isnan(composite["composite_amplitude"])
            assert not np.isnan(composite["composite_latency"])
            assert composite["best_electrode"] is not None
        else:
            assert np.isnan(composite["composite_amplitude"])

    def test_quantify_p300_includes_composite(self, temp_output_dir, mock_evoked):
        """Test that quantification includes composite features."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = pipeline._quantify_p300(mock_evoked, "TEST001", "2024-01-01", n_epochs=3)

        assert "p300_composite_amplitude_uV" in features
        assert "p300_composite_latency_ms" in features
        assert "p300_n_valid_electrodes" in features
        assert "p300_best_electrode" in features
        assert "p300_n_flagged_electrodes" in features

        assert "qc_notes" in features
        assert isinstance(features["qc_notes"], str)

        assert "p300_valid_electrodes" not in features
        assert "p300_flagged_electrodes" not in features
        assert "p300_composite_amplitude_std_uV" not in features
        assert "Pz_is_valid" not in features
        assert "Pz_is_positive" not in features
        assert "Pz_issues" not in features

        for elec in ["Pz", "Cz", "Fz"]:
            assert f"p300_amplitude_{elec}_uV" in features
            assert f"p300_latency_{elec}_ms" in features

        assert "p300_amplitude_uV" in features
        assert "p300_latency_ms" in features

    def test_quantify_p300_custom_electrodes(self, temp_output_dir, mock_evoked):
        """Test custom electrode analysis mode."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = pipeline._quantify_p300(
            mock_evoked,
            "TEST001",
            "2024-01-01",
            n_epochs=3,
            custom_electrodes=["Fz", "Cz"],
        )

        assert features["patient_id"] == "TEST001"
        assert features["date"] == "2024-01-01"

        assert "p300_amplitude_Fz_uV" in features
        assert "p300_latency_Fz_ms" in features
        assert "p300_amplitude_Cz_uV" in features
        assert "p300_latency_Cz_ms" in features

        assert "p300_amplitude_Pz_uV" not in features
        assert "p300_composite_amplitude_uV" not in features
        assert "p300_n_valid_electrodes" not in features

        assert "qc_notes" in features
        assert "Custom electrode analysis" in features["qc_notes"]


# ── TestENG03Integration ────────────────────────────────────────────────────


class TestENG03Integration:
    def test_process_session_requires_eng03_epochs(self, temp_output_dir, mock_aligned_events):
        """_process_session returns error if ENG-03 epochs are not on disk."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)
        pipeline.loader.load_clean_epochs = Mock(
            side_effect=FileNotFoundError("oddball-epo.fif not found"),
        )

        result = pipeline._process_session("TEST001", "2024-01-01", mock_aligned_events)

        assert result["status"] == "error"
        assert "ENG-03" in result["error"]

    def test_process_session_with_eng03_epochs(
        self,
        temp_output_dir,
        mock_aligned_events,
        mock_eng03_epochs,
    ):
        """_process_session succeeds when ENG-03 epochs are available."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)
        pipeline.loader.load_clean_epochs = Mock(return_value=mock_eng03_epochs)

        trial_start = float(mock_eng03_epochs.metadata["start_time_unix"].iloc[0])
        aligned = pd.DataFrame(
            [
                {
                    "patient_id": "TEST001",
                    "date": "2024-01-01",
                    "session_id": "1",
                    "trial_type": "oddball",
                    "start_time": trial_start,
                    "end_time": trial_start + 35.0,
                    "duration": 35.0,
                    "sentences": [
                        {"event": "rare", "event_start": trial_start + 5.0, "correlation_score": 0.9},
                        {"event": "rare", "event_start": trial_start + 15.0, "correlation_score": 0.9},
                        {"event": "rare", "event_start": trial_start + 25.0, "correlation_score": 0.9},
                    ],
                    "dc_channel": "DC1",
                    "alignment_method": "peak_detection",
                }
            ]
        )

        result = pipeline._process_session("TEST001", "2024-01-01", aligned)
        assert result["patient_id"] == "TEST001"
        assert result["status"] in ("success", "insufficient_epochs")
