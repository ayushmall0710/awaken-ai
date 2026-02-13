"""
Unit tests for ERP Pipeline (ENG-02b)
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.data_processing.erp_pipeline import ERP_CONFIG, OddballERPPipeline


# Patch UnifiedDataLoader for all tests to avoid requiring data files in CI
@pytest.fixture(autouse=True)
def mock_unified_loader():
    """Auto-mock UnifiedDataLoader to avoid requiring data files."""
    with patch("src.data_processing.erp_pipeline.UnifiedDataLoader") as mock:
        mock_instance = Mock()
        mock_instance.load_aligned_trials = Mock(return_value=pd.DataFrame())
        mock_instance.load_eeg = Mock()
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
    # Create synthetic data
    sfreq = 500.0  # Sampling frequency
    duration = 60.0  # 60 seconds
    n_samples = int(sfreq * duration)
    n_channels = 10

    # Channel names (including Pz, Cz, Fz)
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "Pz", "O1", "O2", "DC1"]
    ch_types = ["eeg"] * 9 + ["stim"]

    # Create synthetic data (random noise)
    data = np.random.randn(n_channels, n_samples) * 1e-5  # Scale to realistic EEG amplitude

    # Create Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    return raw


@pytest.fixture
def mock_epochs(mock_raw_eeg):
    """Create mock MNE Epochs object."""
    # Create events at specific time points
    sfreq = mock_raw_eeg.info["sfreq"]
    event_times = [5.0, 10.0, 15.0]  # seconds
    events = np.array([[int(t * sfreq), 0, 1] for t in event_times])

    # Create epochs
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


class TestOddballERPPipeline:
    """Test suite for OddballERPPipeline class."""

    def test_initialization(self, temp_output_dir):
        """Test pipeline initialization."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        assert pipeline.output_dir == temp_output_dir
        assert not pipeline.verbose

        # Check that output directories were created
        assert (temp_output_dir / "epochs").exists()
        assert (temp_output_dir / "erps").exists()
        assert (temp_output_dir / "features").exists()
        assert (temp_output_dir / "plots" / "erp").exists()
        assert (temp_output_dir / "qc").exists()

    def test_extract_rare_events(self, temp_output_dir, mock_aligned_events):
        """Test extraction of rare events from aligned trials."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        rare_events = pipeline._extract_rare_events(mock_aligned_events)

        # Should extract 3 rare events
        assert len(rare_events) == 3

        # Check structure
        assert all("timestamp_unix" in e for e in rare_events)
        assert all("date" in e for e in rare_events)
        assert all("trial_idx" in e for e in rare_events)

        # Check timestamps
        expected_timestamps = [1704110405.0, 1704110410.0, 1704110415.0]
        actual_timestamps = [e["timestamp_unix"] for e in rare_events]
        assert actual_timestamps == expected_timestamps

    def test_timestamp_conversion(self, temp_output_dir, mock_raw_eeg):
        """Test conversion from Unix to EDF-relative time."""
        # EDF starts at 2024-01-01 00:00:00 UTC
        edf_start_unix = mock_raw_eeg.info["meas_date"].timestamp()

        # Event at 00:00:10 UTC
        event_unix = edf_start_unix + 10.0

        rare_events = [{"timestamp_unix": event_unix, "date": "2024-01-01", "trial_idx": 0}]

        # Add EDF-relative times (simulating what _create_epochs does)
        sfreq = mock_raw_eeg.info["sfreq"]
        for event in rare_events:
            event["edf_time"] = event["timestamp_unix"] - edf_start_unix
            event["sample_idx"] = int(event["edf_time"] * sfreq)

        # Check conversion
        assert rare_events[0]["edf_time"] == 10.0
        assert rare_events[0]["sample_idx"] == int(10.0 * sfreq)

    def test_create_epochs(self, temp_output_dir, mock_raw_eeg):
        """Test epoch creation."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        edf_start_unix = mock_raw_eeg.info["meas_date"].timestamp()

        # Create rare events at known times
        rare_events = [
            {
                "timestamp_unix": edf_start_unix + 5.0,
                "date": "2024-01-01",
                "trial_idx": 0,
            },
            {
                "timestamp_unix": edf_start_unix + 10.0,
                "date": "2024-01-01",
                "trial_idx": 0,
            },
            {
                "timestamp_unix": edf_start_unix + 15.0,
                "date": "2024-01-01",
                "trial_idx": 0,
            },
        ]

        epochs = pipeline._create_epochs(mock_raw_eeg, rare_events)

        # Check epochs properties
        assert len(epochs) == 3
        assert epochs.tmin == ERP_CONFIG["tmin"]
        assert epochs.tmax == ERP_CONFIG["tmax"]

        # Check shape: (n_epochs, n_eeg_channels, n_timepoints)
        data = epochs.get_data()
        assert data.shape[0] == 3  # 3 epochs
        assert data.shape[1] == 9  # 9 EEG channels (excluding DC1)

        # Check baseline correction was applied
        baseline_data = data[:, :, : int(0.2 * mock_raw_eeg.info["sfreq"])]
        assert np.abs(np.mean(baseline_data)) < 1e-6  # Should be close to zero

    def test_compute_erp(self, temp_output_dir, mock_epochs):
        """Test ERP computation (averaging)."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        erp = pipeline._compute_erp(mock_epochs)

        # Check that ERP is an Evoked object
        assert isinstance(erp, mne.Evoked)

        # Check shape: (n_channels, n_timepoints)
        assert erp.data.ndim == 2
        assert erp.data.shape[0] == len(mock_epochs.ch_names)

        # Check times match
        assert np.allclose(erp.times, mock_epochs.times)

    def test_detect_p300_peak(self, temp_output_dir, mock_evoked):
        """Test P300 peak detection."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Add synthetic P300 peak to Pz channel
        pz_idx = mock_evoked.ch_names.index("Pz")
        time_idx_400ms = np.argmin(np.abs(mock_evoked.times - 0.4))  # Peak at 400ms
        mock_evoked.data[pz_idx, time_idx_400ms] += 5e-6  # Add 5µV peak

        result = pipeline._detect_p300_peak(mock_evoked, "Pz")

        # Check that peak was detected
        assert "amplitude" in result
        assert "latency" in result
        assert not np.isnan(result["amplitude"])
        assert not np.isnan(result["latency"])

        # Check latency is within P300 window
        assert 300 <= result["latency"] <= 600

    def test_detect_p300_missing_electrode(self, temp_output_dir, mock_evoked):
        """Test P300 detection with missing electrode."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        result = pipeline._detect_p300_peak(mock_evoked, "MISSING")

        # Should return NaN for missing electrode
        assert np.isnan(result["amplitude"])
        assert np.isnan(result["latency"])

    def test_quantify_p300(self, temp_output_dir, mock_evoked):
        """Test P300 quantification feature schema."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = pipeline._quantify_p300(mock_evoked, "TEST001", "2024-01-01", n_epochs=3)

        # Required metadata fields
        assert features["patient_id"] == "TEST001"
        assert features["date"] == "2024-01-01"
        assert features["n_epochs"] == 3
        assert "processing_timestamp" in features

        # Baseline field
        assert "baseline_std_uV" in features

        # Per-electrode fields
        assert "p300_amplitude_Pz_uV" in features
        assert "p300_latency_Pz_ms" in features
        assert "p300_amplitude_Cz_uV" in features
        assert "p300_latency_Cz_ms" in features
        assert "p300_amplitude_Fz_uV" in features
        assert "p300_latency_Fz_ms" in features

        # Composite fields
        assert "p300_composite_amplitude_uV" in features
        assert "p300_composite_latency_ms" in features
        assert "p300_best_electrode" in features
        assert "p300_n_valid_electrodes" in features
        assert "p300_n_flagged_electrodes" in features

        # QC notes field
        assert "qc_notes" in features
        assert isinstance(features["qc_notes"], str)

        # Compatibility fields
        assert "p300_amplitude_uV" in features
        assert "p300_latency_ms" in features

        # Removed fields stay absent
        assert "n_rejected" not in features
        assert "timezone_offset_seconds" not in features
        assert "Pz_is_valid" not in features
        assert "p300_composite_amplitude_std_uV" not in features
        assert "p300_valid_electrodes" not in features

    def test_timezone_offset_consistent_with_eng02_logic(self, temp_output_dir, mock_raw_eeg):
        """Timezone detection follows ENG-02 half-hour rounding."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)
        edf_start_unix = mock_raw_eeg.info["meas_date"].timestamp()
        # 7 hours + 10 minutes -> floor to 7 hours with ENG-02 logic.
        rare_events = [
            {
                "timestamp_unix": edf_start_unix + (7 * 3600) + 600,
                "date": "2024-01-01",
                "trial_idx": 0,
            }
        ]
        offset = pipeline._detect_timezone_offset(mock_raw_eeg, rare_events)
        assert offset == -7 * 3600

    def test_epoch_window_boundary_filtering(self, temp_output_dir, mock_raw_eeg):
        """Events too close to epoch boundaries should be rejected pre-MNE."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)
        edf_start_unix = mock_raw_eeg.info["meas_date"].timestamp()
        # 60s recording, tmin=-0.2, tmax=0.7 -> valid center must be [0.2, 59.3].
        rare_events = [
            {
                "timestamp_unix": edf_start_unix + 0.1,
                "date": "2024-01-01",
                "trial_idx": 0,
            },  # too early
            {
                "timestamp_unix": edf_start_unix + 59.8,
                "date": "2024-01-01",
                "trial_idx": 0,
            },  # too late
            {
                "timestamp_unix": edf_start_unix + 10.0,
                "date": "2024-01-01",
                "trial_idx": 0,
            },  # valid
        ]
        epochs = pipeline._create_epochs(mock_raw_eeg, rare_events)
        assert len(epochs) == 1
        assert pipeline._last_epoch_diagnostics["n_too_close_to_start"] == 1
        assert pipeline._last_epoch_diagnostics["n_too_close_to_end"] == 1
        assert pipeline._last_epoch_diagnostics["n_valid_events_pre_mne"] == 1

    @patch("src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR")
    def test_load_aligned_trials(self, mock_aligned_dir, temp_output_dir, mock_aligned_events):
        """Test loading aligned trials."""
        # Create temporary aligned events file
        with tempfile.TemporaryDirectory() as tmpdir:
            aligned_dir = Path(tmpdir)
            mock_aligned_dir.return_value = aligned_dir

            aligned_file = aligned_dir / "TEST001_events.parquet"
            mock_aligned_events.to_parquet(aligned_file)

            # Update mock to return the path
            mock_aligned_dir.__truediv__ = lambda self, x: aligned_dir / x

            pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

            # Mock the config path
            with patch(
                "src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR",
                aligned_dir,
            ):
                trials = pipeline._load_aligned_trials("TEST001")

            # Should load oddball trials
            assert len(trials) == 1
            assert trials.iloc[0]["trial_type"] == "oddball"

    def test_save_outputs(self, temp_output_dir, mock_epochs, mock_evoked):
        """Test saving epochs and ERPs."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        features = {
            "patient_id": "TEST001",
            "date": "2024-01-01",
            "p300_amplitude_uV": 4.5,
        }

        pipeline._save_outputs("TEST001", "2024-01-01", mock_epochs, mock_evoked, features)

        # Check that files were created
        epochs_file = temp_output_dir / "epochs" / "TEST001_2024-01-01_oddball-epo.fif"
        erp_file = temp_output_dir / "erps" / "TEST001_2024-01-01_oddball-ave.fif"
        session_features_file = temp_output_dir / "features" / "TEST001_2024-01-01_p300_features.parquet"
        master_features_file = temp_output_dir / "features" / "p300_features.parquet"

        assert epochs_file.exists()
        assert erp_file.exists()
        assert session_features_file.exists()
        assert master_features_file.exists()

        # Verify files can be loaded
        loaded_epochs = mne.read_epochs(epochs_file, verbose=False)
        loaded_erp = mne.read_evokeds(erp_file, verbose=False)[0]

        assert len(loaded_epochs) == len(mock_epochs)
        assert loaded_erp.data.shape == mock_evoked.data.shape

    def test_plot_individual_erp(self, temp_output_dir, mock_evoked):
        """Test individual ERP plotting."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        pipeline._plot_individual_erp(mock_evoked, "TEST001", "2024-01-01")

        # Check that plot was saved
        plot_file = temp_output_dir / "plots" / "erp" / "TEST001_2024-01-01_oddball_erp.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0  # File is not empty

    def test_compute_grand_average(self, temp_output_dir, mock_evoked):
        """Test grand average computation."""
        # Save multiple ERPs
        erps_dir = temp_output_dir / "erps"
        erps_dir.mkdir(parents=True, exist_ok=True)

        mock_evoked.save(erps_dir / "TEST001_2024-01-01_oddball-ave.fif", overwrite=True)
        mock_evoked.save(erps_dir / "TEST002_2024-01-02_oddball-ave.fif", overwrite=True)
        mock_evoked.save(erps_dir / "grand_average_oddball-ave.fif", overwrite=True)

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        grand_avg = pipeline.compute_grand_average()

        # Check grand average was computed
        assert isinstance(grand_avg, mne.Evoked)

        # Check grand average file was saved
        grand_avg_file = temp_output_dir / "erps" / "grand_average_oddball-ave.fif"
        assert grand_avg_file.exists()

        # Check grand average plot was created
        grand_avg_plot = temp_output_dir / "plots" / "erp" / "grand_average_oddball_erp.png"
        assert grand_avg_plot.exists()

        # Ensure aggregate file was excluded from input set (should average only two session ERPs).
        assert grand_avg.nave == 2

    def test_update_master_feature_table_deduplicates_session(self, temp_output_dir):
        """Master table should keep latest row for the same patient/date key."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        df1 = pd.DataFrame(
            [
                {
                    "patient_id": "TEST001",
                    "date": "2024-01-01",
                    "n_epochs": 3,
                    "p300_amplitude_uV": 4.0,
                }
            ]
        )
        df2 = pd.DataFrame(
            [
                {
                    "patient_id": "TEST001",
                    "date": "2024-01-01",
                    "n_epochs": 5,
                    "p300_amplitude_uV": 6.0,
                }
            ]
        )

        pipeline._update_master_feature_table(df1)
        master = pipeline._update_master_feature_table(df2)

        assert len(master) == 1
        assert int(master.iloc[0]["n_epochs"]) == 5
        assert float(master.iloc[0]["p300_amplitude_uV"]) == 6.0

    def test_generate_qc_report(self, temp_output_dir):
        """Test QC report generation."""
        # Create mock feature table
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

        # Check report structure
        assert report["total_patients"] == 2
        assert report["total_sessions"] == 2
        assert report["total_epochs"] == 8
        assert "p300_detection_rate" in report
        assert "mean_amplitude_uV" in report
        assert "mean_latency_ms" in report
        assert "by_patient" in report

        # Check report file was saved
        report_file = temp_output_dir / "qc" / "erp_qc_report.json"
        assert report_file.exists()

    def test_get_patients_with_oddball(self, temp_output_dir):
        """Test getting list of patients with oddball data."""
        # Create mock aligned events files
        aligned_dir = temp_output_dir / "aligned_events"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        # Patient with oddball
        df_with_oddball = pd.DataFrame([{"trial_type": "oddball"}])
        df_with_oddball.to_parquet(aligned_dir / "TEST001_events.parquet")

        # Patient without oddball
        df_without_oddball = pd.DataFrame([{"trial_type": "language"}])
        df_without_oddball.to_parquet(aligned_dir / "TEST002_events.parquet")

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        with patch("src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR", aligned_dir):
            patient_ids = pipeline._get_patients_with_oddball()

        # Should only return TEST001
        assert len(patient_ids) == 1
        assert "TEST001" in patient_ids
        assert "TEST002" not in patient_ids

    def test_process_all_patients_empty(self, temp_output_dir):
        """Test batch processing with no patients."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Mock to return no patients
        with patch.object(pipeline, "_get_patients_with_oddball", return_value=[]):
            result = pipeline.process_all_patients()

        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestERPConfig:
    """Test ERP configuration constants."""

    def test_config_values(self):
        """Test that config values are reasonable."""
        assert ERP_CONFIG["tmin"] < 0  # Should start before stimulus
        assert ERP_CONFIG["tmax"] > 0  # Should end after stimulus
        assert ERP_CONFIG["baseline"][1] == 0  # Baseline should end at stimulus
        assert ERP_CONFIG["p300_window"][0] >= 0.3  # P300 starts around 300ms
        assert ERP_CONFIG["p300_window"][1] <= 0.7  # P300 ends before 700ms
        assert ERP_CONFIG["min_epochs"] >= 2  # Need at least 2 epochs for averaging


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.integration
    def test_full_pipeline_single_patient(self, temp_output_dir, mock_aligned_events, mock_raw_eeg):
        """Test full pipeline on single patient (requires all components)."""
        # Setup: Save mock aligned events
        aligned_dir = temp_output_dir / "aligned_events"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        mock_aligned_events.to_parquet(aligned_dir / "TEST001_events.parquet")

        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Mock the loader to return mock_raw_eeg
        with patch.object(pipeline.loader, "load_edf", return_value=mock_raw_eeg):
            with patch(
                "src.data_processing.erp_pipeline.config.ALIGNED_EVENTS_DIR",
                aligned_dir,
            ):
                result = pipeline.process_patient("TEST001")

        # Check result
        assert result["status"] == "success"
        assert "features" in result

        # Check outputs were created
        assert (temp_output_dir / "epochs").exists()
        assert (temp_output_dir / "erps").exists()
        assert (temp_output_dir / "plots" / "erp").exists()

    def test_validate_p300_electrode_positive(self, temp_output_dir):
        """Test validation of expected P300."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Expected P300
        validation = pipeline._validate_p300_electrode("Pz", 6.5, 420, "TEST001")

        assert validation["is_valid"] is True
        assert validation["is_positive"] is True
        assert validation["is_on_time"] is True
        assert validation["is_expected_latency"] is True
        assert len(validation["issues"]) == 0

    def test_validate_p300_electrode_inverted(self, temp_output_dir):
        """Test detection of negative-polarity P300."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Negative-polarity example
        validation = pipeline._validate_p300_electrode("Pz", -7.14, 537, "CON009")

        assert validation["is_valid"] is False
        assert validation["is_positive"] is False
        assert "negative_or_zero_amplitude" in validation["issues"]

    def test_validate_p300_electrode_late_latency(self, temp_output_dir):
        """Test detection of abnormal latency."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Too late latency
        validation = pipeline._validate_p300_electrode("Pz", 5.0, 650, "TEST001")

        assert validation["is_valid"] is False
        assert validation["is_on_time"] is False
        assert "latency_out_of_range" in validation["issues"]

    def test_validate_p300_electrode_atypical_latency(self, temp_output_dir):
        """Test detection of atypical but acceptable latency."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Atypical but within acceptable range
        validation = pipeline._validate_p300_electrode("Pz", 5.0, 280, "TEST001")

        assert validation["is_valid"] is True  # Still valid
        assert validation["is_expected_latency"] is False  # But not in typical range
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

        # If any valid electrodes exist
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

        # Core composite fields
        assert "p300_composite_amplitude_uV" in features
        assert "p300_composite_latency_ms" in features
        assert "p300_n_valid_electrodes" in features
        assert "p300_best_electrode" in features
        assert "p300_n_flagged_electrodes" in features

        # QC notes field
        assert "qc_notes" in features
        assert isinstance(features["qc_notes"], str)

        # Fields intentionally omitted
        assert "p300_valid_electrodes" not in features
        assert "p300_flagged_electrodes" not in features
        assert "p300_composite_amplitude_std_uV" not in features
        assert "Pz_is_valid" not in features
        assert "Pz_is_positive" not in features
        assert "Pz_issues" not in features

        # Individual electrode measurements remain
        assert "p300_amplitude_Pz_uV" in features
        assert "p300_amplitude_Cz_uV" in features
        assert "p300_amplitude_Fz_uV" in features
        assert "p300_latency_Pz_ms" in features
        assert "p300_latency_Cz_ms" in features
        assert "p300_latency_Fz_ms" in features

        # Compatibility fields
        assert "p300_amplitude_uV" in features
        assert "p300_latency_ms" in features

    def test_quantify_p300_custom_electrodes(self, temp_output_dir, mock_evoked):
        """Test custom electrode analysis mode."""
        pipeline = OddballERPPipeline(output_dir=temp_output_dir, verbose=False)

        # Analyze custom electrodes
        features = pipeline._quantify_p300(
            mock_evoked,
            "TEST001",
            "2024-01-01",
            n_epochs=3,
            custom_electrodes=["Fz", "Cz"],
        )

        # Metadata fields
        assert features["patient_id"] == "TEST001"
        assert features["date"] == "2024-01-01"

        # Custom electrode fields
        assert "p300_amplitude_Fz_uV" in features
        assert "p300_latency_Fz_ms" in features
        assert "p300_amplitude_Cz_uV" in features
        assert "p300_latency_Cz_ms" in features

        # Default electrode not included
        assert "p300_amplitude_Pz_uV" not in features

        # Composite fields are not emitted in custom mode
        assert "p300_composite_amplitude_uV" not in features
        assert "p300_n_valid_electrodes" not in features

        # QC notes indicate custom mode
        assert "qc_notes" in features
        assert "Custom electrode analysis" in features["qc_notes"]
