import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

from src.data_processing.timestamp_aligner import TimestampAligner


@pytest.fixture
def aligner(mock_loader):
    """TimestampAligner instance with mocked loader."""
    return TimestampAligner(patient_id="P001", verbose=False)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_detect_timezone_offset_no_offset(aligner, mock_raw):
    """Test 0 offset case (same timezone)."""
    trials_df = pd.DataFrame(
        {"start_time": [mock_raw.info["meas_date"].timestamp() + 1.0]}
    )
    offset = aligner._detect_timezone_offset(mock_raw, trials_df)
    assert offset == 0.0


def test_detect_timezone_offset_positive(aligner, mock_raw):
    """Test offset when trial is hours ahead (e.g. EDF=UTC, Trial=CET)."""
    # Logic: if trial > edf, offset is negative to bring trial back to edf time
    base_ts = mock_raw.info["meas_date"].timestamp()
    trials_df = pd.DataFrame({"start_time": [base_ts + 7201.0]})  # 2h + 1s difference
    offset = aligner._detect_timezone_offset(mock_raw, trials_df)
    assert offset == -7200.0


def test_detect_timezone_offset_negative(aligner, mock_raw):
    """Test offset when trial is hours behind."""
    base_ts = mock_raw.info["meas_date"].timestamp()
    trials_df = pd.DataFrame({"start_time": [base_ts - 7200.0]})  # exact -2h
    offset = aligner._detect_timezone_offset(mock_raw, trials_df)
    assert offset == 7200.0


@patch("src.data_processing.timestamp_aligner.utils")
def test_align_correlation(mock_utils, aligner, mock_raw, sample_trials_df):
    """Test correlation alignment logic."""
    # Setup aligner context manually
    aligner.patient_id = "P001"
    aligner.raw = mock_raw
    aligner.sr = 1000.0
    aligner.dc_channel = "DC1"
    aligner.dc_signal = mock_raw.get_data()[0]
    aligner.edf_start_unix = mock_raw.info["meas_date"].timestamp()
    aligner.timezone_offset = 0.0

    # Mock loader returning a path
    aligner.loader.get_stimulus_audio_path.return_value = Path("fake_audio.wav")
    aligner.loader.load_stimulus_audio.return_value = (
        44100,
        np.zeros(44100),
    )  # 1s audio

    # Mock file existence check
    with patch("pathlib.Path.exists", return_value=True):
        # Mock signal processing
        mock_utils.resample_signal.return_value = np.zeros(1000)
        mock_utils.audio_envelope.return_value = np.zeros(1000)

        # Lag=22050 samples (0.5s at 44100Hz)
        def mock_cross_corr(sig1, sig2):
            return 22050, 0.95

        mock_utils.cross_correlate.side_effect = mock_cross_corr

        trial = sample_trials_df.iloc[0]
        result_df = aligner._align_correlation(trial)

    assert not result_df.empty
    aligned_sentence = result_df.iloc[0]["sentences"][0]

    expected_start = aligner.edf_start_unix + 0.5
    assert aligned_sentence["event_start"] == expected_start
    assert aligned_sentence["correlation_score"] == 0.95


@patch("src.data_processing.timestamp_aligner.utils")
def test_align_peaks(mock_utils, aligner, mock_raw, sample_trials_df):
    """Test peak detection alignment."""
    # Setup aligner context
    aligner.patient_id = "P001"
    aligner.raw = mock_raw
    aligner.sr = 1000.0
    aligner.dc_channel = "DC1"
    aligner.dc_signal = mock_raw.get_data()[0]
    aligner.edf_start_unix = mock_raw.info["meas_date"].timestamp()
    aligner.timezone_offset = 0.0

    # Mock peaks: return index 500 (relative to chunk)
    mock_utils.detect_peaks.return_value = (np.array([500]), {"widths": np.array([50])})

    trial = sample_trials_df.iloc[0]
    trial["trial_type"] = "oddball"

    result_df = aligner._align_peaks(trial)

    assert not result_df.empty
    aligned_sentence = result_df.iloc[0]["sentences"][0]

    expected_unix = aligner.edf_start_unix + 1.5
    assert aligned_sentence["event_start"] == expected_unix
    assert "peak_amplitude" in aligned_sentence


def test_align_end_to_end(aligner, mock_raw, sample_trials_df):
    """Test full pipeline flow for a patient session."""
    # Mock patient and session data
    patient = MagicMock()
    patient.trials_df = sample_trials_df
    patient.list_sessions.return_value = ["2024-01-01"]
    patient.get_raw.return_value = mock_raw

    aligner.loader.get_patient.return_value = patient

    with patch.object(aligner, "_align_correlation") as mock_align_corr:
        mock_align_corr.return_value = pd.DataFrame(
            [{"patient_id": "P001", "trial_type": "language"}]
        )

        with patch.object(aligner, "_detect_dc_channel", return_value="DC1"):
            with patch(
                "src.data_processing.timestamp_aligner.utils.select_best_dc_channel",
                return_value="DC1",
            ):
                results = aligner.align(save=False)

    assert "P001" in results
    assert len(results["P001"]) == 1

    patient.get_raw.assert_called_with("2024-01-01")


@patch("pandas.read_parquet")
def test_validate(mock_read_parquet):
    """Test validation reporting."""
    # Create sample validation df
    df = pd.DataFrame(
        {
            "patient_id": ["P001", "P001"],
            "trial_type": ["language", "oddball"],
            "sentences": [
                [{"event": "1", "event_start": 100.0, "correlation_score": 0.9}],
                [{"event": "2", "event_start": None}],
            ],
        }
    )
    mock_read_parquet.return_value = df

    # Mock Path.exists to pass the file check
    with patch("pathlib.Path.exists", return_value=True):
        report = TimestampAligner.validate("P001")

    # Check for success
    assert (
        report.get("status") != "error"
    ), f"Validation failed with: {report.get('message')}"
    assert report["patient_id"] == "P001"
    assert report["trials"] == 2
