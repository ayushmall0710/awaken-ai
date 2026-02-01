import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.data_processing.timestamp_aligner import TimestampAligner, AudioMatch


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


def test_align_sentence_trials(aligner, mock_raw, sample_trials_df):
    """Test alignment logic for sentence trials."""
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

    # Mock _compute_audio_match to return a valid AudioMatch dataclass
    # We patch the METHOD on the instance or class
    with patch.object(aligner, "_compute_audio_match") as mock_match:
        # Return match at offset 0.5s, duration 1.0s, score 0.95
        mock_match.return_value = AudioMatch(
            offset_seconds=0.5, duration_seconds=1.0, score=0.95
        )

        trial = sample_trials_df.iloc[0]
        # Ensure trial has sentences
        result_df = aligner._align_sentence_trials(trial)

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

    # Mock Instruction detection to return 0 (no instruction found/masked)
    # We can patch _detect_instruction_end directly
    with patch.object(aligner, "_detect_instruction_end", return_value=0):
        # Mock peaks: return index 500 (relative to chunk)
        mock_utils.detect_peaks.return_value = (
            np.array([500]),
            {"widths": np.array([50])},
        )
        mock_utils.highpass_filter.return_value = np.zeros(1000)
        mock_utils.audio_envelope.return_value = np.zeros(1000)

        trial = sample_trials_df.iloc[0]
        trial["trial_type"] = "oddball"

        result_df = aligner._align_peaks(trial)

    assert not result_df.empty
    aligned_sentence = result_df.iloc[0]["sentences"][0]

    # Index 500 @ 1000Hz = 0.5s into the chunk
    # Trial started at t=1.0. Chunk starts at 1.0.
    # So peak is at 1.0 + 0.5 = 1.5
    expected_unix = aligner.edf_start_unix + 1.5
    assert aligned_sentence["event_start"] == expected_unix
    assert "peak_amplitude" in aligned_sentence


def test_align_end_to_end(aligner, mock_raw, sample_trials_df):
    """Test full pipeline flow for a patient session."""
    # Mock patient and session data
    patient = MagicMock()
    patient.trials_df = sample_trials_df
    # Ensure trial types map to method
    sample_trials_df["trial_type"] = "language"

    patient.list_sessions.return_value = ["2024-01-01"]
    patient.get_raw.return_value = mock_raw

    aligner.loader.get_patient.return_value = patient
    aligner.loader.load_edf.return_value = mock_raw

    # Patch the specific alignment method used for 'language' trials
    with patch.object(aligner, "_align_sentence_trials") as mock_align_lang:
        mock_align_lang.return_value = pd.DataFrame(
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

    aligner.loader.load_edf.assert_called_with(
        "P001", date="2024-01-01", use_clipped=True
    )


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
