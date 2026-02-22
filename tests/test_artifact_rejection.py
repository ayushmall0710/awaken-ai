"""Tests for ENG-03: Artifact Rejection (ICA)."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")  # ENG-03 depends on MNE; skip tests if not installed

from src.data_processing.artifact_rejection import (  # noqa: E402
    WINDOW_SEC_BY_TRIAL_TYPE,
    ArtifactRejector,
    _apply_car_reference,
    _classify_components_correlation,
    _classify_components_iclabel,
    _find_eog_channels,
    _note,
    _pick_eeg_indices,
    _trial_type_window_sec,
    _try_set_montage,
)
from src.utils.signal_processing import exclude_non_eeg_channels, normalize_channel_names  # noqa: E402
from src.utils.time_utils import detect_timezone_offset  # noqa: E402

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_raw(meas_ts: float = 1000.0, sfreq: float = 1000.0, duration_sec: float = 100.0, ch_names=None):
    """Lightweight mock for mne.io.Raw used in unit tests."""
    raw = MagicMock()
    raw.info = {
        "sfreq": sfreq,
        "meas_date": MagicMock(timestamp=lambda: meas_ts),
        "ch_names": ch_names or ["EEG1", "EEG2", "Fp1"],
    }
    raw.ch_names = raw.info["ch_names"]
    raw.times = np.linspace(0, duration_sec, int(duration_sec * sfreq) + 1)
    return raw


# ── Trial-type window mapping ────────────────────────────────────────────────


def test_trial_type_window_mapping_known_types():
    assert _trial_type_window_sec("language", None) == WINDOW_SEC_BY_TRIAL_TYPE["language"]
    assert _trial_type_window_sec("oddball", None) == WINDOW_SEC_BY_TRIAL_TYPE["oddball"]
    assert _trial_type_window_sec("left_command", None) == WINDOW_SEC_BY_TRIAL_TYPE["left_command"]


def test_trial_type_window_fallback_duration():
    assert _trial_type_window_sec("unknown_type", 10.5) == 10.5
    assert _trial_type_window_sec("unknown_type", 0.2) is None  # too short
    assert _trial_type_window_sec("unknown_type", 1000.0) is None  # too long


# ── Shared time-utils (from src.utils.time_utils) ───────────────────────────


def test_detect_timezone_offset_zero_when_close():
    raw = _mock_raw(meas_ts=1000.0)
    df = pd.DataFrame({"start_time": [1001.0]})
    assert detect_timezone_offset(raw, df) == 0.0


def test_detect_timezone_offset_rounds_to_30min_steps():
    raw = _mock_raw(meas_ts=1000.0)
    # 2 hours + 1 second ahead => correction should be -7200
    df = pd.DataFrame({"start_time": [1000.0 + 7201.0]})
    assert detect_timezone_offset(raw, df) == -7200.0


# ── Channel-selection helpers (from signal_processing + artifact_rejection) ──


def test_exclude_non_eeg_channels_drops_dc():
    """DC/AUX/STIM channels should be identified for exclusion."""
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "DC1", "DC2", "AUX1", "Fp1"])
    excluded = exclude_non_eeg_channels(raw)
    assert "DC1" in excluded
    assert "DC2" in excluded
    assert "AUX1" in excluded
    assert "EEG1" not in excluded
    assert "Fp1" not in excluded


def test_exclude_non_eeg_channels_drops_polysomnography():
    """EMG, ECG, respiratory, and misc sensor channels should be excluded."""
    raw = _mock_raw(
        ch_names=[
            "C3",
            "C4",
            "Fp1",
            "Fp2",
            "EMG1",
            "EMG2",
            "ECGL",
            "ECGR",
            "RESP",
            "ABD",
            "FLOW",
            "SNORE",
            "DC2",
            "DC3",
            "OSAT",
            "PR",
            "POS",
            "IO1",
            "IO2",
            "LAT1",
            "RAT1",
            "DIF5",
        ]
    )
    excluded = exclude_non_eeg_channels(raw)
    for ch in [
        "EMG1",
        "EMG2",
        "ECGL",
        "ECGR",
        "RESP",
        "ABD",
        "FLOW",
        "SNORE",
        "DC2",
        "DC3",
        "OSAT",
        "PR",
        "POS",
        "IO1",
        "IO2",
        "LAT1",
        "RAT1",
        "DIF5",
    ]:
        assert ch in excluded, f"{ch} should be excluded"
    for ch in ["C3", "C4", "Fp1", "Fp2"]:
        assert ch not in excluded, f"{ch} should NOT be excluded"


def test_exclude_non_eeg_channels_empty_when_no_aux():
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "Fp1", "Fp2"])
    assert exclude_non_eeg_channels(raw) == []


def test_find_eog_channels_fallback_to_fp():
    """When no typed EOG or EOG-named channels exist, should fall back to Fp1/Fp2."""
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "Fp1", "Fp2"])
    eog = _find_eog_channels(raw)
    assert set(eog) == {"Fp1", "Fp2"}


def test_find_eog_channels_prefers_io_over_fp():
    """IO1/IO2 (infraorbital) should be preferred over Fp1/Fp2 as EOG reference."""
    raw = _mock_raw(ch_names=["C3", "C4", "Fp1", "Fp2", "IO1", "IO2"])
    eog = _find_eog_channels(raw)
    assert set(eog) == {"IO1", "IO2"}


def test_find_eog_channels_combines_eog_and_io():
    """When both EOG-named and IO channels exist, return the union of both."""
    raw = _mock_raw(ch_names=["C3", "C4", "EOG1", "Fp1", "IO1", "IO2"])
    eog = _find_eog_channels(raw)
    assert set(eog) == {"EOG1", "IO1", "IO2"}


def test_pick_eeg_indices_excludes_dc():
    """_pick_eeg_indices should exclude DC/AUX channels from epoch picks."""
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "DC1", "Fp1"])
    picks = _pick_eeg_indices(raw)
    picked_names = [raw.ch_names[i] for i in picks]
    assert "DC1" not in picked_names
    assert "EEG1" in picked_names
    assert "Fp1" in picked_names


def test_pick_eeg_indices_excludes_all_non_eeg():
    """_pick_eeg_indices must exclude EMG, ECG, respiratory, etc."""
    raw = _mock_raw(
        ch_names=[
            "C3",
            "C4",
            "O1",
            "O2",
            "Fp1",
            "Fp2",
            "EMG1",
            "ECGL",
            "RESP",
            "SNORE",
            "DC2",
            "OSAT",
            "PR",
        ]
    )
    picks = _pick_eeg_indices(raw)
    picked_names = [raw.ch_names[i] for i in picks]
    assert set(picked_names) == {"C3", "C4", "O1", "O2", "Fp1", "Fp2"}


# ── _note helper ─────────────────────────────────────────────────────────────


def test_note_appends_and_logs(caplog):
    """_note should both append to the list and emit a logger.debug call."""
    notes: list = []
    with caplog.at_level(logging.DEBUG, logger="src.data_processing.artifact_rejection"):
        _note(notes, "[TEST] hello")
    assert notes == ["[TEST] hello"]
    assert "[TEST] hello" in caplog.text


# ── Montage setup ────────────────────────────────────────────────────────────


def test_try_set_montage_with_standard_names():
    """Channels matching 10-20 names should result in a successful montage."""
    raw = _mock_raw(ch_names=["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz", "Pz"])
    notes: list = []
    _try_set_montage(raw, notes)
    # On a mock, set_montage may raise — but the function should handle it gracefully.
    # We verify the logic tries to set the montage (matched >= 5).
    matched_note = [n for n in notes if "[MONTAGE]" in n]
    assert len(matched_note) > 0


def test_try_set_montage_skips_non_standard_names():
    """With < 5 matching channels, montage should be skipped."""
    raw = _mock_raw(ch_names=["CH1", "CH2", "CH3", "CH4", "CH5", "CH6"])
    notes: list = []
    result = _try_set_montage(raw, notes)
    assert result is False
    assert any("skipped" in n for n in notes)


def test_try_set_montage_strips_eeg_prefix():
    """Channels with 'EEG ' prefix should be renamed and matched."""
    # This now relies on normalize_channel_names internally
    raw = _mock_raw(ch_names=["EEG Fp1", "EEG Fp2", "EEG C3", "EEG C4", "EEG O1", "EEG O2"])
    notes: list = []
    _try_set_montage(raw, notes)
    rename_notes = [n for n in notes if "renamed_channels" in n]
    assert len(rename_notes) > 0


def test_normalize_channel_names_utility():
    """Verify the utility function directly (since we imported it)."""
    raw_names = ["EEG Fp1", "EEG-Fp2", "C3-Ref", "O1", "ECG"]
    normalized = normalize_channel_names(raw_names)
    assert normalized == ["Fp1", "Fp2", "C3", "O1", "ECG"]


# ── CAR reference ────────────────────────────────────────────────────────────


def test_apply_car_reference_records_note():
    """_apply_car_reference should call set_eeg_reference and log a note."""
    raw = _mock_raw(ch_names=["C3", "C4", "O1", "O2", "Fz"])
    raw.set_eeg_reference = MagicMock()
    notes: list = []
    _apply_car_reference(raw, notes)
    raw.set_eeg_reference.assert_called_once()
    assert any("[REFERENCE]" in n and "CAR" in n for n in notes)


def test_apply_car_reference_handles_failure():
    """_apply_car_reference should log a note when CAR fails."""
    raw = _mock_raw(ch_names=["C3", "C4"])
    raw.set_eeg_reference = MagicMock(side_effect=RuntimeError("CAR error"))
    notes: list = []
    _apply_car_reference(raw, notes)
    assert any("CAR failed" in n for n in notes)


# ── ICLabel classification ───────────────────────────────────────────────────


def test_classify_iclabel_returns_dict_on_success():
    """When label_components works, should return a classification dict."""
    raw = _mock_raw(ch_names=["Fp1", "C3", "O1"])
    ica = MagicMock()
    ica.get_components.return_value = np.eye(3)

    mock_result = {
        "labels": ["brain", "eye", "line_noise"],
        "y_pred_proba": np.array(
            [
                [0.9, 0.0, 0.05, 0.0, 0.05, 0.0, 0.0],
                [0.1, 0.0, 0.8, 0.0, 0.1, 0.0, 0.0],
                [0.05, 0.0, 0.0, 0.0, 0.0, 0.9, 0.05],
            ]
        ),
    }

    notes: list = []
    with patch(
        "src.data_processing.artifact_rejection.label_components",
        return_value=mock_result,
    ):
        result = _classify_components_iclabel(raw, ica, notes, threshold=0.5)

    assert result is not None
    assert 1 in result["eog_components"]  # "eye" at index 1
    assert 2 in result["line_noise_components"]  # "line_noise" at index 2
    assert 0 not in result["excluded"]  # "brain" should not be excluded
    assert result["iclabel_labels"] == ["brain", "eye", "line_noise"]


def test_classify_iclabel_returns_none_on_runtime_error():
    """When label_components raises at runtime, should return None and log."""
    raw = _mock_raw()
    ica = MagicMock()
    notes: list = []

    with patch(
        "src.data_processing.artifact_rejection.label_components",
        side_effect=RuntimeError("ONNX backend error"),
    ):
        result = _classify_components_iclabel(raw, ica, notes)

    assert result is None
    assert any("classification failed" in n for n in notes)


# ── Correlation-based classification ─────────────────────────────────────────


def test_classify_correlation_returns_dict():
    """Correlation classifier should return a dict with expected keys."""
    raw = _mock_raw(ch_names=["C3", "C4", "Fp1", "Fp2"])
    ica = MagicMock()
    ica.find_bads_eog.return_value = ([], [])
    ica.find_bads_ecg.return_value = ([], [])
    notes: list = []

    result = _classify_components_correlation(raw, ica, notes)
    assert isinstance(result, dict)
    assert "excluded" in result
    assert "eog_components" in result
    assert "ecg_components" in result
    assert "muscle_components" in result
    assert result["iclabel_labels"] is None


# ── Fallback path ────────────────────────────────────────────────────────────


def test_apply_ica_falls_back_to_correlation_when_no_montage():
    """When montage cannot be set, _apply_ica should use correlation method."""
    # This is an integration-level test using mocks.
    raw = _mock_raw(ch_names=["CH1", "CH2", "CH3"])  # Non-standard names
    raw.copy.return_value = raw
    raw.filter = MagicMock()
    raw.info["dig"] = None

    ica_mock = MagicMock()
    ica_mock.n_components_ = 2
    ica_mock.get_components.return_value = np.eye(2)
    ica_mock.find_bads_eog.return_value = ([], [])
    ica_mock.find_bads_ecg.return_value = ([], [])
    ica_mock.apply.return_value = raw

    with patch("src.data_processing.artifact_rejection._fit_ica", return_value=ica_mock):
        ar = ArtifactRejector.__new__(ArtifactRejector)
        ar.ica_filter_hz = (1.0, 100.0)
        ar.iclabel_threshold = 0.5
        ar.verbose = False

        _raw_clean, summary = ar._apply_ica(raw)

    assert summary.classification_method == "correlation"
    assert summary.method == "infomax"


# ── Lazy __init__.py imports ─────────────────────────────────────────────────


def test_data_processing_lazy_import():
    """from src.data_processing import ArtifactRejector should work at runtime."""
    from src.data_processing import ArtifactRejector as AR

    assert AR is not None
    assert AR.__name__ == "ArtifactRejector"


def test_data_processing_lazy_import_bad_name():
    """Accessing a nonexistent name should raise AttributeError or ImportError."""
    with pytest.raises((AttributeError, ImportError)):
        from src.data_processing import NonExistentClass  # noqa: F401


def test_data_loading_lazy_import():
    """from src.data_loading import UnifiedDataLoader should work at runtime."""
    from src.data_loading import UnifiedDataLoader as UDL

    assert UDL is not None
    assert UDL.__name__ == "UnifiedDataLoader"


# ── QC PTP stats ─────────────────────────────────────────────────────────────


def test_qc_ptp_stats_empty():
    assert ArtifactRejector._ptp_stats(np.array([])) == {}


def test_qc_ptp_stats_non_empty():
    stats = ArtifactRejector._ptp_stats(np.array([1.0, 2.0, 100.0]))
    assert "ptp_uv_p50" in stats
    assert "ptp_uv_p95" in stats
    assert "ptp_uv_p99" in stats
    assert stats["ptp_uv_max"] == 100.0


def test_default_ica_filter_is_0_5hz():
    """Ensure the default high-pass filter is set to 0.5 Hz for sentence analysis."""
    from src.data_processing.artifact_rejection import DEFAULT_ICA_FILTER_HZ

    assert DEFAULT_ICA_FILTER_HZ == (0.5, 100.0)
