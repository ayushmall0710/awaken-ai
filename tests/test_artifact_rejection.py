"""Tests for ENG-03: Artifact Rejection (ICA)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")  # ENG-03 depends on MNE; skip tests if not installed

from src.data_processing.artifact_rejection import (  # noqa: E402
    WINDOW_SEC_BY_TRIAL_TYPE,
    ArtifactRejector,
    _find_eog_channels,
    _pick_eeg_indices,
    _trial_type_window_sec,
)
from src.utils.signal_processing import exclude_non_eeg_channels  # noqa: E402
from src.utils.time_utils import detect_timezone_offset, unix_to_edf  # noqa: E402

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


def test_unix_to_edf_matches_eng02_formula():
    """unix_to_edf should use the exact same formula as TimestampAligner._unix_to_edf."""
    assert unix_to_edf(1010.0, edf_start_unix=1000.0, timezone_offset=0.0) == 10.0
    assert unix_to_edf(1010.0, edf_start_unix=1000.0, timezone_offset=3600.0) == 3610.0


def test_detect_timezone_offset_zero_when_close():
    raw = _mock_raw(meas_ts=1000.0)
    df = pd.DataFrame({"start_time": [1001.0]})
    assert detect_timezone_offset(raw, df) == 0.0


def test_detect_timezone_offset_rounds_to_30min_steps():
    raw = _mock_raw(meas_ts=1000.0)
    # 2 hours + 1 second ahead => correction should be −7200
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
    # All non-EEG channels should be excluded
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
    # Real EEG channels must NOT be excluded
    for ch in ["C3", "C4", "Fp1", "Fp2"]:
        assert ch not in excluded, f"{ch} should NOT be excluded"


def test_exclude_non_eeg_channels_empty_when_no_aux():
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "Fp1", "Fp2"])
    assert exclude_non_eeg_channels(raw) == []


def test_find_eog_channels_fallback_to_fp():
    """When no typed EOG or EOG-named channels exist, should fall back to Fp1/Fp2."""
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "Fp1", "Fp2"])
    # MNE pick_types on a mock won't return real EOG; the function should fall through.
    eog = _find_eog_channels(raw)
    assert set(eog) == {"Fp1", "Fp2"}


def test_find_eog_channels_prefers_io_over_fp():
    """IO1/IO2 (infraorbital) should be preferred over Fp1/Fp2 as EOG reference."""
    raw = _mock_raw(ch_names=["C3", "C4", "Fp1", "Fp2", "IO1", "IO2"])
    eog = _find_eog_channels(raw)
    assert set(eog) == {"IO1", "IO2"}


def test_pick_eeg_indices_excludes_dc():
    """_pick_eeg_indices should exclude DC/AUX channels from epoch picks."""
    raw = _mock_raw(ch_names=["EEG1", "EEG2", "DC1", "Fp1"])
    picks = _pick_eeg_indices(raw)
    picked_names = [raw.ch_names[i] for i in picks]
    assert "DC1" not in picked_names
    assert "EEG1" in picked_names
    assert "Fp1" in picked_names


def test_pick_eeg_indices_excludes_all_non_eeg():
    """_pick_eeg_indices must exclude EMG, ECG, respiratory, etc. even when all
    channels are typed as 'eeg' (the common EDF case)."""
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
    # Only real EEG channels should survive
    assert set(picked_names) == {"C3", "C4", "O1", "O2", "Fp1", "Fp2"}


# ── QC PTP stats ─────────────────────────────────────────────────────────────


def test_qc_ptp_stats_empty():
    assert ArtifactRejector._ptp_stats(np.array([])) == {}


def test_qc_ptp_stats_non_empty():
    stats = ArtifactRejector._ptp_stats(np.array([1.0, 2.0, 100.0]))
    assert "ptp_uv_p50" in stats
    assert "ptp_uv_p95" in stats
    assert "ptp_uv_p99" in stats
    assert stats["ptp_uv_max"] == 100.0
