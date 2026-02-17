"""
Shared time-conversion utilities for EDF ↔ Unix timestamp alignment.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def detect_timezone_offset(raw: Any, stimulus_df: pd.DataFrame) -> float:
    """
    Detect timezone offset between EDF recording start and stimulus log timestamps.

    Compares EDF meas_date (converted to Unix) with the earliest trial start_time.
    If the difference exceeds 30 minutes, rounds to the nearest 30-minute block
    and returns a correction in seconds.

    Args:
        raw: MNE Raw-like object (must have raw.info["meas_date"] with .timestamp()).
        stimulus_df: DataFrame with a ``start_time`` column (Unix timestamps).

    Returns:
        Timezone correction in seconds.  Positive means the stimulus clock is behind
        the EDF clock; negative means it is ahead.
    """
    meas_date = raw.info.get("meas_date")
    if meas_date is None or stimulus_df.empty:
        return 0.0

    edf_start_unix = meas_date.timestamp()
    first_trial_start = float(stimulus_df["start_time"].min())
    diff = abs(first_trial_start - edf_start_unix)

    if diff > 1800:  # 30 mins (handles fractional timezones like IST)
        correction = (diff // 1800) * 1800
        if first_trial_start > edf_start_unix:
            correction = -correction
        logger.info("Timezone offset detected: %.1f hours", correction / 3600)
        return float(correction)

    return 0.0


def unix_to_edf(unix_time: float, *, edf_start_unix: float, timezone_offset: float) -> float:
    """
    Convert a Unix timestamp to EDF-relative seconds.

    Formula (consistent with ENG-02 TimestampAligner._unix_to_edf):
        edf_time = (unix_time − edf_start_unix) + timezone_offset

    Args:
        unix_time: The Unix timestamp to convert.
        edf_start_unix: The EDF recording start as a Unix timestamp
                        (``raw.info["meas_date"].timestamp()``).
        timezone_offset: Correction returned by :func:`detect_timezone_offset`.

    Returns:
        Time in seconds relative to the start of the EDF recording.
    """
    return (unix_time - edf_start_unix) + timezone_offset


def edf_to_unix(edf_time: float, *, edf_start_unix: float, timezone_offset: float) -> float:
    """
    Convert EDF-relative seconds back to a Unix timestamp.

    Inverse of :func:`unix_to_edf`.
    """
    return edf_time - timezone_offset + edf_start_unix
