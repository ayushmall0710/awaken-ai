"""
Signal Processing Utilities

Helper functions for signal processing: peak detection, resampling, normalization, cross-correlation.
"""

from typing import Tuple, List, Any
import logging

import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import uniform_filter1d
import mne

logger = logging.getLogger(__name__)


def select_best_dc_channel(
    raw: mne.io.Raw, keywords: List[str] = ["DC", "AUX", "AUDIO", "DIG", "STIM"]
) -> str:
    """
    Select the best DC channel based on keywords and standard deviation.

    Args:
        raw: MNE Raw object
        keywords: List of keywords to identify candidate channels

    Returns:
        Best DC channel name

    Raises:
        ValueError: If no candidate channels are found
    """
    candidates = []
    for ch in raw.ch_names:
        if any(kw in ch.upper() for kw in keywords):
            candidates.append(ch)

    if not candidates:
        raise ValueError(f"No DC channel found. Available: {raw.ch_names}")

    if len(candidates) == 1:
        logger.info(f"Detected single DC channel: {candidates[0]}")
        return candidates[0]

    # Multiple candidates: choose one with highest standard deviation (most signal activity)
    logger.info(f"Multiple DC candidates found: {candidates}. Selecting by max std.")

    # Load entire signal for all candidates (vectorized)
    data = raw.get_data(picks=candidates)

    # Compute std for all channels
    std_vals = np.std(data, axis=1)

    # Find channel with maximum std
    best_idx = np.argmax(std_vals)
    best_ch = candidates[best_idx]
    max_std = std_vals[best_idx]

    logger.info(f"Selected best DC channel: {best_ch} (std: {max_std:.2f})")
    return best_ch


def detect_peaks(
    signal_data: np.ndarray,
    sfreq: float,
    prominence: float = 1.5,
    min_distance_sec: float = 0.8,
    normalize: bool = False,
) -> Tuple[np.ndarray, Any]:
    """
    Detect peaks in signal using scipy.signal.find_peaks.

    Args:
        signal_data: 1D signal array
        sfreq: Sampling frequency in Hz
        prominence: Required prominence of peaks
        min_distance_sec: Minimum distance between peaks in seconds
        normalize: If True, z-score normalize signal before peak detection

    Returns:
        Tuple of (peaks indices, properties dict)
    """
    min_dist_samples = int(min_distance_sec * sfreq)

    # Normalize if requested
    data = normalize_signal(signal_data) if normalize else signal_data

    peaks, properties = find_peaks(
        data,
        prominence=prominence,
        distance=min_dist_samples,
        width=1,  # Request peak widths
    )
    return peaks, properties


def resample_signal(signal_data: np.ndarray, src_hz: int, target_hz: int) -> np.ndarray:
    """
    Resample signal from source frequency to target frequency.

    Args:
        signal_data: Input signal array
        src_hz: Source sampling frequency in Hz
        target_hz: Target sampling frequency in Hz

    Returns:
        Resampled signal array
    """
    if src_hz == target_hz:
        return signal_data
    return signal.resample_poly(signal_data, target_hz, src_hz)


def normalize_signal(signal_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize signal (zero mean, unit variance).

    Args:
        signal_data: Input signal array

    Returns:
        Normalized signal array
    """
    std = np.std(signal_data)
    if std < 1e-9:
        return signal_data - np.mean(signal_data)
    return (signal_data - np.mean(signal_data)) / std


def cross_correlate(recording: np.ndarray, template: np.ndarray) -> Tuple[int, float]:
    """
    Find best match of template in recording using cross-correlation.

    Args:
        recording: Recording signal array to search in
        template: Template signal array to search for

    Returns:
        Tuple of (lag in samples, correlation score [0.0-1.0])
    """
    rec_norm = normalize_signal(recording)
    tmpl_norm = normalize_signal(template)

    corr = signal.correlate(rec_norm, tmpl_norm, mode="valid", method="fft")

    if len(corr) == 0:
        return 0, 0.0

    lag = int(np.argmax(np.abs(corr)))
    score = float(np.max(np.abs(corr)) / len(tmpl_norm))

    return lag, min(max(score, 0.0), 1.0)


def audio_envelope(
    audio: np.ndarray, sample_rate: float = None, smooth_ms: float = 20
) -> np.ndarray:
    """
    Compute audio amplitude envelope using Hilbert transform.

    The envelope representation often correlates better with DC channel signals
    than raw audio waveforms, as DC channels may record amplitude-modulated signals.

    Args:
        audio: Raw audio waveform array
        sample_rate: Sample rate in Hz (required if smooth_ms > 0)
        smooth_ms: Smoothing window in milliseconds (default 20ms)

    Returns:
        Amplitude envelope of the audio
    """
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)

    if smooth_ms > 0 and sample_rate is not None:
        window_samples = max(1, int(sample_rate * smooth_ms / 1000))
        envelope = uniform_filter1d(envelope, size=window_samples)

    return envelope
