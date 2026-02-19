"""
Signal Processing Utilities

Helper functions for signal processing: peak detection, resampling, normalization, cross-correlation.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# Keywords that identify non-EEG auxiliary channels in EDF files.
# Used by select_best_dc_channel (to *find* them) and exclude_non_eeg_channels (to *drop* them).
# Covers: DC/stimulus, polysomnography (EMG/ECG/respiratory/SpO2), movement, and misc sensors.
NON_EEG_CHANNEL_KEYWORDS: List[str] = [
    # Original
    "DC",
    "AUX",
    "AUDIO",
    "DIG",
    "STIM",
    # Electrophysiological (non-brain)
    "EMG",
    "ECG",
    "EKG",
    # Infraorbital / EOG reference (eye movement, not scalp EEG)
    "IO1",
    "IO2",
    # Leg / limb movement
    "LAT1",
    "LAT2",
    "RAT1",
    "RAT2",
    # Respiratory
    "RESP",
    "ABD",
    "FLOW",
    "SNORE",
    "THORAX",
    # Differential / misc sensors
    "DIF",
    # Body position
    "POS",
    # Pulse oximetry
    "OSAT",
    "SAT",
    "SPO2",
    "PR",
    "PULSE",
]

# DC-only subset used by select_best_dc_channel.
DC_CHANNEL_KEYWORDS: List[str] = ["DC", "AUX", "AUDIO", "DIG", "STIM"]


def exclude_non_eeg_channels(raw: Any) -> List[str]:
    """
    Return a list of channel names that match non-EEG keywords.

    Useful for dropping DC/AUX/STIM channels before ICA or epoching.

    Args:
        raw: MNE Raw-like object (expects ``.ch_names``).

    Returns:
        List of channel names to exclude.
    """
    excluded = []
    for ch in raw.ch_names:
        if any(kw in ch.upper() for kw in NON_EEG_CHANNEL_KEYWORDS):
            excluded.append(ch)
    return excluded


def select_best_dc_channel(raw: Any, keywords: List[str] = DC_CHANNEL_KEYWORDS) -> str:
    """
    Select the best DC channel based on keywords and standard deviation.

    Args:
        raw: MNE Raw-like object (expects `.ch_names` and optionally `.get_data(picks=...)`)
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
    data = normalize_signal(signal_data) if normalize else signal_data.copy()

    return scipy.signal.find_peaks(data, prominence=prominence, distance=min_dist_samples, width=1)


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
    return scipy.signal.resample_poly(signal_data, target_hz, src_hz)


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

    corr = scipy.signal.correlate(rec_norm, tmpl_norm, mode="valid", method="fft")

    if len(corr) == 0:
        return 0, 0.0

    lag = int(np.argmax(np.abs(corr)))
    score = float(np.max(np.abs(corr)) / len(tmpl_norm))

    return lag, min(max(score, 0.0), 1.0)


def audio_envelope(audio: np.ndarray, sample_rate: float = None, smooth_ms: float = 20) -> np.ndarray:
    """
    Compute audio amplitude envelope using Hilbert transform (linear scale).

    Difference from compute_band_envelope:
    - No bandpass filter (broadband).
    - Returns linear amplitude (abs(analytic)).
    - Uniform smoothing.

    The envelope representation often correlates better with DC channel signals
    than raw audio waveforms, as DC channels may record amplitude-modulated signals.

    Args:
        audio: Raw audio waveform array
        sample_rate: Sample rate in Hz (required if smooth_ms > 0)
        smooth_ms: Smoothing window in milliseconds (default 20ms)

    Returns:
        Amplitude envelope of the audio
    """
    analytic_signal = scipy.signal.hilbert(audio)
    envelope = np.abs(analytic_signal)

    if smooth_ms > 0 and sample_rate is not None:
        window_samples = max(1, int(sample_rate * smooth_ms / 1000))
        envelope = uniform_filter1d(envelope, size=window_samples)

    return envelope


def compute_band_envelope(
    data: np.ndarray, sfreq: float, band: Tuple[float, float], smooth_sec: float = 0.5
) -> np.ndarray:
    """Compute smoothed band power envelope using Hilbert transform (squared).

    Difference from audio_envelope:
    - Bandpass filtered.
    - Returns POWER envelope (squared).
    - Convolution smoothing.
    """
    nyq = sfreq / 2
    b, a = scipy.signal.butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
    filtered = scipy.signal.filtfilt(b, a, data)
    envelope = np.abs(scipy.signal.hilbert(filtered)) ** 2

    # Vectorized smoothing using convolution
    window = int(smooth_sec * sfreq)
    if window > 1:
        kernel = np.ones(window) / window
        envelope = np.convolve(envelope, kernel, mode="same")
    return envelope


def highpass_filter(signal_data: np.ndarray, sfreq: float, cutoff_hz: float = 50.0, order: int = 4) -> np.ndarray:
    """
    Apply highpass filter to remove low-frequency components (e.g., baseline drift).

    Args:
        signal_data: Input signal array
        sfreq: Sampling frequency in Hz
        cutoff_hz: Cutoff frequency in Hz
        order: Order of the filter

    Returns:
        Filtered signal array
    """
    nyq = sfreq / 2
    cutoff = min(cutoff_hz, nyq - 1)
    if cutoff <= 0:
        logger.warning(f"Invalid cutoff frequency {cutoff_hz}Hz for sfreq {sfreq}Hz. Returning original signal.")
        return signal_data.copy()

    b, a = scipy.signal.butter(order, cutoff / nyq, btype="high")
    return scipy.signal.filtfilt(b, a, signal_data)


def normalize_channel_names(ch_names: List[str]) -> List[str]:
    """
    Normalize channel names by stripping common prefixes.

    Removes prefixes like 'EEG ', 'EEG-', and suffixes like '-Ref'.
    Useful for unifying channel names across different recording systems.

    Args:
        ch_names: List of original channel names.

    Returns:
        List of normalized channel names.
    """
    normalized = []
    for ch in ch_names:
        clean = ch
        clean_folded = clean.casefold()
        # Strip prefixes case-insensitively
        for prefix in ("eeg ", "eeg-"):
            if clean_folded.startswith(prefix):
                clean = clean[len(prefix) :]
                break
        # Strip suffixes/extra info
        clean = clean.replace("-Ref", "").split("-")[0]
        normalized.append(clean)
    return normalized


def compute_welch_psd(
    data: np.ndarray,
    sfreq: float,
    n_per_seg: int = None,
    n_overlap: int = None,
    fmin: float = 0.0,
    fmax: float = np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density (PSD) using Welch's method.
    Fully vectorized for (n_channels, n_times) or (n_epochs, n_channels, n_times) arrays.

    Args:
        data: Input data array. Shape (n_channels, n_times) or (n_epochs, n_channels, n_times).
        sfreq: Sampling frequency in Hz.
        n_per_seg: Length of each Welch segment in samples. Defaults to 2s if None.
        n_overlap: Overlap between segments. Defaults to 50% if None.
        fmin: Minimum frequency of interest (for output cropping).
        fmax: Maximum frequency of interest (for output cropping).

    Returns:
        freqs: Frequency array of shape (n_freqs,)
        psd: PSD array of shape (..., n_freqs) matching input batch dimensions.
    """
    if n_per_seg is None:
        n_per_seg = int(sfreq * 2)  # Default 2 second window

    if n_overlap is None:
        n_overlap = n_per_seg // 2

    # Scipy's welch is already vectorized over the last axis if specified (axis=-1)
    freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=n_per_seg, noverlap=n_overlap, axis=-1)

    # Vectorized frequency masking (no loops)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[..., mask]


def calculate_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    relative: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute average power in specific frequency bands.

    Args:
        psd: PSD array of shape (..., n_freqs).
        freqs: Frequency array of shape (n_freqs,).
        bands: Dictionary of {band_name: (low, high)}.
        relative: If True, normalize by total power (sum over all frequencies).

    Returns:
        Dictionary where keys are band names and values are arrays of power values.
    """
    # Pre-calculate frequency resolution (assuming uniform spacing)
    if len(freqs) > 1:
        dx = freqs[1] - freqs[0]
    else:
        dx = 1.0

    total_power = np.sum(psd, axis=-1, keepdims=True) * dx if relative else 1.0

    results = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)

        # Integration via sum * dx (trapezoidal approximation)
        band_power = np.sum(psd[..., mask], axis=-1) * dx

        if relative:
            band_power = band_power / total_power
            if isinstance(total_power, np.ndarray) and total_power.ndim > band_power.ndim:
                band_power = band_power.squeeze()

        results[band_name] = band_power

    return results
