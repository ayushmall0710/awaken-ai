from unittest.mock import MagicMock

import numpy as np
import pytest

from src.utils import signal_processing as utils

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sine_wave():
    """Create a 10Hz sine wave at 100Hz sampling."""
    return np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))


@pytest.fixture
def pulse_signal():
    """Signal with distinct peaks."""
    sig = np.zeros(100)
    sig[20] = 5.0
    sig[50] = 5.0
    sig[80] = 5.0
    return sig


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_select_best_dc_channel_single():
    """Test selecting single available DC channel."""
    raw = MagicMock()
    raw.ch_names = ["EEG1", "DC1", "EMG"]

    selected = utils.select_best_dc_channel(raw)
    assert selected == "DC1"


def test_select_best_dc_channel_multiple():
    """Test selecting best DC channel by std dev."""
    raw = MagicMock()
    raw.ch_names = ["DC1", "DC2"]
    # DC2 has higher variance
    raw.get_data.return_value = np.array(
        [[0, 1, 0], [0, 10, 0]]  # DC1 std ~ 0.47  # DC2 std ~ 4.7
    )

    selected = utils.select_best_dc_channel(raw)
    assert selected == "DC2"


def test_select_best_dc_channel_none():
    """Test raising error when no DC channel found."""
    raw = MagicMock()
    raw.ch_names = ["EEG1", "EEG2"]

    with pytest.raises(ValueError, match="No DC channel found"):
        utils.select_best_dc_channel(raw)


def test_detect_peaks(pulse_signal):
    """Test peak detectionWrapper."""
    # Prominence 1.0 < 5.0 so all should be found
    peaks, properties = utils.detect_peaks(
        pulse_signal,
        sfreq=100.0,
        prominence=1.0,
        min_distance_sec=0.1,  # 10 samples
    )

    np.testing.assert_array_equal(peaks, [20, 50, 80])
    assert "widths" in properties


def test_resample_signal(sine_wave):
    """Test signal resampling."""
    # Downsample 100Hz -> 50Hz
    resampled = utils.resample_signal(sine_wave, src_hz=100, target_hz=50)

    assert len(resampled) == 50
    # Basic shape check (std should be somewhat similar)
    assert np.isclose(np.std(resampled), np.std(sine_wave), rtol=0.1)


def test_resample_signal_same_rate(sine_wave):
    """Test fast path for same sampling rate."""
    resampled = utils.resample_signal(sine_wave, src_hz=100, target_hz=100)
    assert resampled is sine_wave  # Identity check


def test_normalize_signal(sine_wave):
    """Test Z-score normalization."""
    norm = utils.normalize_signal(sine_wave)

    assert np.isclose(np.mean(norm), 0.0)
    assert np.isclose(np.std(norm), 1.0)


def test_normalize_signal_flat():
    """Test normalization of flat signal (avoid div by zero)."""
    flat = np.zeros(100)
    norm = utils.normalize_signal(flat)
    assert np.all(norm == 0.0)


def test_cross_correlate():
    """Test cross correlation lag detection."""
    # Signal: [0, 0, 0, 10, 5, 0, 0]
    # Template: [10, 5]
    # Match should be at index 3 (where 10 starts)
    # Using more zeros to avoid edge effects in correlation
    sig = np.array([0, 0, 0, 1, -1, 0, 0])
    template = np.array([1, -1])

    lag, score = utils.cross_correlate(sig, template)

    assert lag == 3
    assert score > 0.9


def test_audio_envelope(sine_wave):
    """Test Hilbert envelope extraction."""
    # Envelope of constant sine wave is constant (amplitude)
    envelope = utils.audio_envelope(sine_wave, sample_rate=100, smooth_ms=0)

    # Envelope of sin(wt) is 1.0
    # Allowing some edge artifacts
    mid_vals = envelope[10:-10]
    assert np.allclose(mid_vals, 1.0, atol=0.1)


def test_highpass_filter():
    """Test 50Hz highpass filter removes low freq drift."""
    # Create signal: 10Hz (drift) + 100Hz (signal)
    # At 1000Hz fs
    t = np.linspace(0, 1, 1000)
    drift = np.sin(2 * np.pi * 10 * t)  # Low freq
    signal = np.sin(2 * np.pi * 100 * t)  # High freq
    combined = drift + signal

    filtered = utils.highpass_filter(combined, sfreq=1000, cutoff_hz=50)

    # Low frequency 10Hz should be attenuated (<50Hz)
    # High frequency 100Hz should be preserved (>50Hz)
    # Check middle to avoid edge effects
    assert np.allclose(filtered[100:-100], signal[100:-100], atol=0.2)
    assert np.std(filtered) < np.std(combined)
