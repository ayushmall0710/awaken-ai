"""
Basic unit tests for timestamp alignment functionality.

These tests verify the core functionality without requiring actual EDF/CSV files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timezone


def test_imports():
    """Test that modules can be imported."""
    from data_loading import EEGDataLoader, TimestampAligner
    print("✓ Module imports successful")


def test_timestamp_conversion():
    """Test EDF time to Unix timestamp conversion."""
    from data_loading import TimestampAligner
    
    # Create a mock aligner with a known start time
    class MockLoader:
        class MockRaw:
            class MockInfo:
                sfreq = 500.0
                meas_date = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
            info = MockInfo()
            ch_names = ['DC1', 'EEG1', 'EEG2']
    
    mock_loader = MockLoader()
    mock_loader.raw = mock_loader.MockRaw()
    
    # Manually set up the aligner
    aligner = object.__new__(TimestampAligner)
    aligner.loader = mock_loader
    aligner.dc_channel_name = 'DC1'
    aligner.sampling_freq = 500.0
    aligner.edf_start_time = mock_loader.raw.info.meas_date
    
    # Test conversion
    edf_times = np.array([0.0, 1.0, 2.0, 10.0])
    unix_times = aligner.edf_time_to_unix(edf_times)
    
    # Verify
    expected_start = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    expected_times = expected_start + edf_times
    
    assert np.allclose(unix_times, expected_times), "Time conversion failed"
    print("✓ Timestamp conversion works correctly")
    print(f"  EDF times: {edf_times}")
    print(f"  Unix times: {unix_times}")


def test_peak_detection():
    """Test stimulus onset detection with synthetic data."""
    from data_loading import TimestampAligner
    
    # Create synthetic signal with known peaks
    sampling_freq = 500.0  # Hz
    duration = 10.0  # seconds
    n_samples = int(duration * sampling_freq)
    times = np.linspace(0, duration, n_samples)
    
    # Create signal with peaks at t=1, 3, 5, 7 seconds
    signal = np.random.randn(n_samples) * 0.1  # noise
    peak_times_true = np.array([1.0, 3.0, 5.0, 7.0])
    
    for peak_time in peak_times_true:
        peak_idx = int(peak_time * sampling_freq)
        # Add a peak (Gaussian bump)
        for i in range(max(0, peak_idx-50), min(n_samples, peak_idx+50)):
            t_offset = (i - peak_idx) / sampling_freq
            signal[i] += 5.0 * np.exp(-0.5 * (t_offset / 0.02) ** 2)
    
    # Create mock aligner
    class MockLoader:
        class MockRaw:
            class MockInfo:
                sfreq = sampling_freq
                meas_date = datetime.now(timezone.utc)
            info = MockInfo()
            ch_names = ['DC1']
    
    mock_loader = MockLoader()
    mock_loader.raw = mock_loader.MockRaw()
    
    aligner = object.__new__(TimestampAligner)
    aligner.loader = mock_loader
    aligner.dc_channel_name = 'DC1'
    aligner.sampling_freq = sampling_freq
    aligner.edf_start_time = mock_loader.raw.info.meas_date
    
    # Detect peaks
    peak_times, peak_values = aligner.detect_stimulus_onsets(
        signal, times, threshold=2.0, min_distance=0.5
    )
    
    # Verify we found approximately the right number of peaks
    assert len(peak_times) >= 3, f"Expected at least 3 peaks, found {len(peak_times)}"
    print(f"✓ Peak detection successful")
    print(f"  Expected peaks at: {peak_times_true}")
    print(f"  Detected {len(peak_times)} peaks at: {peak_times}")
    
    # Check approximate timing (allow some tolerance)
    for true_time in peak_times_true:
        closest_detected = peak_times[np.argmin(np.abs(peak_times - true_time))]
        error = abs(closest_detected - true_time)
        assert error < 0.1, f"Peak timing error too large: {error}s"
    
    print(f"  Peak timing accuracy: < 0.1s")


def test_alignment_validation():
    """Test alignment validation metrics."""
    from data_loading import TimestampAligner
    
    # Create mock alignment data
    alignment_df = pd.DataFrame({
        'peak_idx': [0, 1, 2, 3],
        'edf_time': [1.0, 2.0, 3.0, 4.0],
        'unix_time': [1000.0, 1001.0, 1002.0, 1003.0],
        'csv_time': [1000.01, 1000.99, 1002.02, 1003.01],
        'offset_ms': [10, -10, 20, 10]
    })
    
    # Create mock aligner
    aligner = object.__new__(TimestampAligner)
    
    # Validate
    metrics = aligner.validate_alignment(alignment_df, target_precision_ms=50)
    
    print("✓ Alignment validation successful")
    print(f"  Mean offset: {metrics['mean_offset_ms']:.2f}ms")
    print(f"  Std offset: {metrics['std_offset_ms']:.2f}ms")
    print(f"  Max offset: {metrics['max_offset_ms']:.2f}ms")
    print(f"  Within ±50ms: {metrics['within_target_pct']:.1f}%")
    print(f"  N aligned: {metrics['n_aligned']}")
    
    assert metrics['n_aligned'] == 4, "Wrong number of aligned events"
    assert metrics['within_target_pct'] == 100.0, "All events should be within target"
    assert metrics['mean_offset_ms'] < 50, "Mean offset should be < 50ms"


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running Timestamp Alignment Unit Tests")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_timestamp_conversion,
        test_peak_detection,
        test_alignment_validation
    ]
    
    failed = []
    for test in tests:
        print(f"\n{test.__name__}:")
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed.append((test.__name__, e))
    
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
        return 1
    else:
        print(f"SUCCESS: All {len(tests)} tests passed!")
        return 0


if __name__ == '__main__':
    exit(run_all_tests())
