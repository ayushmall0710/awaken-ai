"""
Test script for EEGDataLoader class.

This script verifies the loader functionality with CON008 data.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_loading.eeg_data_loader import EEGDataLoader, EEGDataLoadingError


def test_loader_initialization():
    """Test 1: Initialize loader with CON008 data"""
    print("\n" + "="*70)
    print("TEST 1: Loader Initialization")
    print("="*70)
    
    data_root = project_root / "data" / "EEG Project Data" / "EEG"
    edf_path = data_root / "edf" / "CON008_clipped.EDF"
    csv_path = data_root / "CON008_2025-08-14_stimulus_results.csv"
    
    try:
        loader = EEGDataLoader(
            patient_id="CON008",
            edf_path=edf_path,
            stimulus_csv_path=csv_path,
            verbose=False
        )
        print(f"✓ Loader initialized: {loader}")
        return loader
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None


def test_load_data(loader):
    """Test 2: Load EDF and CSV data"""
    print("\n" + "="*70)
    print("TEST 2: Load EDF and CSV Data")
    print("="*70)
    
    if loader is None:
        print("✗ Skipping (no loader)")
        return False
    
    try:
        loader.load()
        print(f"✓ Data loaded successfully")
        print(f"  {loader}")
        return True
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return False


def test_trial_access(loader):
    """Test 3: Access trial metadata for different trial types"""
    print("\n" + "="*70)
    print("TEST 3: Trial Metadata Access")
    print("="*70)
    
    if not loader or not loader.csv_loaded:
        print("✗ Skipping (data not loaded)")
        return
    
    try:
        # Get all trial types
        trial_types = loader.get_trial_types()
        print(f"✓ Trial types found: {trial_types}")
        
        # Get trials for each type
        for trial_type in trial_types:
            trials = loader.get_trials(trial_type=trial_type)
            print(f"  - {trial_type}: {len(trials)} trials")
        
        # Get specific trial
        trial_0 = loader.get_trial(0)
        print(f"\n✓ First trial details:")
        print(f"  - Type: {trial_0['trial_type']}")
        print(f"  - Start: {trial_0['start_time']}")
        print(f"  - Duration: {trial_0['duration']:.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ Trial access failed: {e}")
        return False


def test_eeg_info(loader):
    """Test 4: Verify EDF channels are accessible"""
    print("\n" + "="*70)
    print("TEST 4: EEG Channel Information")
    print("="*70)
    
    if not loader or not loader.edf_loaded:
        print("✗ Skipping (EDF not loaded)")
        return
    
    try:
        info = loader.get_eeg_info()
        print(f"✓ EEG Info retrieved:")
        print(f"  - Patient: {info['patient_id']}")
        print(f"  - Channels: {info['n_channels']}")
        print(f"  - Sampling rate: {info['sampling_rate']} Hz")
        print(f"  - Duration: {info['duration_minutes']:.1f} minutes")
        print(f"  - Measurement date: {info['measurement_date']}")
        
        # Show first few channels
        print(f"\n  First 10 channels:")
        for i, (ch_name, ch_type) in enumerate(zip(info['channel_names'][:10], info['channel_types'][:10])):
            print(f"    {i+1}. {ch_name} ({ch_type})")
        
        return True
    except Exception as e:
        print(f"✗ EEG info access failed: {e}")
        return False


def test_timestamp_validation(loader):
    """Test 5: Check that trial timestamps are within EDF duration"""
    print("\n" + "="*70)
    print("TEST 5: Timestamp Validation")
    print("="*70)
    
    if not loader or not loader.edf_loaded or not loader.csv_loaded:
        print("✗ Skipping (data not loaded)")
        return
    
    try:
        validation_results = loader.validate()
        print(f"✓ Validation completed:")
        for key, value in validation_results.items():
            status = "✓" if value else "✗" if value is False else "⚠"
            print(f"  {status} {key}: {value}")
        
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def test_missing_file_handling():
    """Test 6: Handle missing file scenarios gracefully"""
    print("\n" + "="*70)
    print("TEST 6: Missing File Error Handling")
    print("="*70)
    
    try:
        # Try to load with non-existent EDF
        loader = EEGDataLoader(
            patient_id="FAKE",
            edf_path="/fake/path/to/file.EDF",
            stimulus_csv_path="/fake/path/to/file.csv"
        )
        print("✗ Should have raised an error for missing files")
        return False
    except EEGDataLoadingError as e:
        print(f"✓ Correctly caught missing file error: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("EEGDataLoader Test Suite")
    print("="*70)
    print(f"Project root: {project_root}")
    
    # Run tests
    loader = test_loader_initialization()
    
    if loader:
        test_load_data(loader)
        test_trial_access(loader)
        test_eeg_info(loader)
        test_timestamp_validation(loader)
    
    test_missing_file_handling()
    
    print("\n" + "="*70)
    print("Test Suite Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
