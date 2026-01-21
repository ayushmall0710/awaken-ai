"""
Basic test script for EEGDataLoader class (without loading large EDF files).

This script verifies the loader functionality without triggering heavy MNE operations.
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
    
    # Check if files exist
    print(f"Checking files:")
    print(f"  EDF exists: {edf_path.exists()} - {edf_path}")
    print(f"  CSV exists: {csv_path.exists()} - {csv_path}")
    
    if not edf_path.exists() or not csv_path.exists():
        print("✗ Required files not found. Skipping tests.")
        return None
    
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
        import traceback
        traceback.print_exc()
        return None


def test_csv_loading(loader):
    """Test 2: Load CSV data only"""
    print("\n" + "="*70)
    print("TEST 2: Load CSV Data Only")
    print("="*70)
    
    if loader is None:
        print("✗ Skipping (no loader)")
        return False
    
    try:
        loader.load_stimulus_timing()
        print(f"✓ CSV loaded successfully")
        print(f"  Total trials: {len(loader.stimulus_df)}")
        print(f"  Columns: {list(loader.stimulus_df.columns)}")
        return True
    except Exception as e:
        print(f"✗ CSV loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trial_access(loader):
    """Test 3: Access trial metadata"""
    print("\n" + "="*70)
    print("TEST 3: Trial Metadata Access")
    print("="*70)
    
    if not loader or not loader.csv_loaded:
        print("✗ Skipping (CSV not loaded)")
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
        print(f"  - Sentences: {trial_0['sentences']}")
        
        return True
    except Exception as e:
        print(f"✗ Trial access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_file_handling():
    """Test 4: Handle missing file scenarios gracefully"""
    print("\n" + "="*70)
    print("TEST 4: Missing File Error Handling")
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
        print(f"✓ Correctly caught missing file error")
        print(f"  Error message: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected error type: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("EEGDataLoader Basic Test Suite")
    print("="*70)
    print(f"Project root: {project_root}")
    
    # Run tests
    loader = test_loader_initialization()
    
    if loader:
        csv_loaded = test_csv_loading(loader)
        if csv_loaded:
            test_trial_access(loader)
    
    test_missing_file_handling()
    
    print("\n" + "="*70)
    print("Test Suite Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
