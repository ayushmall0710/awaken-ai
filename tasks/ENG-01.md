# ENG-01: Base Data Loader

**Task ID:** ENG-01  
**Assignee:** Arnav Dixit  
**Due Date:** January 22, 2026  
**Status:** ✅ COMPLETED  
**Completion Date:** January 21, 2026

---

## Objective

Create a Python class `EEGDataLoader` that loads EDF files (EEG recordings) and links them to harmonized CSV stimulus timing files, providing a unified interface for accessing patient data.

---

## What Was Done

### 1. Core Implementation

**File Created:** `src/data_loading/eeg_data_loader.py` (411 lines)

Implemented the `EEGDataLoader` class with the following capabilities:

#### Data Loading
- **EDF Loading**: Uses MNE-Python to load EEG recordings
  - Configurable preloading (memory vs speed tradeoff)
  - Proper error handling for corrupt files
  - Extracts channel information and metadata
  
- **CSV Loading**: Parses stimulus timing files
  - Validates required schema columns
  - Checks patient ID consistency
  - Logs trial type distribution

#### Factory Method
- `from_patient_id()`: Auto-discover EDF and CSV files for a patient
  - Eliminates manual path construction
  - Supports clipped vs raw EDF preference
  - Handles multiple file locations (main folder, old stimulus software)
  - Smart CSV selection (date-based or most recent)

#### Data Access Methods
- `get_trials(trial_type)`: Filter trials by type (language, oddball+p, commands)
- `get_trial(idx)`: Access specific trial by index
- `get_eeg_info()`: Retrieve EEG metadata (channels, sampling rate, duration)
- `get_trial_types()`: List all available trial types

#### Validation System
- **Path Validation**: Checks file existence on initialization (fail fast)
- **Schema Validation**: Ensures CSV has required columns
- **Timestamp Alignment Validation**: 
  - Compares CSV timestamps with EDF recording duration
  - Detects trials starting before recording
  - Detects trials extending beyond recording
  - Handles missing measurement dates gracefully
- **Data Completeness**: Checks for missing values in timing columns

#### Error Handling
- Custom `EEGDataLoadingError` exception for clear error types
- Warnings for data quality issues (non-blocking)
- Exceptions for critical failures (missing files, invalid schema)
- Informative error messages with specific values

### 2. Module Integration

**File Modified:** `src/data_loading/__init__.py`
- Exported `EEGDataLoader` and `EEGDataLoadingError`
- Clean module interface for imports

### 3. Testing

**Files Created:**
- `tests/test_eeg_data_loader.py`: Full test suite (6 test cases)
- `tests/test_eeg_data_loader_basic.py`: Lightweight CSV-only tests

**Test Coverage:**
- Loader initialization
- EDF and CSV loading
- Trial access and filtering
- Metadata retrieval
- Validation logic
- Error handling

**Test Results:** All tests pass ✓

### 4. Documentation

Created comprehensive usage documentation with:
- API reference with all methods
- Usage examples
- Integration patterns for downstream tasks
- Common troubleshooting scenarios

---

## Why This Design

### Architecture Decisions

1. **Class-Based Design**
   - Maintains state (EDF + CSV loaded together)
   - Enables progressive loading (can load just CSV for quick checks)
   - Encapsulates validation logic
   - Supports method chaining

2. **Separate Load Methods**
   - `load_edf()` and `load_stimulus_timing()` can be called independently
   - Useful when only CSV metadata needed (fast)
   - `load()` convenience method loads both

3. **Non-Blocking Validation**
   - Uses warnings instead of exceptions for data quality issues
   - Allows pipeline to continue with degraded data
   - Returns structured validation results for decision-making
   - Tri-state logic (True/False/None) for "couldn't check" scenarios

4. **Defensive Copying**
   - Returns `.copy()` of DataFrames from `get_trials()`
   - Prevents callers from accidentally modifying internal state
   - Small performance cost for safety

5. **Type Hints Throughout**
   - Enables IDE autocomplete and type checking
   - Documents expected types
   - Catches errors early with mypy

### Key Design Patterns

- **Fail Fast**: Validate paths in `__init__`, catch errors early
- **Progressive Disclosure**: Simple API with advanced options (preload, verbose)
- **Factory Pattern**: `from_patient_id()` for convenience, explicit constructor for control
- **Separation of Concerns**: Loading → Validation → Access
- **Principle of Least Surprise**: Methods do what names suggest

---

## Usage Example

```python
from data_loading import EEGDataLoader

# Method 1: Auto-discovery (recommended for batch processing)
loader = EEGDataLoader.from_patient_id("CON008", use_clipped=True)
loader.load()

# Method 2: Explicit paths (full control)
loader = EEGDataLoader(
    patient_id="CON008",
    edf_path="data/EEG Project Data/EEG/edf/CON008_clipped.EDF",
    stimulus_csv_path="data/EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv"
).load()

# Access trial data
language_trials = loader.get_trials(trial_type='language')
oddball_trials = loader.get_trials(trial_type='oddball+p')

# Get EEG metadata
info = loader.get_eeg_info()
print(f"Channels: {info['n_channels']}, SR: {info['sampling_rate']} Hz")

# Validate data quality
validation = loader.validate()
if not validation['timestamp_alignment']:
    print("Need DC channel alignment (ENG-02)")
```

---

## What Was NOT Done (Deferred)

### 1. Timestamp Alignment (ENG-02)
**Why deferred:** Separate task, depends on DC audio channel analysis

The loader **detects** alignment issues but doesn't fix them. ENG-02 will:
- Use DC audio channel to detect stimulus onsets
- Cross-reference with CSV timing
- Achieve sub-50ms alignment precision

**Current State:** Loader provides both `raw` (EDF) and `stimulus_df` (CSV) for ENG-02 to use.

### 2. Epoch Extraction (ENG-02b)
**Why deferred:** Requires timestamp alignment from ENG-02

The loader provides trial metadata but doesn't extract epochs. ENG-02b will:
- Use aligned timestamps to extract EEG segments
- Create MNE Epochs objects for each trial
- Extract P300 ERPs from oddball trials

**Current State:** `get_trials()` provides timing info needed for epoching.

### 3. Artifact Rejection (ENG-03)
**Why deferred:** Separate preprocessing task

The loader loads raw data but doesn't clean it. ENG-03 will:
- Apply ICA for artifact removal
- Reject bad epochs
- Generate QC reports

**Current State:** Raw EEG data accessible via `loader.raw` for preprocessing.

### 4. Batch Loading
**Why deferred:** Not in requirements, can add later if needed

Currently loads one patient at a time. Could add:
- Multi-patient loading
- Auto-discovery of matching EDF/CSV pairs
- Progress bars for large datasets

**Decision:** Single-patient focus matches current requirements and keeps API simple.

### 5. CSV Schema Harmonization (DAT-03 dependency)
**Why deferred:** Waiting on DAT-03 completion

Currently works with existing CON008 CSV structure. When DAT-03 completes:
- Update required columns list
- Add validation for new schema
- Migration is straightforward (just update `_validate_csv_schema()`)

---

## Integration Points

### Downstream Tasks Enabled

**ENG-02 (Timestamp Alignment) - Due Jan 24:**
```python
loader = EEGDataLoader(...).load()
dc_channel = loader.raw.copy().pick_channels(['DC'])
oddball_trials = loader.get_trials(trial_type='oddball+p')
# Use DC audio waveform for precise alignment
```

**ENG-02b (ERP Pipeline) - Due Jan 30:**
```python
loader = EEGDataLoader(...).load()
trials = loader.get_trials(trial_type='oddball+p')
# Extract epochs based on trial timing
```

**ENG-03+ (Artifact Rejection, Epoching):**
- Access to `loader.raw` for preprocessing
- Access to `loader.stimulus_df` for trial filtering

---

## Known Issues & Limitations

### 1. Data Quality Issue Detected
**Issue:** CON008 CSV extends 7.9 hours beyond EDF recording
- EDF duration: 64.4 minutes
- CSV last trial: 473.7 minutes after start
- **Impact:** Only first ~82 trials have EEG data
- **Resolution:** Need to investigate correct CSV file or filter trials

**Validation Response:** 
```
✗ timestamp_alignment: False (detected correctly)
✓ complete_trials: True
```

### 2. Missing Measurement Dates
Some EDF files don't have measurement dates (optional in EDF spec).
- **Impact:** Can't validate timestamp alignment using EDF clock
- **Solution:** ENG-02 will use DC audio channel (more reliable anyway)
- **Loader Response:** Sets `timestamp_alignment = None` (couldn't check)

### 3. Python 3.12+ Required
Uses modern type hints (`Union`, `Optional`).
- **Requirement:** Python 3.12+ (available in project environment)
- **Compatibility:** Could backport if needed

---

## Dependencies

### Upstream
- **DAT-03 (CSV Schema Unification)**: Loader works with existing schema, easy to update
- **DAT-01 (File Inventory)**: Assumes files are synced from OneDrive

### Downstream
- **ENG-02**: Uses `loader.raw` and `loader.stimulus_df`
- **ENG-02b**: Uses `loader.get_trials()` for epoch extraction
- **ENG-03+**: Uses `loader.raw` for preprocessing

---

## Files Created/Modified

### Created
- `src/data_loading/eeg_data_loader.py` (411 lines)
- `tests/test_eeg_data_loader.py` (178 lines)
- `tests/test_eeg_data_loader_basic.py` (142 lines)

### Modified
- `src/data_loading/__init__.py` (added exports)

### Documentation
- This file (`tasks/ENG-01.md`)
- Inline docstrings (comprehensive)

**Total Implementation:** ~750 lines of code + tests

---

## Success Criteria

✅ Load EDF files using MNE-Python  
✅ Parse and validate CSV stimulus timing files  
✅ Support patient-specific data loading  
✅ Handle both raw and clipped EDF variants  
✅ Provide trial-level access and filtering  
✅ Include error handling and validation  
✅ Zero linter errors  
✅ Comprehensive testing  
✅ Ready for ENG-02 integration  

---

## Next Steps

1. **Immediate (ENG-02 - Due Jan 24):**
   - Use loader to access DC audio channel
   - Implement precise timestamp alignment
   - Achieve sub-50ms synchronization

2. **Follow-up (ENG-02b - Due Jan 30):**
   - Use loader's trial access methods
   - Extract oddball epochs
   - Generate P300 ERPs

3. **Data Quality:**
   - Investigate CON008 CSV/EDF mismatch
   - Determine correct file or filter strategy
   - Document trial selection criteria

---

**Status: Ready for Production Use** ✅
