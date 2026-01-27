# ENG-01: Unified Data Loader

**Task ID:** ENG-01  
**Assignee:** Arnav Dixit  
**Due Date:** January 22, 2026  
**Status:** ✅ COMPLETED  
**Completion Date:** January 26, 2026

---

## Objective

Create a unified data loader that works with the DAT-03 Parquet file (containing all patients' trial data) and provides flexible access patterns for both cross-patient analysis and single-patient workflows, with on-demand EDF loading.

---

## What Was Done

### 1. Core Implementation

**Files Created:**
- `src/data_loading/unified_data_loader.py` (580 lines)
- `src/data_loading/patient_data.py` (280 lines)

### UnifiedDataLoader Class

Main class that loads the unified Parquet file and provides comprehensive access to trial data:

#### Data Loading
- **Parquet Loading**: Loads unified stimulus results at initialization
  - Validates schema on load (required columns, data types)
  - Holds all trial data in memory (efficient for 79KB Parquet)
  - Provides fast querying via pandas DataFrame
  
- **EDF Loading**: On-demand loading with LRU cache
  - Auto-discovers EDF files by patient ID
  - LRU cache (default size: 3 patients) for memory efficiency
  - Lazy loading - only loads when explicitly requested

#### Cross-Patient Access Methods
- `get_all_trials()`: Get all trials from all patients
- `get_trials_by_type(trial_type, patient_ids)`: Filter by type and/or patients
- `get_patient_ids()`: List all unique patient IDs
- `get_trial_types()`: List all unique trial types
- `get_trial_summary()`: Get counts by patient and trial type

#### Single-Patient Access Methods  
- `get_patient_trials(patient_id)`: Get all trials for one patient
- `get_patient(patient_id)`: Get PatientData view object

#### EDF Management
- `load_edf(patient_id, use_clipped)`: Load EDF with caching
- `get_cached_edfs()`: Check cache statistics
- `clear_edf_cache()`: Clear cached EDFs
- `_find_edf()`: Auto-discovery logic (searches multiple locations)

#### Validation System
- **Schema Validation**: Validates Parquet structure on init
  - Required columns: patient_id, date, trial_type, sentences, start_time, end_time, duration, source_file
  - Data type checks for timestamps and sentences
  - Null value detection
  
- **Per-Patient Validation**: `validate_patient(patient_id)`
  - EDF file existence and loadability
  - Timestamp alignment (CSV times within EDF duration)
  - Trial completeness (no missing timing data)
  - Sentence structure validity (List[Dict] format)
  
- **Cross-Patient Validation**: `validate_all_patients()`
  - Returns DataFrame with validation results per patient
  - Identifies problematic patients before analysis

#### Metadata
- `get_info()`: Overall dataset information

### PatientData View Class

Focused interface for single-patient workflows:

#### Core Features
- Lightweight view over main DataFrame (filtered to one patient)
- Lazy EDF loading via `.raw` property
- Familiar API for single-patient analysis
- Defensive copying to prevent state mutation

#### Methods
- `raw` (property): Lazy-loads and returns EEG data
- `get_trials_by_type(trial_type)`: Filter patient's trials
- `get_trial(trial_idx)`: Get specific trial by index
- `get_trial_types()`: List trial types for this patient
- `get_eeg_info()`: Get EEG recording metadata
- `validate()`: Validate this patient's data quality

### 2. Module Integration

**File Modified:** `src/data_loading/__init__.py`
- Exported `UnifiedDataLoader`, `UnifiedDataLoadingError`, `PatientData`
- Clean module interface with `__all__`

### 3. Comprehensive Testing

**File Created:** `tests/test_unified_data_loader.py` (350 lines)

**Test Coverage (7 test categories):**
1. **Initialization Tests**: Load Parquet, validate schema
2. **Cross-Patient Queries**: Test filtering, aggregation, summary stats
3. **Single-Patient Access**: PatientData creation and methods
4. **EDF Management**: Lazy loading, caching, auto-discovery
5. **Validation**: Schema, per-patient, cross-patient validation
6. **Metadata Access**: Info retrieval, EEG metadata
7. **Error Handling**: Missing files, invalid patients, out-of-range indices

**Test Results:** All tests pass ✓

### 4. Documentation

Updated comprehensive documentation with:
- Complete API reference for both classes
- Usage examples for cross-patient and single-patient workflows
- Integration patterns with downstream tasks (ENG-02, ENG-02b)
- Design decisions and architecture overview

---

## Architecture

### Data Flow

```
Unified Parquet File (79KB)
    └─> UnifiedDataLoader (loads once)
        ├─> Cross-patient queries (filter by type/patient)
        ├─> Single-patient access
        │   └─> PatientData view (lazy EDF loading)
        └─> EDF files (LRU cached, on-demand)
```

### Design Decisions

1. **Fresh API Design**
   - No compatibility constraints with old EEGDataLoader
   - Optimized for both cross-patient and single-patient use cases
   - Clean separation between bulk queries and focused views

2. **Parquet in Memory**
   - File is only 79KB (trial metadata, not raw EEG)
   - Enables fast pandas queries without I/O
   - Reasonable memory footprint

3. **LRU Cache for EDFs**
   - Default size: 3 patients (~150-300 MB per EDF)
   - Automatic eviction of least recently used
   - Adjustable via `edf_cache_size` parameter
   - Balances speed and memory usage

4. **Lazy EDF Loading**
   - EDFs only loaded when explicitly requested
   - Enables working with trial metadata without loading heavy files
   - PatientData uses property for transparent lazy loading

5. **Defensive Copying**
   - All public methods return `.copy()` of DataFrames
   - Prevents accidental state mutation
   - Small performance cost for safety

6. **Comprehensive Validation**
   - Schema validation on initialization (fail fast)
   - Per-patient validation on demand
   - Cross-patient validation summary for batch workflows
   - Tri-state logic (True/False/None) for "couldn't check" cases

---

## Usage Examples

### Cross-Patient Analysis

```python
from data_loading import UnifiedDataLoader

# Load unified data
loader = UnifiedDataLoader("data/EEG/unified_stimulus_results.parquet")

# Get all language trials across all patients
all_language = loader.get_trials_by_type('language')
print(f"Total language trials: {len(all_language)}")

# Filter to specific patients
con8_con9_oddball = loader.get_trials_by_type(
    'oddball', 
    patient_ids=['CON008', 'CON009']
)

# Get summary statistics
summary = loader.get_trial_summary()
print(summary)
# Output:
#   patient_id  trial_type      count
#   CON008      language        72
#   CON008      oddball         5
#   CON009      language        89
#   ...

# Get all patient IDs
patient_ids = loader.get_patient_ids()

# Get all trial types
trial_types = loader.get_trial_types()
```

### Single-Patient Workflow

```python
# Get focused view for one patient
patient = loader.get_patient('CON008')

# Access trial data (no EDF loaded yet)
language_trials = patient.get_trials_by_type('language')
oddball_trials = patient.get_trials_by_type('oddball')

# Lazy load EDF when needed
raw = patient.raw  # Triggers EDF loading on first access
print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

# Get EEG metadata
info = patient.get_eeg_info()

# Validate data quality
validation = patient.validate()
if not validation['timestamp_alignment']:
    print("Need DC channel alignment (ENG-02)")
```

### Batch Processing

```python
# Validate all patients before processing
validation_df = loader.validate_all_patients()
valid_patients = validation_df[
    validation_df['edf_exists'] & 
    validation_df['edf_loadable']
]['patient_id'].tolist()

# Process only valid patients
for patient_id in valid_patients:
    patient = loader.get_patient(patient_id)
    
    # Process oddball trials
    oddball = patient.get_trials_by_type('oddball')
    if len(oddball) > 0:
        raw = patient.raw
        # ... analysis code
        
# Check cache stats
cache_stats = loader.get_cached_edfs()
print(f"Cache: {cache_stats['size']}/{cache_stats['maxsize']} EDFs")

# Clear cache if needed
loader.clear_edf_cache()
```

---

## Integration with Downstream Tasks

### ENG-02 (Timestamp Alignment) - Due Jan 24

```python
loader = UnifiedDataLoader(parquet_path)
patient = loader.get_patient('CON008')

# Get oddball trials for alignment
oddball_trials = patient.get_trials_by_type('oddball')

# Access DC channel for precise alignment
dc_channel = patient.raw.copy().pick_channels(['DC'])

# Use DC audio waveform to detect beep onsets
# Cross-reference with oddball_trials timestamps
# Achieve sub-50ms alignment precision
```

### ENG-02b (ERP Pipeline) - Due Jan 30

```python
loader = UnifiedDataLoader(parquet_path)

# Process all patients with oddball data
for patient_id in loader.get_patient_ids():
    patient = loader.get_patient(patient_id)
    
    # Get oddball trials
    oddball_trials = patient.get_trials_by_type('oddball')
    
    if len(oddball_trials) > 0:
        # Extract epochs based on trial timing
        raw = patient.raw
        # Create MNE Epochs objects
        # Extract P300 ERPs
```

### Cross-Patient Analysis

```python
loader = UnifiedDataLoader(parquet_path)

# Get all oddball trials across patients
all_oddball = loader.get_trials_by_type('oddball')

# Compare P300 responses across patients
for patient_id in loader.get_patient_ids():
    patient = loader.get_patient(patient_id)
    oddball = patient.get_trials_by_type('oddball')
    raw = patient.raw
    # Extract P300 features
    # Aggregate for group analysis
```

---

## Files Created/Modified

### Created
- `src/data_loading/unified_data_loader.py` (580 lines) - Main loader class
- `src/data_loading/patient_data.py` (280 lines) - Patient view class
- `tests/test_unified_data_loader.py` (350 lines) - Comprehensive test suite

### Modified
- `src/data_loading/__init__.py` - Added exports for new classes

### Documentation
- This file (`tasks/ENG-01.md`) - Updated with new implementation

**Total Implementation:** ~1,210 lines of code + tests + documentation

---

## Dependencies

### Upstream
- **DAT-03 (CSV Schema Unification)**: ✅ Completed - unified Parquet file available
  - Location: `data/EEG/unified_stimulus_results.parquet` (79KB)
  - Schema: patient_id, date, trial_type, sentences (List[Dict]), timestamps, source_file

### Downstream (Enables)
- **ENG-02 (Timestamp Alignment)**: Uses `patient.raw` and `patient.get_trials_by_type()`
- **ENG-02b (ERP Pipeline)**: Uses trial metadata and EEG data access
- **ENG-03+ (Artifact Rejection, Feature Extraction)**: Uses validated data access

### Python Dependencies
- pandas (Parquet reading, DataFrame operations)
- numpy (array operations)
- mne (EDF loading, EEG processing)
- pyarrow (Parquet format support)
- functools (LRU cache)

All dependencies already in `requirements.txt` ✓

---

## Success Criteria

✅ Load unified Parquet file efficiently (79KB loads instantly)  
✅ Support cross-patient queries (filtering, aggregation)  
✅ Support single-patient workflows (PatientData view)  
✅ LRU cache working correctly (auto-evicts old EDFs)  
✅ Full validation suite operational (schema, per-patient, cross-patient)  
✅ All tests passing (7 test categories, 100% pass rate)  
✅ Zero linter errors  
✅ Comprehensive documentation with usage examples  
✅ Ready for ENG-02 timestamp alignment integration  

---

## Known Considerations

### 1. Parquet File Structure
- **Sentences Column**: List[Dict] format with 'event' and 'onset_time' keys
- **Trial Types**: Normalized by DAT-03 (language, oddball, left_command, right_command, etc.)
- **Source File**: Tracks provenance for debugging

### 2. EDF File Discovery
- Searches multiple locations: main edf folder, old stimulus software folder
- Prefers clipped files over raw (configurable with `use_clipped` parameter)
- Clear error messages if files not found

### 3. Memory Management
- Parquet in memory: 79KB (negligible)
- LRU cache: Default 3 patients × ~100-200 MB per EDF = ~300-600 MB
- Configurable via `edf_cache_size` parameter
- Cache can be cleared with `clear_edf_cache()`

### 4. Validation Warnings
- Some patients may have timestamp misalignment (detected by validation)
- ENG-02 will resolve alignment issues using DC audio channel
- Missing measurement dates handled gracefully (sets validation to None)

---

## Design Patterns Used

- **Factory Pattern**: `get_patient()` creates PatientData views
- **Lazy Loading**: EDFs loaded on first access to `.raw` property
- **LRU Caching**: Automatic memory management for EDFs
- **Defensive Copying**: All public methods return copies
- **Fail Fast**: Schema validation on initialization
- **Tri-State Logic**: True/False/None for validation results
- **View Pattern**: PatientData is lightweight view over main DataFrame

---

## Next Steps

### Immediate (ENG-02 - Due Jan 24)
- Use `UnifiedDataLoader` to access patient data
- Implement DC audio channel alignment
- Achieve sub-50ms synchronization

### Follow-up (ENG-02b - Due Jan 30)
- Use `PatientData.get_trials_by_type()` for trial filtering
- Extract oddball epochs
- Generate P300 ERPs

### Future Enhancements (Optional)
- Export methods (to_csv, to_hdf5)
- Integration with DAT-04 patient metadata
- Stimulus manifest integration
- Progress bars for batch processing
- Advanced caching strategies

---

**Status: ✅ Ready for Production Use**

The UnifiedDataLoader provides a robust, efficient, and flexible foundation for all downstream EEG analysis tasks. Both cross-patient and single-patient workflows are fully supported with comprehensive validation and error handling.
