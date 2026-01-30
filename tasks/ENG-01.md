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
  - Auto-discovers EDF files by patient ID and session date
  - Supports multi-session patients (returns Dict[date, Raw])
  - LRU cache (min size: 5) for memory efficiency with multi-session support
  - Lazy loading - only loads when explicitly requested
  - Direct file loading via filepath parameter for advanced use

#### Cross-Patient Access Methods
- `get_all_trials()`: Get all trials from all patients
- `get_trials_by_type(trial_type, patient_ids)`: Filter by type and/or patients
- `get_patient_ids()`: List all unique patient IDs
- `get_trial_types()`: List all unique trial types
- `get_trial_summary()`: Get counts by patient and trial type

#### Single-Patient Access Methods  
- `get_patient_trials(patient_id)`: Get all trials for one patient
- `get_patient(patient_id)`: Get PatientData view object
- `get_patient_sessions(patient_id)`: List recording sessions for a patient

#### EDF Management
- `load_edf(patient_id, date, filepath, use_clipped)`: Load EDF with caching (multi-session aware)
- `get_patient_sessions(patient_id)`: List recording sessions for a patient
- `get_cached_edfs()`: Check cache statistics
- `clear_edf_cache()`: Clear cached EDFs
- `_find_edf()`: Auto-discovery logic (searches multiple locations, date-specific files)

#### Validation System
- **Schema Validation**: Validates Parquet structure on init
  - Required columns: patient_id, date, trial_type, sentences, start_time, end_time, duration, source_file
  - Data type checks for timestamps and sentences
  - Null value detection
  
- **Per-Patient Validation**: `validate_patient(patient_id)`
  - EDF file existence and loadability
  - Trial completeness (no missing timing data)
  - Sentence structure validity (List[Dict] format)
  - Timestamp alignment: Deferred to ENG-02 (DC audio channel alignment)
  
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
- `raw` (property): Lazy-loads and returns EEG data (Dict for multi-session patients)
- `get_raw(date)`: Get EEG data for specific session
- `list_sessions()`: List recording sessions for this patient
- `edf_paths` (property): Get EDF file path(s)
- `edf_filenames` (property): Get EDF filename(s)
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

### Single-Patient Workflow (Single Session)

```python
# Get focused view for one patient
patient = loader.get_patient('CON008')

# Check sessions
sessions = patient.list_sessions()
print(f"Sessions: {sessions}")  # ['2025-08-14']

# Access trial data (no EDF loaded yet)
language_trials = patient.get_trials_by_type('language')
oddball_trials = patient.get_trials_by_type('oddball')

# Lazy load EDF when needed
raw = patient.raw  # Returns single Raw for single-session
print(f"Channels: {len(raw.ch_names)}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

# Get EDF filenames
print(f"EDF file: {patient.edf_filenames}")  # 'CON008_clipped.EDF'

# Get EEG metadata
info = patient.get_eeg_info()

# Validate data quality
validation = patient.validate()
```

### Multi-Session Patient Workflow

```python
# Some patients have multiple recording sessions
patient = loader.get_patient('CON005')

# Check sessions
sessions = patient.list_sessions()
print(f"Sessions: {sessions}")  # ['2025-02-14', '2025-05-06']

# Get EDF filenames for all sessions
filenames = patient.edf_filenames
print(filenames)
# Output: {'2025-02-14': 'CON005_20250214_clipped.EDF',
#          '2025-05-06': 'CON005_20250506_clipped.EDF'}

# Load all sessions (returns Dict)
edfs = patient.raw  # Returns Dict[date, Raw] for multi-session
print(f"Type: {type(edfs)}")  # <class 'dict'>
print(f"Sessions loaded: {list(edfs.keys())}")

# Access specific session from Dict
raw_feb = edfs['2025-02-14']
raw_may = edfs['2025-05-06']

# OR: Load specific session directly
raw_specific = patient.get_raw('2025-02-14')  # Returns single Raw
print(f"Type: {type(raw_specific)}")  # <class 'mne.io.Raw'>

# Handle both single and multi-session programmatically
raw_data = patient.raw
if isinstance(raw_data, dict):
    # Multi-session: choose or loop
    for date, raw in raw_data.items():
        print(f"Session {date}: {len(raw.ch_names)} channels")
else:
    # Single session
    raw = raw_data
    print(f"Channels: {len(raw.ch_names)}")
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

### ENG-02: Timestamp Alignment (Due Jan 24)

The loader provides comprehensive support for ENG-02's DC audio channel alignment:

**Helper Methods Added:**
- `PatientData.get_dc_channel(channel_name, date)`: Extract DC audio channel
- `PatientData.get_trial_timing_info(trial_type, date)`: Get trial timing with EDF-relative times

**Multi-Session Handling:**

For single-session patients (e.g., CON008):
```python
patient = loader.get_patient('CON008')

# Extract DC channel for timestamp alignment
dc = patient.get_dc_channel('DC')
audio_data = dc.get_data()[0]
sfreq = dc.info['sfreq']

# Get oddball trial timing with EDF-relative times
timing = patient.get_trial_timing_info('oddball')
for _, trial in timing.iterrows():
    edf_start = trial['edf_start_time']  # Relative to EDF start
    sentences = trial['sentences']  # ['standard', 'rare', ...]
```

For multi-session patients (e.g., CON005):
```python
patient = loader.get_patient('CON005')
sessions = patient.list_sessions()  # ['2025-02-14', '2025-05-06']

# Process each session independently (CRITICAL)
for session_date in sessions:
    # Get session-specific DC channel
    dc = patient.get_dc_channel('DC', date=session_date)
    audio_data = dc.get_data()[0]
    
    # Get timing for THIS session only
    timing = patient.get_trial_timing_info('oddball', date=session_date)
    
    # Now edf_start_time is relative to this session's EDF
    for _, trial in timing.iterrows():
        edf_start = trial['edf_start_time']
        # ... DC audio alignment logic ...
```

**Why Session-Awareness Matters:**
- Multi-session patients have separate EDF files with different start times
- Trial times must be matched to the correct session's EDF
- DC audio beeps are session-specific
- ENG-02 must loop through sessions for multi-session patients

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
- LRU cache: Min 5 sessions (increased from 3 for multi-session support)
- Each session: ~100-200 MB per EDF
- Total cache: ~500-1000 MB (5 sessions)
- Configurable via `edf_cache_size` parameter
- Cache can be cleared with `clear_edf_cache()`
- Cache key: (patient_id, date, use_clipped) for session-specific caching

### 4. Multi-Session Support
- Some patients have multiple recording sessions (e.g., CON005: 2 sessions)
- API automatically handles single vs multi-session:
  - `load_edf(patient_id)` returns Raw for single-session, Dict[date, Raw] for multi-session
  - `load_edf(patient_id, date)` always returns single Raw
  - `get_patient_sessions(patient_id)` lists all sessions
- Cache key includes session date: (patient_id, date, use_clipped)
- File naming convention: `{patient_id}_{YYYYMMDD}_clipped.EDF`

### 5. Validation Strategy
- Timestamp alignment validation deferred to ENG-02
- ENG-02 will use DC audio channel for precise (<50ms) synchronization
- Current validation focuses on data completeness and format correctness
- Missing measurement dates handled gracefully (validation returns None)

---

## Known Limitations

### 1. Timestamp Alignment
**Status:** Deferred to ENG-02

- Current validation removed (assumed single EDF per patient)
- ENG-02 will implement DC audio channel alignment for precise (<50ms) synchronization
- PatientData.validate() returns `timestamp_alignment: None` until ENG-02 is complete

### 2. EDF File Naming Conventions
**Assumption:** Date-based naming for multi-session patients

- Expected format: `{patient_id}_{YYYYMMDD}_clipped.EDF`
- Falls back to `{patient_id}_clipped.EDF` for single-session patients
- If actual files use different naming, discovery may fail

### 3. Multi-Session API Complexity
**Type Handling:** Return types vary based on session count

- Single-session: `load_edf()` returns `mne.io.Raw`
- Multi-session: `load_edf()` returns `Dict[str, mne.io.Raw]`
- Code must check type with `isinstance()` for robustness
- Specifying `date` parameter always returns single `Raw`

### 4. EDF File Availability
**Local Sync Dependency:**

- EDF files (~100-200 MB each) not in git repository
- Must be synced from OneDrive locally
- Tests gracefully handle missing files (don't fail, just skip)
- `edf_paths` property may raise error if files not found

---

## Design Patterns Used

- **Factory Pattern**: `get_patient()` creates PatientData views
- **Lazy Loading**: EDFs loaded on first access to `.raw` property
- **LRU Caching**: Automatic memory management for EDFs (session-specific keys)
- **Defensive Copying**: All public methods return copies
- **Fail Fast**: Schema validation on initialization
- **Tri-State Logic**: True/False/None for validation results
- **View Pattern**: PatientData is lightweight view over main DataFrame
- **Union Types**: Polymorphic returns for single vs multi-session handling

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
