# Timestamp Alignment Implementation

This document describes the implementation of ENG-01 (Base Data Loader) and ENG-02 (Timestamp Alignment) for the Awaken AI project.

## Overview

The timestamp alignment module synchronizes CSV Unix timestamps with EDF internal clocks using the **DC audio input channel** for precise alignment. This is critical for accurate event-related potential (ERP) analysis.

## Components

### 1. EEGDataLoader (ENG-01)

**Location:** `src/data_loading/eeg_data_loader.py`

A base class for loading EEG data from EDF files and linking to CSV stimulus logs.

**Key Features:**
- Load EDF files using MNE-Python
- Load and parse CSV stimulus logs
- Extract patient IDs from filenames
- Auto-detect DC/audio channels
- Filter trials by type (e.g., oddball trials)
- Extract specific channel data

**Usage:**
```python
from data_loading import EEGDataLoader

# Load EDF and CSV
loader = EEGDataLoader(
    edf_path='path/to/CON008.EDF',
    csv_path='path/to/CON008_stimulus_results.csv'
)

# Get information
info = loader.get_info()
print(f"Patient: {info['patient_id']}")
print(f"Duration: {info['duration_seconds']}s")

# Find DC channel
dc_channel = loader.find_dc_channel()

# Get oddball trials
oddball_trials = loader.get_oddball_trials()
```

### 2. TimestampAligner (ENG-02)

**Location:** `src/data_loading/timestamp_alignment.py`

Synchronizes CSV timestamps with EDF data using the DC audio channel.

**Alignment Strategy:**
1. Extract DC audio channel from EDF
2. Detect stimulus onset peaks using signal processing
3. Convert EDF sample times to Unix timestamps
4. Match detected onsets with CSV timestamps
5. Validate alignment precision (target: ±50ms)

**Key Features:**
- Auto-detect DC audio channel
- Peak detection with configurable thresholds
- EDF time ↔ Unix timestamp conversion
- Alignment validation metrics
- Per-trial synchronization workflow

**Usage:**
```python
from data_loading import EEGDataLoader, TimestampAligner

# Initialize loader
loader = EEGDataLoader(edf_path='CON008.EDF', csv_path='CON008.csv')

# Initialize aligner
aligner = TimestampAligner(eeg_loader=loader)

# Extract DC channel
dc_data, dc_times = aligner.extract_dc_channel()

# Detect stimulus onsets
peak_times, peak_values = aligner.detect_stimulus_onsets(dc_data, dc_times)

# Convert to Unix timestamps
peak_times_unix = aligner.edf_time_to_unix(peak_times)

# Synchronize a specific trial
trial = loader.stimulus_df.iloc[0]
alignment_df, metrics = aligner.synchronize_trial(
    trial_start_unix=trial['start_time'],
    trial_end_unix=trial['end_time']
)

print(f"Detected {metrics['n_peaks_detected']} peaks")
```

## Example Script

**Location:** `examples/timestamp_alignment_demo.py`

A demonstration script showing the complete workflow.

**Usage:**
```bash
# With both EDF and CSV
python examples/timestamp_alignment_demo.py \
    --edf data/CON008.EDF \
    --csv data/CON008_stimulus_results.csv \
    --trial-type oddball

# With just EDF (basic peak detection)
python examples/timestamp_alignment_demo.py \
    --edf data/CON008.EDF
```

## Technical Details

### DC Channel Detection

The `find_dc_channel()` method searches for channels containing keywords:
- DC, dc
- AUX, aux
- Audio, audio
- TRIG, trig

### Peak Detection Algorithm

Uses `scipy.signal.find_peaks()` with:
- **Threshold:** Automatically computed as `median + 3*MAD` (Median Absolute Deviation)
- **Min Distance:** Configurable, default 0.5s (prevents double-detection)
- **Normalization:** Data is normalized to z-scores before detection
- **Absolute Value:** Detects both positive and negative peaks

### Time Conversion

```
EDF Time (seconds from recording start)
    ↓
Unix Timestamp = EDF_start_timestamp + EDF_time
    ↓
Alignment with CSV timestamps
```

### Validation Metrics

The `validate_alignment()` method computes:
- **Mean offset:** Average absolute difference
- **Std offset:** Standard deviation of differences
- **Max offset:** Largest absolute difference
- **% within target:** Percentage within ±50ms
- **N aligned:** Number of successfully aligned events

## Dependencies

- `mne >= 1.6`: EDF file reading
- `pandas >= 2.2`: CSV handling and data frames
- `numpy >= 1.26`: Numerical operations
- `scipy >= 1.11`: Signal processing (peak detection)

## Testing

While there is no formal test suite, the implementation can be validated using:

1. **Manual verification** with the demo script on real data
2. **Jupyter notebook** `eda/dc_channel_analysis.ipynb` for interactive exploration
3. **Alignment metrics** to verify ±50ms precision target

## Next Steps (ENG-02b and beyond)

This timestamp alignment implementation provides the foundation for:

1. **ERP/Oddball Pipeline (ENG-02b):**
   - Use aligned timestamps to identify deviant beeps
   - Extract 500-700ms epochs around each stimulus
   - Average segments to reveal P300 ERP

2. **Command Following (ENG-04):**
   - Epoch motor command blocks using aligned timing
   - Analyze lateralized responses

3. **Language Tracking (ENG-05):**
   - Extract language trial segments with precise timing
   - Compute ITPC at sentence frequency

## References

- MNE-Python documentation: https://mne.tools/
- Project schedule: `PROJECT_SCHEDULE.md`
- Oddball pipeline design: `docs/Oddball_pipeline.md`
- DC channel analysis: `eda/dc_channel_analysis.ipynb`
