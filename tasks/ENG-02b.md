# ENG-02b: ERP/Oddball Pipeline

**Status:** ✅ Complete  
**Assignee:** Arnav Dixit  
**Dependencies:** ENG-02 (Timestamp Alignment), ENG-01 (Data Loader)

---

## Overview

ERP analysis pipeline for oddball trials. It extracts EEG epochs around rare beep events, computes averaged ERPs, quantifies P300 features across midline electrodes, and generates plots and QC reports. It supports default analysis (Pz/Cz/Fz) and custom electrode sets.

## Implementation Summary

The `OddballERPPipeline` class in `src/data_processing/erp_pipeline.py` provides:

**Epoch Extraction**: Loads aligned events from ENG-02, converts Unix timestamps to EDF-relative time with timezone detection, validates epoch boundaries, and creates MNE Epochs objects.

**ERP Computation**: Averages epochs to produce clean ERP waveforms with baseline correction (-200 to 0ms).

**P300 Quantification**: Detects and measures P300 peaks (amplitude, latency). By default, the pipeline analyzes Pz/Cz/Fz, validates each electrode (positive amplitude, latency 250-600ms), and computes composite metrics from valid electrodes. It also supports custom electrode analysis.

**Batch Processing**: Processes multiple patients with progress tracking. Updates both per-session feature files and a master `p300_features.parquet` table with deduplication.

**Visualization**: Generates two-panel plots showing butterfly plots (all channels) and electrode-specific traces (Pz/Cz/Fz by default, or custom electrodes if specified). Creates individual and grand average ERPs, with automatic exclusion of aggregate files to prevent contamination.

**QC Reporting**: Produces QC reports with summary statistics and concise notes on electrode validation issues.

**Additional Capabilities**:
- Multi-session patient support (different recording dates)
- Custom electrode selection via `--electrodes` flag
- Electrode discovery via `--list-electrodes` command
- Case-insensitive electrode matching
- Graceful handling of missing data, low epoch counts, and electrode variations

## Technical Details

### Epoch Parameters

```python
ERP_CONFIG = {
    "tmin": -0.2,              # Start 200ms before stimulus
    "tmax": 0.7,               # End 700ms after stimulus
    "baseline": (None, 0),     # Baseline correction: -200ms to 0ms
    "p300_window": (0.3, 0.6), # P300 search window: 300-600ms
    "min_epochs": 2,           # Minimum rare events needed
    "midline_electrodes": ["Pz", "Cz", "Fz"]
}
```

### Data Flow

```
Aligned Events (ENG-02)
    ↓
Load Oddball Trials
    ↓
Extract Rare Beep Events
    ↓
Convert Timestamps (Unix → EDF)
    ↓
Validate Full Epoch Window Bounds
    ↓
Create MNE Epochs (-200ms to +700ms)
    ↓
Average Epochs → ERP
    ↓
Quantify P300 (amplitude, latency)
    ↓
Save Outputs + Generate Plots
```

### Output Structure

```
data/processed/
├── epochs/
│   └── {patient_id}_{date}_oddball-epo.fif
├── erps/
│   ├── {patient_id}_{date}_oddball-ave.fif
│   └── grand_average_oddball-ave.fif
├── features/
│   ├── {patient_id}_{date}_p300_features.parquet
│   └── p300_features.parquet
├── plots/erp/
│   ├── {patient_id}_{date}_oddball_erp.png
│   └── grand_average_oddball_erp.png
└── qc/
    └── erp_qc_report.json
```

### Feature Schema

The pipeline outputs 19 essential columns focused on P300 analysis:

#### Metadata
| Column | Type | Description |
|--------|------|-------------|
| patient_id | str | Patient ID |
| date | str | Session date (YYYY-MM-DD) |
| n_epochs | int | Rare beeps averaged |
| processing_timestamp | datetime | Extraction timestamp |

#### Baseline Quality
| Column | Type | Description |
|--------|------|-------------|
| baseline_std_uV | float | Baseline noise (-200 to 0ms) |

#### Individual Electrode Measurements
| Column | Type | Description |
|--------|------|-------------|
| p300_amplitude_pz_uV | float | Peak amplitude at Pz (300-600ms) |
| p300_latency_pz_ms | float | Time of peak at Pz |
| p300_amplitude_cz_uV | float | Peak amplitude at Cz |
| p300_latency_cz_ms | float | Time of peak at Cz |
| p300_amplitude_fz_uV | float | Peak amplitude at Fz |
| p300_latency_fz_ms | float | Time of peak at Fz |

#### Composite P300 (multi-electrode)
| Column | Type | Description |
|--------|------|-------------|
| p300_composite_amplitude_uV | float | Mean amplitude across valid electrodes |
| p300_composite_latency_ms | float | Mean latency across valid electrodes |
| p300_best_electrode | str | Electrode with max valid amplitude |
| p300_n_valid_electrodes | int | Valid electrode count (1-3) |
| p300_n_flagged_electrodes | int | Flagged electrode count |

#### Backward Compatibility
| Column | Type | Description |
|--------|------|-------------|
| p300_amplitude_uV | float | Alias for composite amplitude |
| p300_latency_ms | float | Alias for composite latency |

#### Quality Control
| Column | Type | Description |
|--------|------|-------------|
| qc_notes | str | QC summary (e.g., "Pz inverted, used Fz only") |

**Design Note**: The schema omits extra diagnostic fields that are not used downstream (for example, per-electrode QC booleans and redundant string lists). `qc_notes` keeps a short summary of any issues.

### Validation Criteria

An electrode passes QC if:
1. Amplitude > 0 µV (must be positive)
2. Latency in 250-600ms (healthy controls usually 300-500ms)
3. No NaN values

### Legacy Field Aliases

Two additional fields provide simplified access to the composite metrics:
- `p300_amplitude_uV` → Alias for `p300_composite_amplitude_uV`
- `p300_latency_ms` → Alias for `p300_composite_latency_ms`

These aliases let scripts access P300 values without requiring the composite prefix.

### Example: Patient with Inverted Pz
```
p300_composite_amplitude_uV: 10.05  ← Used Fz (only valid electrode)
p300_composite_latency_ms: 314.5
p300_best_electrode: "Fz"
p300_n_valid_electrodes: 1
p300_n_flagged_electrodes: 2
qc_notes: "Pz inverted, Cz inverted (used Fz only)"
```

### Electrode Selection Modes

The pipeline operates in two mutually exclusive modes:

**Default Mode** (Composite P300):
- Analyzes Pz, Cz, Fz
- Validates each electrode (amplitude > 0, latency 250-600ms)
- Computes composite amplitude/latency from valid electrodes only
- Identifies best electrode (highest valid amplitude)
- Outputs individual electrode metrics and composite metrics
- QC notes describe which electrodes were flagged and why

**Custom Mode**:
- Analyzes only the electrodes you specify with `--electrodes`
- No composite scoring, just individual measurements
- Useful for targeted checks in specific brain regions
- Outputs only individual electrode metrics + QC note indicating custom mode

```bash
# See what electrodes are available in your data
python scripts/run_erp_pipeline.py --list-electrodes

# Default mode (composite scoring with Pz/Cz/Fz)
python scripts/run_erp_pipeline.py --patient CON008

# Custom mode (analyze specific electrodes)
python scripts/run_erp_pipeline.py --patient CON008 --electrodes "T5,T6"
```

Use `--list-electrodes` before custom runs to confirm channel names. Missing electrodes produce NaN values and warnings.

### Visualization

Each patient output includes a two-panel plot:

**Panel 1 (Top)**: Butterfly plot with all EEG channels overlaid.

**Panel 2 (Bottom)**: Midline electrodes used for composite scoring:
- **Red**: Fz (frontal) - usually smaller P300
- **Green**: Cz (central) - intermediate
- **Blue**: Pz (parietal) - usually largest in healthy controls
- **Gray band**: P300 search window (300-600ms)

With `--electrodes`, Panel 2 shows the requested channels instead of Fz/Cz/Pz.

Grand-average plots use the same layout and average across all included sessions. Use `--grand-average` with `--all`.

Plots show waveforms directly. Feature extraction uses the validation and composite rules described above.

## Usage

### Command Line Interface

```bash
# Process single patient
python scripts/run_erp_pipeline.py --patient CON008

# Process all patients
python scripts/run_erp_pipeline.py --all

# Process all patients and compute grand average
python scripts/run_erp_pipeline.py --all --grand-average

# Process specific session
python scripts/run_erp_pipeline.py --patient CON008 --date 2025-08-14

# List patients with oddball data
python scripts/run_erp_pipeline.py --list

# List available electrodes (check montage)
python scripts/run_erp_pipeline.py --list-electrodes

# Custom electrode analysis
python scripts/run_erp_pipeline.py --patient CON008 --electrodes "T5,T6"

# Verbose output
python scripts/run_erp_pipeline.py --patient CON008 --verbose
```

### Python API

```python
from pathlib import Path
from src.data_processing.erp_pipeline import OddballERPPipeline

# Initialize pipeline
pipeline = OddballERPPipeline(
    data_root=Path("data"),
    output_dir=Path("data/processed"),
    verbose=True
)

# Process single patient
result = pipeline.process_patient("CON008")
print(result["features"])

# Process all patients
features_df = pipeline.process_all_patients()

# Compute grand average
grand_avg = pipeline.compute_grand_average()

# Generate QC report
qc_report = pipeline.generate_qc_report()
```

## Testing

Unit tests in `tests/test_erp_pipeline.py` cover:

- Pipeline initialization, directory creation
- Rare event extraction from aligned trials
- Timestamp conversion (Unix → EDF)
- Epoch creation parameters
- ERP computation (averaging)
- P300 peak detection
- Feature quantification (default and custom electrode modes)
- Output saving (epochs, ERPs, features)
- Plotting (individual, grand average)
- QC report generation
- Batch processing
- Grand-average exclusion of aggregate files
- Epoch boundary filtering
- ENG-02 timezone consistency
- Master feature table upsert

Run: `pytest tests/test_erp_pipeline.py -v`

## Configuration

The pipeline uses standardized output directories defined in `src/data_loading/config.py`:

```python
EPOCHS_DIR = PROCESSED_DATA_DIR / "epochs"
ERPS_DIR = PROCESSED_DATA_DIR / "erps"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
ERP_PLOTS_DIR = PROCESSED_DATA_DIR / "plots" / "erp"
QC_REPORTS_DIR = PROCESSED_DATA_DIR / "qc"
```

## Design Decisions

**1. No Artifact Rejection Yet**
- Keeping all epochs for now (ENG-03 will add ICA-based rejection)
- `reject=None` in `mne.Epochs()` to allow future filtering

**2. Multi-Electrode P300**
- Pz (parietal) is primary - usually strongest P300
- Cz, Fz as secondaries for topography
- Composite averaging across valid electrodes reduces sensitivity to single-electrode failures
- Case-insensitive matching (`Pz` = `PZ` = `pz`)

**3. Session-Level Processing**
- Each date processed separately
- Handles multi-session patients naturally
- Each session → separate epoch/ERP files
- Features aggregated in master table

**4. Timestamp Conversion**
```python
# ENG-02 gives Unix timestamps
event_unix = 1704110405.0

# Convert to EDF-relative time
edf_start_unix = raw.info['meas_date'].timestamp()
edf_time = event_unix - edf_start_unix

# Convert to sample index
sample_idx = int(edf_time * sfreq)
```

## Limitations

1. No artifact rejection yet (all epochs kept) - ENG-03 will add this
2. Fixed 300-600ms P300 window (could make adaptive later)
3. Default analysis limited to Fz/Cz/Pz (use `--electrodes` for others)
4. Loads entire EDF per session (fine for current data sizes)

## Integration Points

### Upstream Dependencies

- **ENG-02**: Provides aligned events with `event_start` timestamps
- **ENG-01**: Provides `UnifiedDataLoader` for EDF access
- **DAT-03**: Provides unified stimulus data schema

### Downstream Consumers

- **ENG-03 (Artifact Rejection)**: Processes saved epochs with ICA
- **SCI-01 (P300 Features)**: Uses P300 feature table for statistical analysis
- **VIS-01 (Validation Plots)**: Uses ERP plots for validation figures
- **MOD-01 (Feature Assembly)**: Integrates P300 features into master feature table

## Validation

Expected for healthy controls:
- P300 detection: >80%
- Amplitude: 3-10 µV at Pz
- Latency: 300-500ms (varies with age/attention)
- Topography: Maximal at parietal sites

Checklist:
- Epochs time-locked to rare beeps ✓
- Baseline correction applied (mean ≈ 0 from -200 to 0ms) ✓
- P300 visible in control patients ✓
- Feature table has all patients ✓
- Grand average shows clear P300 ✓
- Tests pass ✓
- ruff compliant ✓

## Performance

- **Single Patient**: ~10-30 seconds (depends on number of sessions/epochs)
- **Batch Processing**: ~5-10 minutes for 10 patients
- **Memory Usage**: ~500MB per patient (peak during epoch creation)
- **Output Size**: ~50-100MB per patient (epochs + ERPs + plots)

## Future Work

1. Adaptive P300 window (adjust per patient)
2. Topographic maps (spatial distribution plots)
3. Group statistics (compare patient groups)
4. N200-P300 peak-to-peak measurement
5. Parallel processing for batch mode
6. CSV/Excel export

## References

### Scientific Background

- Polich, J. (2007). "Updating P300: An integrative theory of P3a and P3b." *Clinical Neurophysiology*, 118(10), 2128-2148.
- Luck, S. J. (2014). *An Introduction to the Event-Related Potential Technique* (2nd ed.). MIT Press.

### MNE-Python Documentation

- Epoching Guide: https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html
- ERP Tutorial: https://mne.tools/stable/auto_tutorials/evoked/10_evoked_overview.html

### Internal Documentation

- `docs/Oddball_pipeline.md`: Detailed pipeline specification
- `tasks/ENG-02.md`: Timestamp alignment implementation
- `PROJECT_SCHEDULE.md`: Project timeline and milestones

## Status

**Status:** ✅ Complete  
**Implementation:** ~6 days  
**Code**: ~800 lines (pipeline) + ~400 (tests) + ~150 (CLI)

All deliverables complete:
- Core pipeline module with composite P300 scoring
- Epoch extraction and ERP computation
- Multi-electrode validation and composite scoring
- Custom electrode analysis mode
- Batch processing with master table management
- Individual and grand average visualization
- Quality control reporting with concise notes
- Comprehensive test coverage
- Full CLI with electrode discovery

Pipeline ready for ENG-03 (artifact rejection) and downstream analysis tasks.
