# ENG-02b: ERP/Oddball Pipeline

**Status:** ✅ Complete  
**Assignee:** Arnav Dixit  
**Dependencies:** ENG-02 (Timestamp Alignment), ENG-03 (Artifact Rejection), ENG-01 (Data Loader)

---

## Overview

ERP analysis pipeline for oddball trials. Loads ICA-cleaned 35-second oddball epochs produced by ENG-03, maps rare-event timestamps into those trial windows, extracts short 900ms sub-epochs around each rare beep, computes averaged ERPs, quantifies P300 features across midline electrodes, and generates plots and QC reports. Supports default analysis (Pz/Cz/Fz) with composite scoring and custom electrode sets.

### ENG-03 Epoch Integration

The pipeline reads ENG-03's saved oddball trial epochs from disk:
- Loads `data/processed/epochs/{patient_id}/{date}/oddball-epo.fif` via `UnifiedDataLoader.load_clean_epochs()`
- Maps each rare-event timestamp from ENG-02 aligned events into the appropriate 35s trial window
- Extracts 900ms sub-epochs (`-200ms to +700ms` around each rare beep) from within those trials
- **Dependency**: ENG-03 must be run first (`ArtifactRejector.run_session(save=True)`) to generate the epoch files
- Fails with clear error if the `.fif` file is missing

Rare events that fall inside trials dropped by ENG-03's PTP rejection are silently excluded (no fallback to raw EEG).

## Implementation Summary

The `OddballERPPipeline` class in `src/data_processing/erp_pipeline.py` provides:

**Trial Window Mapping**: Reads ENG-03 epoch metadata (`start_time_unix`, `end_time_unix`) to build a table of trial time windows. Each rare-event timestamp is matched to its parent 35s trial by checking `trial_start <= event_time < trial_start + 35s`. Events that don't map to any surviving trial, map to multiple trials, or whose sub-epoch window crosses a trial boundary are excluded.

**Sub-Epoch Extraction**: For each mapped rare event, slices a 900ms window from the parent 35s epoch's data array. The result is an `mne.EpochsArray` with the same channel set and sampling rate as ENG-03's output, with baseline correction applied (`-200ms to 0ms`).

**ERP Computation**: Averages sub-epochs to produce clean ERP waveforms.

**P300 Quantification**: Detects and measures P300 peaks (amplitude, latency). By default, the pipeline analyzes Pz/Cz/Fz, validates each electrode (positive amplitude, latency 250-600ms), and computes composite metrics from valid electrodes. Also supports custom electrode analysis.

**Batch Processing**: Processes multiple patients with progress tracking. All features go into a single `p300_features.parquet` master table with deduplication on `(patient_id, date)`.

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
Aligned Events (ENG-02)          ENG-03 Oddball Epochs (35s .fif)
    ↓                                 ↓
Extract Rare Beep Timestamps     Build Trial Windows (metadata)
    ↓                                 ↓
    └──────── Map Rare Events ────────┘
                   ↓
    Extract 900ms Sub-Epochs from 35s Trials
                   ↓
          Average Sub-Epochs → ERP
                   ↓
    Quantify P300 (amplitude, latency, composite)
                   ↓
       Save Outputs + Generate Plots
```

### Output Structure

```
data/processed/
├── erps/
│   ├── {patient_id}_{date}_oddball-ave.fif
│   └── grand_average_oddball-ave.fif
├── features/
│   └── p300_features.parquet
├── plots/erp/
│   ├── {patient_id}_{date}_oddball_erp.png
│   └── grand_average_oddball_erp.png
└── qc/
    └── erp_qc_report.json
```

> 900ms epochs are not saved — they're re-sliced from ENG-03's 35s `.fif` on demand.

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

**Mapping diagnostics**: The parquet also includes `n_rare_events`, `n_mapped`, `n_unmapped`, `n_duplicate`, `n_boundary_clipped`, `mapping_rate` from the trial-mapping step.

### Validation Criteria

An electrode passes QC if:
1. Amplitude > 0 µV (must be positive)
2. Latency in 250-600ms (healthy controls usually 300-500ms)
3. No NaN values

### Legacy Field Aliases

Two additional fields provide simplified access to the composite metrics:
- `p300_amplitude_uV` → Alias for `p300_composite_amplitude_uV`
- `p300_latency_ms` → Alias for `p300_composite_latency_ms`

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

## Usage

### Prerequisites

1. **ENG-02** must be run first so aligned events exist: `data/processed/aligned_events/{patient_id}_events.parquet`.
2. **ENG-03** must be run first to generate cleaned oddball epoch files:
   ```python
   from src.data_processing.artifact_rejection import ArtifactRejector
   ar = ArtifactRejector(verbose=True)
   ar.run_session(patient_id="CON008", date="2025-08-14", save=True)
   ```
   This creates `data/processed/epochs/CON008/2025-08-14/oddball-epo.fif`.

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

# Custom output directory
python scripts/run_erp_pipeline.py --all --output-dir /path/to/output

# Verbose output
python scripts/run_erp_pipeline.py --patient CON008 --verbose
```

### Testing with real data locally

1. Run ENG-02 for your patient, then ENG-03 (`ArtifactRejector.run_session(save=True)`).
2. Run the ERP pipeline:
   ```bash
   cd awaken-ai
   python scripts/run_erp_pipeline.py --patient CON008 --verbose
   ```
3. Logs should show: `Loading ENG-03 oddball epochs for CON008 - 2025-08-14` then mapping, sub-epoch extraction, and P300 quantification.
4. Outputs under `data/processed/`: `features/p300_features.parquet`, `plots/erp/*.png`, `erps/*-ave.fif`.
5. If you see `ENG-03 oddball epochs not found`, run ENG-03 for that patient/date first.

### Python API

```python
from pathlib import Path
from src.data_processing.erp_pipeline import OddballERPPipeline

pipeline = OddballERPPipeline(
    data_root=Path("data"),
    output_dir=Path("data/processed"),
    verbose=True
)

result = pipeline.process_patient("CON008")
print(result["features"])

features_df = pipeline.process_all_patients()

grand_avg = pipeline.compute_grand_average()

qc_report = pipeline.generate_qc_report()
```

## Testing

Unit tests in `tests/test_erp_pipeline.py` cover:

- Pipeline initialization, directory creation
- Rare event extraction from aligned trials
- Trial window building from ENG-03 epoch metadata
- Rare-event-to-trial mapping (all mapped, unmapped, boundary clipped)
- Sub-epoch extraction (shape, timing, empty case)
- ERP computation (averaging)
- P300 peak detection
- Feature quantification (default and custom electrode modes)
- Output saving (ERPs, features)
- Plotting (individual, grand average)
- QC report generation
- Batch processing
- Grand-average exclusion of aggregate files
- Master feature table upsert
- ENG-03 integration (missing epochs error, successful load)
- Full pipeline integration with mocked ENG-03 epochs

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

**1. ENG-03 Epoch Reuse**
- Loads pre-computed artifact-cleaned 35s epochs from disk rather than re-running ICA each time
- Maps rare events into those epochs via Unix timestamps and extracts 900ms sub-epochs
- Events from ENG-03-dropped trials are silently excluded — no fallback to raw EEG
- Trade-off: some rare events may be lost due to trial-level PTP rejection, but all data used is artifact-cleaned

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

## Limitations

1. Fixed 300-600ms P300 window (could make adaptive later)
2. Default analysis limited to Fz/Cz/Pz (use `--electrodes` for others)
3. Rare events in ENG-03-dropped trials are lost (depends on ENG-03 rejection settings)
4. Event count per session is small (typically 9-15 rare events), making ERP averages sensitive to dropped trials

## Integration Points

### Upstream Dependencies

- **ENG-02**: Provides aligned events with `event_start` timestamps
- **ENG-03**: Provides ICA-cleaned 35s oddball epochs (`.fif` files)
- **ENG-01**: Provides `UnifiedDataLoader` for epoch loading
- **DAT-03**: Provides unified stimulus data schema

### Downstream Consumers

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

- **Single Patient**: ~5-15 seconds (no ICA re-run, loads pre-computed ENG-03 epochs)
- **Batch Processing**: ~2-5 minutes for 10 patients
- **Memory Usage**: ~200MB per patient (peak during epoch loading)
- **Output Size**: ~5-10MB per patient (ERP `.fif` + plot; 900ms epochs are not saved)

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
