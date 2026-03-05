# ENG-02b: ERP/Oddball Pipeline

**Status:** ✅ Complete  
**Assignee:** Arnav Dixit  
**Dependencies:** ENG-02 (Timestamp Alignment), ENG-03 (Artifact Rejection), ENG-01 (Data Loader)

---

## Overview

ERP analysis pipeline for oddball trials. Loads ICA-cleaned 35-second oddball epochs produced by ENG-03, maps rare-event (target) and standard-event (frequent) timestamps into those trial windows, extracts short 900ms sub-epochs, computes averaged ERPs, computes the difference wave (target − standard) to isolate the P300 component, quantifies P300 features across midline electrodes, and generates 3-panel plots plus standalone topomaps and single-trial heatmaps. Supports default analysis (Pz/Cz/Fz) with composite scoring, P3a/P3b subtyping, and custom electrode sets.

### ENG-03 Epoch Integration

The pipeline reads ENG-03's saved oddball trial epochs from disk:
- Loads `data/processed/epochs/{patient_id}/{date}/oddball-epo.fif` via `UnifiedDataLoader.load_clean_epochs()`
- Maps each rare-event timestamp from ENG-02 aligned events into the appropriate 35s trial window
- Extracts 900ms sub-epochs (`-200ms to +700ms` around each rare beep) from within those trials
- **Dependency**: ENG-03 must be run first (`ArtifactRejector.run_session(save=True)`) to generate the epoch files
- Fails with clear error if the `.fif` file is missing

Rare events that fall inside trials dropped by ENG-03's PTP rejection are silently excluded (no fallback to raw EEG).

**Standard (Frequent) Events**: The pipeline also extracts standard stimulus events from the aligned events using the label set `STANDARD_EVENT_LABELS = {"standard", "frequent"}`. Standard epochs are averaged in parallel to the rare epochs, allowing computation of the difference wave (rare − standard) which isolates the endogenous P300 component by removing the auditory N1-P2 envelope.

## Implementation Summary

The `OddballERPPipeline` class in `src/data_processing/erp_pipeline.py` provides:

**Trial Window Mapping**: Reads ENG-03 epoch metadata (`start_time_unix`, `end_time_unix`) to build a table of trial time windows. Each rare-event timestamp is matched to its parent 35s trial by checking `trial_start <= event_time < trial_start + 35s`. Events that don't map to any surviving trial, map to multiple trials, or whose sub-epoch window crosses a trial boundary are excluded.

**Sub-Epoch Extraction**: For each mapped rare event, slices a 900ms window from the parent 35s epoch's data array. The result is an `mne.EpochsArray` with the same channel set and sampling rate as ENG-03's output, with baseline correction applied (`-200ms to 0ms`).

**ERP Computation**: Averages sub-epochs to produce clean ERP waveforms.

**P300 Quantification**: Detects and measures P300 peaks (amplitude, latency). By default, the pipeline analyzes Pz/Cz/Fz, validates each electrode (positive amplitude, latency 250-600ms), and computes composite metrics from valid electrodes. Also supports custom electrode analysis.

**Batch Processing**: Processes multiple patients with progress tracking. All features go into a single `p300_features.parquet` master table with deduplication on `(patient_id, date)`.

**Visualization**: Generates 4-panel plots (when standard ERPs available): (1) butterfly plot of rare ERP, (2) rare vs standard ERP overlay with ±1 SEM bands, (3) difference wave with ±1 SEM. Also generates standalone topomap series (100ms snapshots from -200 to +700ms) of the difference ERP, and single-trial heatmap (ERP image) at Pz for quality control (only if ≥3 epochs). Creates individual and grand average ERPs, with automatic exclusion of aggregate files to prevent contamination.

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
Extract Rare Timestamps          Build Trial Windows (metadata)
Extract Standard Timestamps      (start_time_unix, end_time_unix)
    ↓                                 ↓
    ├── Map Rare Events ─────────────┤
    │                                 │
    ├── Map Standard Events ─────────┤
    │                                 │
    └─ Extract Rare/Standard 900ms Sub-Epochs
                   ↓
    Average Rare Epochs → Rare ERP
    Average Standard Epochs → Standard ERP
                   ↓
    Compute Difference Wave (Rare − Standard) → Diff ERP
                   ↓
    Quantify P300 from Rare ERP (p300_* columns)
    Quantify P300 from Diff ERP (diff_* columns)
    Compute P3a/P3b Subtype
                   ↓
    Save ERPs (.fif) + Generate 4-Panel Plots + Topomaps + ERP Image
    Save Features (p300_features.parquet)
```

### Output Structure

```
data/processed/
├── erps/
│   ├── {patient_id}_{date}_oddball-ave.fif              ← rare ERP
│   ├── {patient_id}_{date}_oddball_standard-ave.fif     ← standard ERP (if computed)
│   ├── {patient_id}_{date}_oddball_diff-ave.fif         ← difference ERP (if computed)
│   └── grand_average_oddball-ave.fif
├── features/
│   ├── p300_oddball_clinical.parquet          ← Table 1: Main analysis (one row per session)
│   ├── p300_oddball_electrode_detail.parquet  ← Table 2: Per-electrode detail (3 rows per session)
│   └── p300_oddball_mapping_qc.parquet        ← Table 3: Mapping & QC diagnostics (one row per session)
├── plots/erp/
│   ├── {patient_id}_{date}_oddball_erp.png              ← 3-panel (butterfly, rare+std, diff)
│   ├── {patient_id}_{date}_oddball_topomap.png          ← standalone topomap series (diff ERP)
│   ├── {patient_id}_{date}_oddball_erp_image.png        ← ERP image (single-trial heatmap at Pz, if ≥3 epochs)
│   └── grand_average_oddball_erp.png
└── qc/
    └── erp_qc_report.json
```

> 900ms epochs are not saved — they're re-sliced from ENG-03's 35s `.fif` on demand.

### Metric Semantics

**Primary P300 Metric**: The `diff_*` columns (difference wave amplitude/latency) are the **primary scientific metric** for P300, as they isolate the endogenous cognitive component by subtracting the standard (frequent) ERP from the rare (target) ERP. This eliminates the auditory N1-P2 envelope and exogenous noise that would otherwise contaminate peak measurements from the rare ERP alone.

**Secondary Metrics**: The `p300_*` columns (rare ERP amplitude/latency) are retained for backward compatibility and as a reference for data quality assessment. These columns show the raw rare ERP measurements but should not be used as the primary P300 measure.

**Subtype Classification**: The `p300_subtype` column ("P3a", "P3b", "mixed", "absent") is determined from the rare ERP composite QC logic (which electrode is valid and maximal) and indicates the likely functional class of the detected potential:
- **P3b** (Pz-max): Working memory update, context closure (parietal-dominant)
- **P3a** (Fz-max): Novelty orienting, stimulus-driven (frontal-dominant)
- **mixed** (Cz-max): Intermediate or transitional
- **absent** (n_valid_electrodes == 0): No valid P300 detected

### Feature Schema

The pipeline outputs **three structured tables** for different analytical purposes:

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
| qc_notes | str | QC summary with subtype separator (e.g., "Pz inverted, used Fz only; P3a pattern (Fz-max) — P3b may be absent") |

#### Difference Wave P300 (Primary Metric)
|| Column | Type | Description |
||--------|------|-------------|
|| diff_amplitude_Pz_uV | float | Peak amplitude in diff ERP at Pz (300-600ms) |
|| diff_latency_Pz_ms | float | Time of peak in diff ERP at Pz |
|| diff_amplitude_Cz_uV | float | Peak amplitude in diff ERP at Cz |
|| diff_latency_Cz_ms | float | Time of peak in diff ERP at Cz |
|| diff_amplitude_Fz_uV | float | Peak amplitude in diff ERP at Fz |
|| diff_latency_Fz_ms | float | Time of peak in diff ERP at Fz |

#### Standard Event Diagnostics
|| Column | Type | Description |
||--------|------|-------------|
|| n_standard_events | int | Total standard events found in aligned data |
|| n_standard_epochs | int | Standard epochs successfully extracted (≥ min_epochs for diff wave) |

#### Subtype Classification
|| Column | Type | Description |
||--------|------|-------------|
|| p300_subtype | str | "P3a" (Fz-max), "P3b" (Pz-max), "mixed" (Cz-max), or "absent" |

**Mapping diagnostics**: The parquet also includes `n_rare_events`, `n_mapped`, `n_unmapped`, `n_duplicate`, `n_boundary_clipped`, `mapping_rate` from the trial-mapping step.

### Validation Criteria

An electrode passes QC if:
1. Amplitude > 0 µV (must be positive)
2. Latency in 250-600ms (healthy controls usually 300-500ms)
3. No NaN values

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

#### 3-Panel ERP Plot

Each patient output includes a three-panel plot (when standard/difference ERPs available):

**Panel 1 (Top)**: Butterfly plot with all EEG channels overlaid, showing the rare (target) ERP.

**Panel 2 (Middle)**: Rare vs. standard ERP overlay with ±1 SEM bands:
- **Red**: Fz (frontal) - usually smaller P300
- **Green**: Cz (central) - intermediate
- **Blue**: Pz (parietal) - usually largest in healthy controls
- **Shaded bands**: Standard error of the mean (SEM) around each condition
- **Gray band**: P300 search window (300-600ms)

**Panel 3 (Bottom)**: Difference wave (rare − standard) with ±1 SEM shading. This isolates the endogenous P300 component by removing the auditory N1-P2 envelope common to both conditions.

With `--electrodes`, the midline panels show the requested channels instead of Fz/Cz/Pz.

Grand-average plots use the same layout and average across all included sessions. Use `--grand-average` with `--all`.

#### Topomap Series (Standalone File)

A separate PNG file (`{patient_id}_{date}_oddball_topomap.png`) contains a series of 10 topographic maps of the difference ERP, plotted at 100ms intervals from −200ms to +700ms post-stimulus. Each topographic map shows:

- **Circular view** of the scalp from above
- **Electrode positions** as small dots overlaid on the circular scalp (implicit in each map)
- **Color scale** from blue (negative/baseline) to red (positive/P300 activity)
- **Time label** above each map (e.g., "−0.2s", "0.0s", "0.3s", etc.)
- **Colorbar** on the right showing the voltage scale (microvolts)

**Reading the topomaps**:
- **Time 0 marks stimulus onset** (rare beep presentation)
- **Pre-stimulus (−0.2 to 0s)**: Baseline activity; should be minimal and symmetric
- **Early post-stimulus (0 to 0.1s)**: Auditory N1 component (removed by difference wave)
- **P300 window (0.3 to 0.6s)**: The characteristic P300 peak; **color intensifies** if P300 is present
- **Post-P300 (0.6 to 0.7s)**: Return toward baseline

**Electrode positions** (on the scalp maps):
- **Fz** (top-front): Frontal cortex — maximal in P3a (novelty orienting)
- **Cz** (top-center): Central midline — intermediate activity
- **Pz** (top-back, vertex): Parietal cortex — maximal in P3b (working memory)
- Other standard 10–20 positions are also shown for reference

**Interpreting P3a vs. P3b patterns**:
- **P3a (frontal-max)**: If red/intense activity is strongest at Fz around 300–400ms, indicates a novelty-orienting response
- **P3b (parietal-max)**: If red/intense activity is strongest at Pz around 300–500ms, indicates context closure and working memory updating (typical in healthy controls)
- **Mixed or absent**: If activity is distributed or minimal, may indicate attention deficits or stimulus insensitivity

**Standard conventions**:
- **Red** = positive voltage (depolarization, active neural firing)
- **Blue** = negative/baseline voltage (no P300 activity)
- **White/light** = near-zero voltage
- This diverging colormap is standard across the neuroscience literature and allows quick visual identification of spatial patterns.

#### ERP Image (Single-Trial Heatmap)

A single-trial heatmap PNG file (`{patient_id}_{date}_oddball_erp_image.png`) shows individual trial responses over time at electrode Pz (parietal). Each horizontal line represents one trial, with color indicating voltage amplitude. This visualization:

- **Detects outlier trials**: Single trials that deviate strongly from the mean (very red or blue)
- **Shows trial-to-trial variability**: Consistent vertical patterns indicate stable P300; noisy patterns suggest attention lapses or artifacts
- **Only generated if ≥3 rare epochs** are available (skipped for low-count sessions)
- **Complements the averaged ERP**: Averaging can hide high variability or split responses

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

**1. Difference Wave as Primary P300 Metric**
- Computes difference ERP (target − standard) to isolate the endogenous P300 component (Sutton et al., 1965)
- Removes auditory N1-P2 envelope and baseline drift that contaminates raw rare-ERP peak measurements
- `diff_*` columns are the canonical P300 metrics; `p300_*` columns are backward-compatible but secondary
- Requires sufficient standard epochs (logs warning if n_standard_epochs < 10)
- Falls back to rare-ERP measurement (legacy) if standard ERPs cannot be computed

**2. ENG-03 Epoch Reuse**
- Loads pre-computed artifact-cleaned 35s epochs from disk rather than re-running ICA each time
- Maps rare events into those epochs via Unix timestamps and extracts 900ms sub-epochs
- Also maps standard (frequent) events in parallel to enable difference-wave computation
- Events from ENG-03-dropped trials are silently excluded — no fallback to raw EEG
- Trade-off: some rare events may be lost due to trial-level PTP rejection, but all data used is artifact-cleaned

**3. Multi-Electrode P300 with Subtype Classification**
- Pz (parietal) is primary - usually strongest P300 (P3b: working memory update)
- Fz (frontal) indicates P3a pattern (novelty orienting, may lack P3b)
- Cz (central) as intermediate for mixed patterns
- Composite averaging across valid electrodes reduces sensitivity to single-electrode failures
- Case-insensitive electrode matching (`Pz` = `PZ` = `pz`)

**4. Session-Level Processing**
- Each date processed separately
- Handles multi-session patients naturally
- Each session → separate epoch/ERP files (rare, standard, diff)
- Features aggregated in master table with deduplication on (patient_id, date)

## Understanding Topomaps: Electrode Positions and P3a/P3b Discrimination

The topomap series is central to P300 analysis because it reveals the **spatial distribution of brain activity** — a key diagnostic feature. The electrode positions on the scalp topomaps are standard 10–20 positions, with **Fz, Cz, and Pz** being the three primary midline sites:

### Scalp Electrode Layout

The scalp is viewed from above in each topomap circular plot:
- **Top-front of circle** = Frontal cortex (Fz, F3, F4, etc.)
- **Top-center/back of circle** = Central and parietal cortex (Cz, Pz, P3, P4, etc.)
- **Sides of circle** = Temporal areas (T7, T8, etc.)

Electrode dots are overlaid on the circular scalp, with Fz, Cz, and Pz labeled or prominently positioned at top-front, top-center, and top-back respectively.

### P3a vs. P3b: Reading from Topomaps

**P3b (Parietal-Maximum, Typical Response)**
- Strongest red (positive) activity at **Pz** during 300–500ms window
- Reflects context closure and stimulus classification
- Associated with: working memory updating, stimulus relevance processing
- Expected in healthy controls attending to task

**P3a (Frontal-Maximum, Novelty Orienting)**
- Strongest red activity at **Fz** during 300–400ms window
- Reflects stimulus-driven attention shift (often before voluntary response)
- Associated with: novelty detection, stimulus-triggered reorienting
- Expected when a truly novel stimulus breaks attention
- May occur *without* a P3b if the stimulus is not task-relevant

**Mixed (Cz-Maximum)**
- Activity centered at **Cz** (frontal-parietal intermediate)
- Suggests transitional or competing processes
- Can indicate attention difficulties or bimodal response patterns

**Absent**
- Minimal or no red activity in 300–600ms window
- No clear P300 at any electrode
- May indicate inattention, drowsiness, or neurological dysfunction

### Clinical Interpretation Tips

1. **Always look at the pre-stimulus topomaps (−0.2 to 0s)**: Should be symmetric and low-amplitude. Asymmetry here suggests electrode contact or noise issues, not P300.

2. **Compare rare vs. standard conditions** (if plotting both): The rare condition should show stronger P300 activity, with the difference wave isolating the component by subtraction.

3. **Check consistency across time**: A robust P300 will show a clear peak around 300–500ms, then decline by 700ms. Noisy or multi-peaked patterns suggest poor trial quality or attention lapses.

4. **Combine with amplitude/latency measurements**: The topomap shows *where*; the `diff_amplitude_*` and `diff_latency_*` columns in `p300_features.parquet` show *how much* and *when*. Use both for complete assessment.

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
2. Group statistics (compare patient groups)
3. N200-P300 peak-to-peak measurement (N200 detection in standard ERP, P300 in diff ERP)
4. Parallel processing for batch mode
5. CSV/Excel export
6. Topographic maps with confidence intervals (currently static)
7. Correlation analysis between P3a/P3b patterns and clinical outcomes

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
