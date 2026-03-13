# ENG-02b: ERP/Oddball Pipeline

**Status:** ✅ Complete  
**Assignee:** Arnav Dixit  
**Dependencies:** ENG-02 (Timestamp Alignment), ENG-03 (Artifact Rejection), ENG-01 (Data Loader)

---

## Overview

ENG-02b is the P300 oddball analysis pipeline for auditory oddball trials. It operates on:

- ENG-02 aligned events
- ENG-03 cleaned 35-second oddball trial epochs
- pre-generated visualization artifacts
- parquet-based report inputs

The current CLI-integrated implementation is `P300OddballPipeline` in `src/pipelines/p300_oddball.py`. It is invoked through:

```bash
awakenai run <patient_id> --pipeline oddball
```

The oddball report is rendered by `OddballQCReport` in `src/reports/oddball_qc_report.py`.

`src/data_processing/erp_pipeline.py` is still present as legacy/reference code, but it is not the primary integrated oddball path used by the current CLI runner.

## Current Architecture

### Inputs

1. **Aligned events from ENG-02**
   - `data/processed/aligned_events/{patient_id}_events.parquet`
   - Contains per-trial metadata plus aligned per-stimulus timestamps inside the `sentences` field

2. **Cleaned ENG-03 oddball epochs**
   - `data/processed/epochs/{patient_id}/{session_id}/oddball-epo.fif`
   - Each epoch is a cleaned 35-second oddball trial window with metadata including `start_time_unix` and `end_time_unix`

3. **Report assets**
   - oddball feature parquets in `data/processed/features/`
   - plot files in `data/processed/plots/erp/`

### Pipeline ownership

The current operational flow is:

```text
awakenai run --pipeline oddball
    -> src/cli/runners/oddball.py
    -> src/pipelines/p300_oddball.py
    -> src/reports/oddball_qc_report.py
```

### ENG-03 dependency

ENG-03 must run first. The oddball pipeline does not reprocess raw EEG from scratch. It loads ENG-03-cleaned trial epochs and slices shorter event-locked sub-epochs from them.

If ENG-03 outputs are missing, oddball processing fails with a clear error.

## Data Flow

```text
Aligned Events (ENG-02)                Clean Oddball Epochs (ENG-03)
data/processed/aligned_events/         data/processed/epochs/{patient}/{session}/oddball-epo.fif
        |                                              |
        | extract rare + standard event timestamps     | read metadata start/end times
        |                                              |
        +---------------- map events into 35s trial windows ----------------+
                                      |
                                      | extract 900 ms sub-epochs
                                      v
                       Rare sub-epochs + Standard sub-epochs
                                      |
                       average -> rare ERP / standard ERP
                                      |
                       difference ERP = rare - standard
                                      |
                 P300 quantification + MMN extraction + Welch support test
                                      |
             save ERPs + plots + three feature tables + HTML oddball report
```

## Core Analysis

### Event extraction

The pipeline reads aligned oddball trials and extracts:

- **rare events** from `sentences` where `event == "rare"`
- **standard events** from `sentences` where `event in {"standard", "frequent"}`

Each event is expected to include an aligned `event_start` timestamp in Unix time.

### Trial-window mapping

Each rare or standard event is mapped into one surviving ENG-03 35-second trial window using:

- `start_time_unix`
- `end_time_unix`

Rules:

- event must map to exactly one surviving ENG-03 trial
- sub-epoch boundaries must remain inside the 35-second trial
- unmapped, duplicate, or boundary-clipped events are excluded

### Sub-epoch extraction

For each mapped event, the pipeline extracts a short event-locked epoch using:

- `tmin = -0.2`
- `tmax = 0.7`
- baseline correction `(None, 0)`

These 900 ms epochs are not saved independently. They are re-sliced from the 35-second ENG-03 trials when needed.

### ERP computation

The pipeline computes:

- rare ERP
- standard ERP
- difference ERP (`rare - standard`)

### P300 quantification

By default, oddball quantification focuses on:

- `Pz`
- `Cz`
- `Fz`

For each electrode, the pipeline measures:

- rare ERP P300 peak amplitude
- rare ERP P300 peak latency
- difference-wave P300 peak amplitude
- difference-wave P300 peak latency

It also computes:

- composite P300 metrics across valid midline electrodes
- best electrode
- subtype classification (`P3b`, `P3a`, `mixed`, `absent`)

### Welch t-test support

The report includes a Welch t-test for rare vs standard support. This test:

- is computed at `Pz`
- uses the `300-600 ms` window
- compares **single-trial mean amplitudes** from rare vs standard epochs

Stored fields:

- `p300_p_value`
- `p300_t_stat`
- `p300_n_rare`
- `p300_n_standard`

Important: this Welch test is a **supporting measure of rare-standard separation**. It is **not** the sole rule for whether a P300-like morphology is present.

### MMN implementation

Mismatch negativity (MMN) is computed from the **difference ERP** (`rare - standard`).

MMN rules:

- peak search window: `100-250 ms`
- peak type: most negative value in the window
- extracted per electrode
- report highlights `MMN at Fz`

In the report, MMN at Fz is considered reliable only if:

- amplitude is negative
- latency is within `100-250 ms`

MMN is a supporting descriptor, not the primary P300 decision metric.

## Configuration

Key analysis windows used by the current integrated path:

```python
ERP_CONFIG = {
    "tmin": -0.2,
    "tmax": 0.7,
    "baseline": (None, 0),
    "p300_window": (0.3, 0.6),
    "mmn_window": (0.100, 0.250),
    "min_epochs": 2,
    "midline_electrodes": ["Pz", "Cz", "Fz"],
}
```

## Output Artifacts

### ERP files

Saved under `data/processed/erps/`:

- `{patient_id}_{session_id}_oddball-ave.fif`
- `{patient_id}_{session_id}_oddball_standard-ave.fif`
- `{patient_id}_{session_id}_oddball_diff-ave.fif`

### Feature tables

Saved under `data/processed/features/`:

- `p300_oddball_clinical.parquet`
- `p300_oddball_electrode_detail.parquet`
- `p300_oddball_mapping_qc.parquet`

### Plot files

Saved under `data/processed/plots/erp/`:

- `{patient_id}_{session_id}_oddball_erp.png`
- `{patient_id}_{session_id}_oddball_erp_image.png`
- `{patient_id}_{session_id}_oddball_topomap.png`
- `{patient_id}_{session_id}_oddball_topomap.gif`

### HTML report

Saved under:

- `data/reports/{patient_id}/{session_id}/oddball/{timestamp}/oddball_qc.html`

The oddball CLI runner writes one timestamped HTML report per patient-session.

Inside the HTML report, figures are intentionally split into:

- **main plots**: waveform-centric figures used for primary temporal interpretation
- **secondary supporting images**: figures used to add spatial or trial-consistency context

Report hierarchy:

- if dedicated `p300` and/or `mmn` plots are present, they are the **main plots**
- otherwise the combined ERP overview figure (`..._oddball_erp.png`) is the **main plot**
- topomap and ERP-image assets are always treated as **secondary supporting images**

## Feature Table Semantics

### Clinical table

One row per patient-session. Key fields include:

- `patient_id`
- `session_id`
- `session_date`
- `n_rare_epochs`
- `n_standard_epochs`
- `baseline_std_uV`
- `p300_rare_amplitude_Pz_uV`
- `p300_rare_latency_Pz_ms`
- `p300_diff_amplitude_Pz_uV`
- `p300_diff_latency_Pz_ms`
- `diff_mmn_amplitude_Fz_uV`
- `diff_mmn_latency_Fz_ms`
- `p300_best_electrode`
- `p300_subtype`
- `p300_amplitude_uV`
- `p300_latency_ms`
- `p300_n_valid_electrodes`
- `qc_notes`
- `qc_pass`
- `p300_p_value`
- `p300_t_stat`
- `p300_n_rare`
- `p300_n_standard`

### Electrode detail table

Three rows per session by default: `Fz`, `Cz`, `Pz`.

Key fields:

- `electrode`
- `p300_amplitude_uV`
- `p300_latency_ms`
- `is_valid`
- `flagged_reason`
- `diff_amplitude_uV`
- `diff_latency_ms`
- `diff_mmn_amplitude_uV`
- `diff_mmn_latency_ms`

### Mapping/QC table

One row per session containing trial mapping diagnostics:

- `n_rare_events_candidate`
- `n_rare_mapped`
- `n_rare_unmapped`
- `n_rare_boundary_clipped`
- `rare_mapping_rate`
- `n_standard_events_candidate`
- `n_standard_mapped`
- `processing_timestamp`
- `pipeline_version`

## Report Interpretation

The oddball report is now **morphology-first** and **p-value-second**. The first row of cards is meant to answer:

1. Did we observe a plausible P300-like candidate at Pz?
2. How confident are we in that interpretation?
3. Did rare and standard trials separate statistically?
4. Was the underlying data quality good enough to trust the summary?

The report must not equate `not significant` with `no P300-like morphology`.

The report now includes:

- a visible **Confidence Interpretation** section that explains the confidence labels and the key thresholds used to read them
- a collapsible **Legend and metric definitions** section for the remaining reference definitions

The confidence section is intentionally visible by default so readers do not need to expand the legend to understand the primary interpretation.

### Row 1 cards

1. **P300 Candidate at Pz**
   - reports the rare-only Pz peak if it is positive and in the `300-600 ms` window
   - otherwise reports that there is no reliable candidate

2. **Confidence**
   - summarizes morphology, rare-count sufficiency, SNR, difference-wave support, and Welch support
   - uses the labels:
     - `Detected`
     - `Low-confidence detected`
     - `No reliable P300 detected`

3. **Rare vs Standard Support**
   - reports the Welch t-test support label:
     - `Separated`
     - `Trend only`
     - `Not separated`
     - `Unavailable`

4. **Data Quality**
   - summarizes rare-trial count and P300 signal/noise

### Row 2 cards

1. **Rare Epochs**
2. **Standard Epochs**
3. **MMN at Fz**
4. **Topography**

## Confidence Semantics

### Visible confidence interpretation in the report

The HTML report includes a dedicated **Confidence Interpretation** block above the collapsible legend. That block explains:

- confidence combines Pz morphology, rare-trial count, signal-to-noise, rare-vs-standard Welch support, and difference-wave support
- what `Detected`, `Low-confidence detected`, and `No reliable P300 detected` mean
- that a non-significant Welch test alone does not rule out P300-like morphology

The block also includes a compact **Key thresholds** list so readers can interpret confidence without opening the full legend.

### Candidate window

P300 candidate at Pz:

- amplitude must be positive
- latency must be inside `300-600 ms`

### Rare-count tiers

- `good`: `>= 20`
- `borderline`: `10-19`
- `poor`: `< 10`

### SNR tiers

Signal/noise is computed as:

```text
rare P300 amplitude at Pz / baseline sigma
```

Thresholds:

- `good`: `>= 2.0`
- `borderline`: `1.25-1.99`
- `poor`: `< 1.25`

### Welch support tiers

- `supportive`: `p < 0.05`
- `weak`: `0.05 <= p < 0.20`
- `not_supported`: `p >= 0.20`
- `unavailable`: missing p-value or insufficient trials for the test

### Key thresholds shown directly in the report

The visible **Confidence Interpretation** section should show these thresholds exactly:

- `P300 candidate window: 300-600 ms`
- `MMN validity window: 100-250 ms`
- `Rare-trial count: >=20 good, 10-19 borderline, <10 poor`
- `Signal-to-noise: >=2.0 good, 1.25-1.99 borderline, <1.25 poor`
- `Welch support: p<0.05 supportive, 0.05-0.19 weak, >=0.20 not supportive`

### Confidence labels

#### `Detected`

Used when:

- valid P300 candidate exists at Pz
- difference-wave support is present
- rare-count tier is `good`
- SNR is not poor
- Welch support is at least weak
- overall quality is not poor

#### `Low-confidence detected`

Used when:

- a valid P300 candidate exists at Pz
- overall quality is not poor
- but one or more support signals are borderline, unavailable, or unsupportive

Typical reasons:

- low rare-trial count
- borderline signal/noise
- non-significant rare-standard contrast
- missing or weak difference-wave support

#### `No reliable P300 detected`

Used when:

- no usable P300 candidate exists at Pz
- Pz metrics are missing
- latency falls outside `300-600 ms`
- or data quality is poor enough that the candidate should not be trusted

## Visualization

### ERP figure

The current integrated visualizer produces a **4-panel** ERP figure when rare, standard, and difference ERPs are all available:

1. rare-only butterfly plot
2. rare vs standard midline overlay
3. rare-only midline P300 panel
4. difference-wave panel

If standard ERP or difference ERP is unavailable, the visualizer falls back to a smaller legacy layout instead of the full 4-panel figure.

This ERP figure is the **default main plot** for the current integrated pipeline. It is the primary waveform visualization used by the report whenever dedicated `p300` / `mmn` focus plots are not available.

### ERP image

The pipeline can generate a single-trial ERP image at `Pz`:

- saved as `{patient_id}_{session_id}_oddball_erp_image.png`
- only produced when at least 3 rare epochs are available
- each row represents one rare trial
- the image emphasizes trial-to-trial consistency versus variability at `Pz`
- the bottom summary trace provides the average response used to visually cross-check the ERP

This is a **secondary supporting image** in the report. It is used to assess single-trial consistency, not to replace the main waveform plot.

### Topomap PNG

The pipeline generates a topomap PNG from the **difference ERP**:

- saved as `{patient_id}_{session_id}_oddball_topomap.png`
- uses a static series of scalp snapshots at 100 ms intervals
- is intended to summarize spatial evolution of the difference wave across the epoch

This is a **secondary supporting image** in the report. It adds spatial context to the waveform interpretation.

### Topomap GIF

The pipeline also generates an animated topomap GIF when a difference ERP is available:

- saved as `{patient_id}_{session_id}_oddball_topomap.gif`
- generated from successive difference-wave scalp maps
- uses 50 ms frame spacing from roughly `-100 ms` to `650 ms`
- is intended to show the temporal progression of spatial activity more smoothly than the static PNG

This is also a **secondary supporting image** in the report. It provides dynamic spatial context and does not replace the main waveform plot.

### Dedicated P300 / MMN focus plots

The report plot resolver can also consume optional dedicated focus plots:

- `p300` plot: a focused view of the P300 portion of the waveform, typically centered on parietal interpretation at `Pz`
- `mmn` plot: a focused view of the MMN portion of the difference waveform, typically centered on frontal interpretation at `Fz`

These are **optional / legacy-compatible** assets. The current integrated oddball pipeline does **not** generate dedicated `p300` and `mmn` PNGs by default; it generates the combined ERP figure plus topomap and ERP-image artifacts. If dedicated `p300` or `mmn` files exist from earlier workflows, the report will display them.

If these dedicated focus plots are present, they become the **main report plots** because they provide the most task-specific waveform views for P300 and MMN interpretation.

### Report plot fallback behavior

The report can resolve these plot keys:

- `p300`
- `mmn`
- `erp`
- `topomap`
- `erp_image`

Behavior:

- if dedicated `p300` / `mmn` assets are present, the report shows them directly
- otherwise it falls back to the combined ERP figure
- if a topomap GIF exists, the report can embed that GIF
- otherwise it uses the topomap PNG
- ERP image is shown only when the file exists

This allows the report to work with both legacy plot artifacts and the current integrated plot set.

Explicit report figure hierarchy:

- **Primary / main plot area**: dedicated `p300` and `mmn` focus plots when available; otherwise the combined ERP overview
- **Secondary / supporting image area**: topomap and ERP image

The same hierarchy applies to explanatory text:

- the visible **Confidence Interpretation** section is the main explanation for confidence labels
- the collapsible legend is secondary reference material

## Scientific Interpretation Notes

### Primary user-facing interpretation

The report centers on:

- `P300 Candidate at Pz`
- `Confidence`

Difference-wave metrics, MMN, and Welch support remain scientifically useful, but the report no longer treats any single numeric field as the whole decision rule.

### What the Welch test does not mean

A non-significant Welch t-test does **not** automatically imply:

- no P300-like morphology
- no visible candidate peak
- no potentially meaningful difference-wave structure

It means that, within the current single-trial window-mean comparison at Pz, the rare and standard conditions were not reliably separated.

## Usage

### Prerequisites

1. Run ENG-02 so aligned oddball events exist.
2. Run ENG-03 so cleaned 35-second oddball epochs exist.

### Primary command

```bash
awakenai run CON008 --pipeline oddball -r
```

This:

- runs the oddball pipeline for the selected patient/session(s)
- updates the oddball feature parquets
- saves ERP and plot artifacts
- generates timestamped HTML oddball report(s)

### Notes

- Custom electrode analysis remains available through the oddball pipeline, but the report interpretation described here is designed around the default midline P300 path.
- Missing channels or low epoch counts may still produce partial outputs; the report should degrade gracefully rather than fail.
