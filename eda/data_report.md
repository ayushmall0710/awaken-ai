# EEG Prognostic Data Pipeline & Characterization

This document summarizes the available data (delivered via OneDrive) and the planned pipeline for EEG-based prognostication in severe brain injury patients, drawing on the project goals outlined in the proposal.

## Project context
Severe brain injury patients can retain residual auditory and cognitive processing even without overt responses. The objective is to leverage EEG markers (language tracking, oddball/P300, command-following paradigms) to improve recovery prognostication in the acute window.

## Data streams in use
- **Trial/event logs (CSV)**: multiple versions of trial logs (e.g., `patient_df*` and `*_stimulus_results*`) capturing `patient_id`, `date`, `trial_type`, stimulus sequence (`sentences`), `start_time`, `end_time`, `duration`; some variants also contain `trial_index`, `paradigm`, `stimulus_details`, and `notes`.
- **Session metadata (CSV)**: brief visit dates and qualitative notes (e.g., `patient_history*`, `patient_notes*`).
- **Raw signals (external)**: EEG recordings (EDF) and audio stimuli (WAV) are available on OneDrive and need to be synced for signal-level analysis.

## Data volume (from current CSV deliveries)
- Largest trial log: `patient_df_043025.csv` with **1,120** rows.
- Other `patient_df` variants: **5–672** rows.
- Stimulus result files: typically **24–182** rows; legacy “old stimulus software” logs: **84** rows each.
- Trial-type mix (typical per session):
  - Language: ~72 trials
  - Control / loved_one_voice: ~50 trials each
  - Command blocks (left/right): ~3 each
  - Oddball/beep: ~4
- Durations (per trial type, from logs):
  - Language: ~15–16 seconds
  - Oddball/beep: ~30–34 seconds
  - Command blocks: ~200 seconds

## Where the data resides
- **Primary location**: OneDrive (CSV trial logs, metadata, EEG EDF, audio WAV).
- **Working copies**: pulled locally for analysis as needed; outputs should be written to an organized processed-data area with versioned schemas.

## Software & access
- Python with pandas for tabular EDA (Jupyter-friendly).
- Planned signal processing: MNE/NumPy/SciPy for EDF parsing, epoching, PSD/ITPC computation, and downstream modeling.

## Planned data products (if creating/augmenting)
- **Aligned EEG epochs** with event logs (language, command, oddball/beep paradigms).
- **Features**:
  - PSD and related bandpower metrics for command-following paradigms.
  - Language-tracking metrics (e.g., ITPC) for comprehension markers.
  - Behavioral/event tables with timing, stimulus labels, and QC flags.
- **Formats**: HDF5/Parquet for tabular features; NumPy arrays for signal-derived data; plots as PNG/HTML. Maintain a `processed` area with a data dictionary and schema versioning.

## Weak points / risks
- Raw EDF/WAV must be synced from OneDrive; without them, timing/artifact validation is blocked.
- Schema drift across `patient_df` variants (optional columns like `trial_index`, `paradigm`, `stimulus_details`, `notes`), requiring harmonization.
- Class imbalance: far fewer command trials than language; some sessions are small (e.g., 24 rows).
- Sparse clinical metadata: no demographics/outcomes in current CSVs.
- Potential duplication/overlap across trial log versions and possible time-sync drift between logs and signals.

## Next actions
1) Sync EDF/WAV from OneDrive and lock a unified schema for all trial logs.  
2) Reconcile duplicate/overlapping `patient_df` versions; deduplicate sessions.  
3) Align events to signals; add QC flags (artifacts, completeness, timing).  
4) Generate processed/feature tables (PSD for command paradigms; language-tracking markers) with a clear data dictionary and versioning.  
5) Document storage layout for processed outputs (processed/ features/ plots) to keep analyses reproducible.

