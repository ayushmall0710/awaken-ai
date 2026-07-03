# Architecture

This document explains how data moves through `awaken-ai`, from raw OneDrive
files to per-patient HTML reports, and where each part of `src/` fits in.

## Data flow

```
OneDrive (raw EEG + audio + stimulus logs)
        │  sync_data.sh
        ▼
data/EEG/*.csv, data/EEG/edf/*.EDF, data/Audio/**        ("local data root")
        │  awakenai unify-data          (src/data_processing/pipeline.py)
        ▼
data/processed/unified_stimulus_results.parquet          (all patients, all trials)
        │  UnifiedDataLoader / PatientData               (src/data_loading/)
        │
        │  awakenai setup <patient>
        ├─ Timestamp alignment          (src/data_processing/timestamp_aligner.py)
        │  data/processed/aligned_events/{patient_id}_events.parquet
        │
        └─ Artifact rejection (ICA)     (src/data_processing/artifact_rejection.py)
           data/processed/epochs/{patient_id}/{session_id}/{trial_type}-epo.fif
                │
                │  awakenai run <patient> [--pipeline ...]
                ▼
        Pipeline classes (src/pipelines/*.py, all subclass BasePipeline)
                │
                ├─ feature tables → data/processed/{erps,features}/...
                └─ HTML reports  → data/reports/{patient_id}/{session_id}/{pipeline_name}/
```

Every stage reads the output of the previous one from disk (Parquet/`.fif`
files under `data/processed/`), so pipelines never touch raw EDFs directly —
they consume the aligned events and cleaned epochs produced by setup.
`awakenai setup <patient>` runs both steps (and a third, clinical-record
sync) and is a hard prerequisite: `awakenai run` refuses to run a pipeline
for a patient/session that hasn't been set up (see `_check_setup` in
`src/cli/main.py`).

## `src/` layout

### `cli/`
The `awakenai` Typer app (`main.py`) and its subcommands:
- `commands/setup_cmd.py` — the guided `awakenai setup` wizard.
- `commands/inspect_cmd.py` — `list` / `info` / `count` read-only queries.
- `runners/*.py` — one module per pipeline (`oddball.py`, `language.py`,
  `command_following.py`, `command_following_claassen.py`, `qc_report.py`).
  Each runner wires a pipeline class to CLI args, prints a results table via
  `cli_utils.print_table`, and optionally renders an HTML report.
- `cli_utils.py` — shared helpers: `get_loader()`, `resolve_patients()`,
  `print_table()`.

### `data_loading/`
- `config.py` — all filesystem paths (single source of truth: `PROJECT_ROOT`,
  `LOCAL_DATA_ROOT`, `EPOCHS_DIR`, `REPORTS_DIR`, etc.) and shared electrode
  sets (`CLINICAL_20`, `LH_FOCUS_CHANNELS`, `RH_FOCUS_CHANNELS`). Override the
  project root with the `AWAKEN_PROJECT_ROOT` env var, and the OneDrive
  source with `ONEDRIVE_ROOT`.
- `unified_data_loader.py` — `UnifiedDataLoader`: loads the unified Parquet,
  provides cross-patient queries (`get_trials_by_type`, `get_trial_summary`),
  LRU-caches EDF loading, and loads the ENG-02/ENG-03 outputs
  (`load_aligned_events`, `load_clean_epochs`).
- `patient_data.py` — `PatientData`: single-patient view returned by
  `loader.get_patient(...)`, enriched with clinical metadata from
  `patient_records.json`.
- `inventory.py` — `DataInventory`, used by `sync_data.sh` to mirror OneDrive
  into the local `data/` tree.
- `create_stimulus_manifest.py`, `digitize_patient_records.py` — one-off /
  setup-time scripts for building the stimulus manifest and clinical records
  JSON.

### `data_processing/`
The "ENG-02/ENG-03" prerequisite stages, run by `awakenai setup`:
- `pipeline.py` — `unify_stimulus_data()`, the `awakenai unify-data` command:
  compiles and deduplicates all per-patient stimulus CSVs into the unified
  Parquet.
- `timestamp_aligner.py` — `TimestampAligner`: aligns each trial's Unix
  stimulus timestamps to the EDF's internal clock via DC-channel
  cross-correlation (language/command trials) or peak detection
  (oddball/beep trials). Output: `aligned_events/{patient_id}_events.parquet`.
- `artifact_rejection.py` — `ArtifactRejector`: runs ICA once per recording
  session (ICLabel classifier, with correlation-based fallback), then epochs
  every trial type into fixed-window `.fif` files.
- `erp_pipeline.py` — legacy standalone ERP/P300 implementation that
  `src/pipelines/p300_oddball.py` now supersedes for CLI use; still used by
  some tests.
- `qc_report.py` — `generate_qc_report()`, aggregates setup/pipeline QC
  metrics into a cross-patient HTML dashboard (`awakenai qc`).
- `normalization.py` — shared helpers for normalizing trial types and
  sentence/event field formats across the CSV ingestion path.

### `pipelines/`
The analysis pipelines proper. All inherit `base.BasePipeline`, which fixes
the control flow as a template method: `run()` → `load()` → `preprocess()` →
`analyze()`, plus a `generate_summary()` hook for a compact
classification/summary dict. `run()` first loads that patient's aligned
events (optionally filtered to one session) via
`loader.load_aligned_events()`.

- `p300_oddball.py` — `P300OddballPipeline`: loads ENG-03 35s oddball
  epochs, maps rare/standard beep events into 900ms P300 sub-epochs
  (-200 to +700ms), computes ERPs, quantifies P300 amplitude/latency at
  Fz/Cz/Pz (with per-electrode QC validation and a composite score), and a
  difference wave (rare − standard) with MMN detection. Writes three feature
  tables (clinical, per-electrode detail, mapping QC) as Parquet.
- `language_tracking.py` — `LanguageTrackingAnalysis`: isolates language
  trials, filters/downsamples/crops epochs, and computes Inter-Trial Phase
  Coherence (ITPC) at sentence/phrase/word rates using both a DFT method and
  Morlet wavelets. Selects an optimal electrode focus via spatial cluster
  permutation, in addition to fixed clinical/LH/RH focuses, and computes
  permutation p-values and lateralization indices. See
  [`docs/language_tracking.md`](language_tracking.md) for the full
  methodology.
- `command_following.py` — `CommandFollowingAnalysis`: detects Event-Related
  Desynchronization (ERD) in Alpha/Beta bands at C3/C4/Cz during motor
  imagery ("keep"/"stop" command pairs), with paired t-tests and a mixed
  effects model.
- `command_following_claassen.py` — SVM-based command-following
  classification, replicating the Claassen et al. methodology.

### `viz/` and `reports/`
- `viz/*.py` — Matplotlib figure builders per pipeline (ERP waveforms,
  topomaps/animated topomaps, ITPC heatmaps, ERD plots). Pipelines call
  these during `analyze()` and save the resulting PNGs/GIFs.
- `reports/*.py` — HTML report builders per pipeline, plus
  `style_utils.py` for shared CSS/layout and `stitch_and_save()`, which
  combines multiple per-session HTML fragments into one patient-level report
  (used when `--report` is passed to `awakenai run` and a patient has more
  than one session).

### `utils/`
- `signal_processing.py` — band power (Welch PSD), channel name
  normalization, non-EEG channel exclusion, spatial cluster permutation
  (`select_optimal_channels`), shared across pipelines.
- `time_utils.py` — Unix ⇄ EDF-internal-clock conversions and timezone
  offset detection, used by the timestamp aligner and command-following
  pipeline.

## Output layout

All pipeline outputs live under `data/processed/` and `data/reports/`
(paths defined in `data_loading/config.py`):

```
data/processed/
  aligned_events/{patient_id}_events.parquet   # ENG-02
  epochs/{patient_id}/{session_id}/{trial_type}-epo.fif   # ENG-03
  qc/{patient_id}/{session_id}/eng03_qc.parquet
  erps/, features/, plots/erp/, oddball_qc/    # oddball pipeline outputs
  unified_stimulus_results.parquet
  patient_records.json

data/reports/{patient_id}/{session_id}/{pipeline_name}/   # per-session reports
data/reports/{patient_id}/combined/{pipeline_name}/       # multi-session summary
```

## Tests

`tests/` mirrors this structure closely — e.g. `test_p300_oddball.py`,
`test_language_tracking.py`, `test_artifact_rejection.py`,
`test_timestamp_aligner.py`, `test_unified_data_loader.py`. Run with
`pytest` (configured in `pyproject.toml`, `testpaths = ["tests"]`).
