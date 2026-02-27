# CLI Orchestrator: Development Log & Status

# CLI Orchestrator: Development Log & Status

The `awakenai` Command Line Interface (CLI) is a robust gateway to the Awaken AI analysis pipelines. It ensures raw data is paired, cleaned, and correctly formatted before executing analytical pipelines.

## 🚀 Quick Highlights

- **Interactive Data Setup**: Guided wizard to align timestamps & reject artifacts before running pipelines.
- **Strict Guardrails**: Refuses to run pipelines on unprepared data, eliminating silent failures.
- **Clinical & Data Inspection**: Explore patient notes, visits, sessions, and available trials from the terminal (like pandas `value_counts` but for clinical data).
- **Fast & Scalable**: Lazy loading ensures instant `--help` response times.

## 💻 Sample Commands

**Setup & Execution (`setup`, `run`)**

```bash
awakenai setup CON008                   # Guided interactive setup for one patient
awakenai setup CON008 -f                # Force setup (auto-yes to all steps)
awakenai setup -a                    # Run interactive setup for ALL patients
awakenai setup run CON008               # Setup, then immediately run applicable pipelines

awakenai run CON008                     # Run all applicable pipelines for a patient
awakenai run CON008 -p command-following # Run only a specific pipeline
awakenai run CON008 -s 2025-08-14       # Run pipelines for a specific session
awakenai run --all                      # Run all pipelines for every available patient
```

**Data Inspection (`list`, `count`)**

```bash
awakenai list patients                  # List all patient IDs in the database
awakenai list sessions CON008           # List all recorded session dates for a patient
awakenai list trials CON008             # Show all trials (type, date)
awakenai list trials CON008 --detailed  # Show exact start/end timestamps and duration
awakenai list trials CON008 -s 2025-08-14 # Filter trials by session
awakenai list trials CON008 -t oddball  # Filter trials by specific type
awakenai count trials CON008            # Quick summary of trial counts grouped by type
```

**Clinical Information (`info`)**

```bash
awakenai info patient CON008            # View patient metadata, visit history, and notes
awakenai info session CON008 2025-08-14 # View trial summary for a single session
awakenai info trial CON008 0            # View exact timestamps and sentences for a specific trial
awakenai info trial-types               # View built-in documentation mapping trial types to pipelines
```

**Global Flags**

```bash
awakenai --help                         # Show main help menu
awakenai -V                             # Show CLI version
awakenai -v run CON008                  # Enable verbose debug logging (MNE, data loading)
```

## 📁 CLI Architecture

```text
src/cli/
├── main.py                  # Entry point (Typer 'app' routing & run_cmd pipeline dispatcher)
├── logging_config.py        # Mutes underlying libs (MNE) unless `-v / --verbose` is passed
├── utils.py                 # Shared helpers (resolve_patients, print_table layout)
├── commands/
│   ├── setup_cmd.py         # The 'setup' wizard & prerequisite checkers
│   └── inspect_cmd.py       # The 'list', 'count', and 'info' commands
└── runners/
    ├── command_following.py # Eng-04 dispatcher
    ├── language.py          # Eng-03 dispatcher
    └── oddball.py           # Eng-02 dispatcher
```

## 🛠️ Features Developed

1. **Interactive Setup Wizard (`awakenai setup`)**
   - Wired the setup command to actual processing classes: `TimestampAligner` and `ArtifactRejector`.
   - The wizard detects whether steps have already been completed for a patient and changes its prompts dynamically (`Status: Complete` defaults to No for running again; `Status: Not complete` defaults to Yes).
   - Added a patient summary at the start of setup, showing available sessions and pipelines so users know what data they are working with.
   - Added a headless execution mode via `-f` / `--force`.

2. **Pipeline Dispatching framework (`awakenai run`)**
   - Built a robust execution engine that strictly guards against running pipelines on unprepared data. If a patient hasn't completed setup, a clear error panel is displayed explaining exactly what commands to run.
   - Standardized dispatching so adding new pipelines in the future requires only a 1-line addition to the runner mapping.

3. **Data Inspection Tools (`awakenai list`, `count` and `info`)**
   - Created tools to explore the dataset without writing Python: `list patients`, `list sessions`, `list trials`.
   - `awakenai count trials` provides a rapid summary of data availability (equivalent to `.value_counts()`).
   - `info patient` summarizes clinical history, listing the most recent visit dates, total visit count, and clinical notes directly from the CLI.
   - Designed clean, dynamically-sizing tables to display this data.
   - Included `info trial-types` as built-in documentation bridging clinical trial terminology into software pipeline names. _Can add more documentation later._

## Design Decisions

1. **Lazy Loading Imports**
   - We deferred importing heavyweight modules inside CLI functions to keep the CLI highly responsive (boots instantly for `--help`).
2. **Setup verification uses disk checks**
   - `setup` relies on verifying actual output existence, ensuring the CLI is perfectly synced with the filesystem truth.
3. **No-op default logging**
   - The logging level was set to `WARNING` by default to ensure CLI output stays clear. Passing `-v` activates **debug level tracking** from `mne` and the loader logic.

## What is Pending

1. `trial_id` and `session_id` metadata columns need to be formally generated and added to the core DataFrames (they are currently returning placeholder columns on-the-fly during `list trials`).
2. The `digitize_patient_records` logic (syncing source spreadsheets to JSON) needs to be integrated directly into the `setup` command so the database self-updates automatically when new data is pulled.

## Future Work

- Integrate the visual representations (Event Related Potentials, topomaps) and summary tables generated by the pipelines directly into a CLI reporting tool once the Language ([PR #44](https://github.com/ayushmall0710/awaken-ai/pull/44)) and Oddball ([PR #37](https://github.com/ayushmall0710/awaken-ai/pull/37)) pipelines are merged into main.
- Implement optimized parallel running of patients/pipelines for faster batch processing across pipelines.
