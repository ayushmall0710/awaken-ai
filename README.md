# awaken-ai

## Installation

1. **Install Python dependencies:**

   ```bash
   # For `dev` - recommended full installation
   pip install -e ".[all]"
   
   # Or minimal installation (core dependencies only)
   pip install -e .

   # Check `pyproject.toml` for more details.
   ```

2. **Install ffmpeg:**
   - **macOS:** `brew install ffmpeg`
   - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Data Sync

Sync EEG data files from OneDrive to local directory.

<img width="800" height="433" alt="onedrive-sync-button-screenshot" src="https://github.com/user-attachments/assets/ed1b0ec4-a6a9-4e4b-a568-83f22ff9031b" />

**Default macOS path:**

```text
~/Library/CloudStorage/OneDrive-SharedLibraries-UW/Peter Schwab - EEG Project Data
```

**Custom path:**

```bash
export ONEDRIVE_ROOT="/path/to/onedrive"
```

**Sync files:**

```bash
./sync_data.sh                    # Interactive sync
./sync_data.sh --sync             # Direct sync
./sync_data.sh --sync --overwrite  # Overwrite existing
./sync_data.sh --sync --overwrite  # Overwrite existing
```

## CLI Usage

Installing the package registers the `awakenai` command (see `[project.scripts]` in `pyproject.toml`). It orchestrates the full workflow: syncing raw stimulus logs into a unified dataset, preparing a patient's EEG data, and running the analysis pipelines.

By default the loader reads `data/processed/unified_stimulus_results.parquet` under the project root (override with `AWAKEN_PROJECT_ROOT`). See [`docs/architecture.md`](docs/architecture.md) for how data flows between these steps.

**1. Build the unified dataset** (run once, or after syncing new data):

```bash
awakenai unify-data
```

**2. Explore what's available:**

```bash
awakenai list patients                       # All patient IDs
awakenai list sessions CON008                 # Sessions for one patient
awakenai list trials CON008 --session 2025-08-14 --type language
awakenai info patient CON008                  # Clinical + trial summary
awakenai info session CON008 2025-08-14
```

**3. Set up a patient** (timestamp alignment + ICA artifact rejection — required before any pipeline can run):

```bash
awakenai setup CON008                         # Guided, prompts per step
awakenai setup CON008 CON009 --force          # Non-interactive, all patients
awakenai setup --all
```

**4. Run analysis pipelines:**

```bash
awakenai run CON008                                   # Auto-detects applicable pipelines from trial types
awakenai run CON008 --pipeline oddball --report       # Force one pipeline, generate HTML report
awakenai run CON008 CON009 --session 2025-08-14
awakenai run --all --pipeline language
awakenai run CON008 --setup                           # Run setup first, then the pipeline(s)
```

Available `--pipeline` values: `command-following`, `command-following-svm`, `language`, `oddball`. Omit `--pipeline` to auto-dispatch based on the trial types present (`left_command`/`right_command` → command-following, `language` → language, `oddball` → oddball).

**5. Generate a QC report** across patients/sessions:

```bash
awakenai qc                                   # All patients, all sessions
awakenai qc -p CON008 -s 2025-08-14 -o /tmp/reports
```

Run `awakenai --help` or `awakenai <command> --help` for the full option list.

## Development

**Code Quality:**

This project uses `ruff` for both linting and formatting.
CI **checks** for style issues but does not fix them automatically.

**Before committing, run:**

```bash
ruff check --fix . && ruff format .
```

This ensures your code passes the CI format check. You can also install `Ruff` extension to auto-format the code everytime you save the file.


## Data Preprocessing

![WhatsApp Image 2026-03-12 at 13 59 41](https://github.com/user-attachments/assets/edff523c-2d2a-49bc-928c-1ff1f1d85796)
