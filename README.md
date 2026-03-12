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
