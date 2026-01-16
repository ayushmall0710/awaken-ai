# awaken-ai

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
```
