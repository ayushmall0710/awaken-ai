# awaken-ai

EEG Prognostic Data Pipeline for the Awaken AI Capstone Project.

## Installation

### Option 1: Install dependencies only (minimal)

```bash
pip install -r requirements.txt
```

### Option 2: Install as package (recommended for development)

```bash
pip install -e .
```

This allows you to import modules directly:
```python
from data_loading import EEGDataLoader, TimestampAligner
```

### Additional Requirements

**Install ffmpeg:**
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
```

## Timestamp Alignment (ENG-02)

The project includes timestamp alignment functionality to synchronize CSV Unix timestamps with EDF internal clocks using the DC audio channel. See [`docs/TIMESTAMP_ALIGNMENT.md`](docs/TIMESTAMP_ALIGNMENT.md) for details.

### Quick Start

```bash
# Run the demo script
python examples/timestamp_alignment_demo.py \
    --edf path/to/CON008.EDF \
    --csv path/to/CON008_stimulus_results.csv \
    --trial-type oddball
```

### Usage in Code

```python
from data_loading import EEGDataLoader, TimestampAligner

# Load EDF and CSV data
loader = EEGDataLoader(
    edf_path='path/to/CON008.EDF',
    csv_path='path/to/CON008_stimulus_results.csv'
)

# Initialize timestamp aligner
aligner = TimestampAligner(eeg_loader=loader)

# Detect stimulus onsets from DC channel
dc_data, dc_times = aligner.extract_dc_channel()
peak_times, peak_values = aligner.detect_stimulus_onsets(dc_data, dc_times)

# Convert to Unix timestamps
peak_times_unix = aligner.edf_time_to_unix(peak_times)
```

## Testing

Run unit tests:

```bash
python tests/test_timestamp_alignment.py
```
