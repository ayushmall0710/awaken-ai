"""
Configuration file for data loading module.

Update these paths to match your system setup.
"""

import os
from pathlib import Path

# Project root directory (adjust if needed)
PROJECT_ROOT = Path(__file__).parent.parent.parent

ONEDRIVE_ROOT = os.environ.get(
    "ONEDRIVE_ROOT",
    str(Path.home() / "Library" / "CloudStorage" / "OneDrive-SharedLibraries-UW" / "Peter Schwab - EEG Project Data"),
    # If your OneDrive is synced to a different location, update this path or export
    # the environment variable
    # Example:
    # ONEDRIVE_ROOT=/path/to/your/onedrive
    # export ONEDRIVE_ROOT
)

# Local data directory (relative to project root)
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"

# Logging configuration
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "data_inventory.log"

# Data subdirectories
EEG_DATA_DIR = LOCAL_DATA_ROOT / "EEG"
PROCESSED_DATA_DIR = LOCAL_DATA_ROOT / "processed"
ALIGNED_EVENTS_DIR = PROCESSED_DATA_DIR / "aligned_events"

# ERP Pipeline output directories (ENG-02b)
EPOCHS_DIR = PROCESSED_DATA_DIR / "epochs"
ERPS_DIR = PROCESSED_DATA_DIR / "erps"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
ERP_PLOTS_DIR = PROCESSED_DATA_DIR / "plots" / "erp"
QC_REPORTS_DIR = PROCESSED_DATA_DIR / "qc"

# Audio directories (relative to LOCAL_DATA_ROOT)
AUDIO_DIR = LOCAL_DATA_ROOT / "Audio"
SENTENCES_DIR = AUDIO_DIR / "sentences"
PROMPTS_DIR = AUDIO_DIR / "prompts"

# Audio files
COMMAND_AUDIO_FILE = "motorcommandprompt.wav"

# Unified Parquet File Path
UNIFIED_PARQUET_PATH = EEG_DATA_DIR / "unified_stimulus_results.parquet"
