"""
Configuration file for data loading module.

Update these paths to match your system setup.
"""

import os
from pathlib import Path

# Project root directory
# Since the package might be installed in site-packages, we default to the current working directory,
# or allow overriding via the AWAKEN_PROJECT_ROOT environment variable.
PROJECT_ROOT = Path(os.environ.get("AWAKEN_PROJECT_ROOT", Path.cwd()))

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

# Preprocessing (ENG-03) outputs
EPOCHS_DIR = PROCESSED_DATA_DIR / "epochs"
QC_DIR = PROCESSED_DATA_DIR / "qc"
# ERP Pipeline output directories (ENG-02b)
ERPS_DIR = PROCESSED_DATA_DIR / "erps"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
ERP_PLOTS_DIR = PROCESSED_DATA_DIR / "plots" / "erp"
QC_REPORTS_DIR = PROCESSED_DATA_DIR / "oddball_qc"

# Analysis (ENG-06) outputs
REPORTS_DIR = PROCESSED_DATA_DIR / "reports"

# Audio directories (relative to LOCAL_DATA_ROOT)
AUDIO_DIR = LOCAL_DATA_ROOT / "Audio"
SENTENCES_DIR = AUDIO_DIR / "sentences"
PROMPTS_DIR = AUDIO_DIR / "prompts"

# Audio files
COMMAND_AUDIO_FILE = "motorcommandprompt.wav"

# Unified Parquet File Path
UNIFIED_PARQUET_PATH = PROCESSED_DATA_DIR / "unified_stimulus_results.parquet"

# Patient clinical records JSON
PATIENT_RECORDS_PATH = PROCESSED_DATA_DIR / "patient_records.json"

# --- Sensor Configurations ---

# Standard 10-20 system electrodes generally available across datasets
# Reference: ENG-05 methodology documentation (docs/language_tracking.md)
CLINICAL_20 = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "Cz",
    "C3",
    "C4",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "O1",
    "O2",
    "T7",
    "T8",
]

# Left Hemisphere channels for Language Tracking paradigm
# Reference: Sokoliuk et al. 2021 (https://doi.org/10.1523/JNEUROSCI.2260-20.2021)
# Selected for spatial coverage over typical left-lateralized language areas.
LH_FOCUS_CHANNELS = [
    "F7",
    "T7",
    "Fp1",  # Inferior/Lateral/Frontal Left
    "F3",
    "C3",
    "P3",  # Superior/Medial Left
]

# Right Hemisphere counterpart channels for Language Tracking
# Included for lateralization analysis and symmetry.
RH_FOCUS_CHANNELS = [
    "F8",
    "T8",
    "Fp2",  # Inferior/Lateral/Frontal Right
    "F4",
    "C4",
    "P4",  # Superior/Medial Right
]
