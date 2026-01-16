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
    str(
        Path.home()
        / "Library"
        / "CloudStorage"
        / "OneDrive-SharedLibraries-UW"
        / "Peter Schwab - EEG Project Data"
    ),
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
