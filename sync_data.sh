#!/bin/bash
# sync_data.sh - Script to verify and sync EEG data from OneDrive

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to project root
cd "$SCRIPT_DIR"

# Run the inventory script with all arguments passed through
python -m src.data_loading.inventory "$@"

