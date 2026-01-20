# DAT-01: Inventory & Sync Script

**Status:** Completed
**Category:** Infrastructure / Data Management
**Date:** January 15, 2026

---

## 1. Completion Summary

**Successfully Implemented:**
- `data_inventory.py`: A flexible script for scanning local data and generating inventory reports.
- **Sync Reporting:** Compares specific `data_root` against a tracked CSV manifest.
- **Configurable:** Uses `src.data_loading.config` (if integrated) or script-level paths.

**Key Features:**
- **Recursive Scanning:** Finds all files in the data directory.
- **Verification:** Checks file sizes and types against expected patterns.
- **Reporting:** Generates Markdown and JSON reports for audit trails.

---

## 2. Usage Guide

### Prerequisites
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Inventory Script
```bash
python3 data_inventory.py
```

### Configuration
- **Data Root:** Default is `.../Data/extracted`.
- **Skip Missing:** The script includes logic to note missing files without crashing.

### Troubleshooting
- **Import Error:** Check python path or venv.
- **Reports Directory:** Automatically created in `reports/`.

---

## 3. Implementation Details
(Derived from `DAT-01_COMPLETION_SUMMARY.md`)

- **Files Processed:** Scans `EDF`, `WAV`, `CSV` files.
- **Output:**
    - `data_inventory_report.md`
    - `file_manifest.csv`
