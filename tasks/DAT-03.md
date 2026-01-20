# DAT-03: Schema Unification

**Status:** Completed (Refactored)
**Category:** Data Engineering
**Date:** January 18, 2026

---

## 1. Overview
The goal of DAT-03 is to unify disparate `stimulus_results.csv` and `patient_df.csv` files into a single, consistent dataset.
**Recent Update:** The pipeline now outputs **Parquet** (`unified_stimulus_results.parquet`) instead of CSV to support nested dictionary structures in the `sentences` column.

---

## 2. Usage Guide

### Pipeline Script
The main entry point for unification is now:
```bash
python3 scripts/unify_stimulus_data.py
```
This script uses the logic in `src/data_processing`.

### Analysis/Verification
To analyze the output:
```bash
python3 eda/run_analysis.py
```

---

## 3. Schema Details

### Target Schema (Parquet)
| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | str | Patient ID (e.g., CON008) |
| `date` | str | Date string |
| `trial_type` | str | Normalized type (e.g., `language`, `oddball`) |
| `sentences` | List[Dict] | `[{'event': '...', 'onset_time': ...}]` |
| `start_time` | float | Unix timestamp |
| `end_time` | float | Unix timestamp |
| `duration` | float | Trial duration in seconds |
| `source_file` | str | Original CSV filename for provenance tracking |

### Normalization Logic
- **Trial Types:** collapsed from ~80 variants to ~7 standard types.
    - `lang_XX` types have their index extracted to `sentences` before normalization.
- **Sentences:** string/int/dict inputs are all converted to `List[Dict]`.
- **Beep/Control:** Kept as separate types (`beep` vs `control`) per user request.

---

## 4. Legacy Info (Obsolete Script)
*Note: `csv_schema_unifier.py` was the previous iteration. It is now superseded by `src/data_processing`.*

**Previous Capabilities:**
- Hash-based deduplication (still relevant concept, to be verified in new pipeline if needed).
- PDF/MD Reports.

**Current Pipeline:**
- Focuses on correctness of data types (Parquet) and data rescue (`lang_XX`).
