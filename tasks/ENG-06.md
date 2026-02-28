# ENG-06: QC Report Generation

## Overview

Implemented QC report generation that aggregates per-session QC metadata produced by ENG-03 (artifact rejection) and generates an HTML dashboard showing artifact rejection rates, SNR metrics, epoch counts, and ICA summaries per patient and trial type. Also produces a machine-readable summary CSV for downstream pipeline consumption.

**Architecture:** Three focused classes instead of one monolith:
- **`QCDataCollector`** — discovers and loads `eng03_qc.parquet` files.
- **`QCMetricsCalculator`** — computes derived metrics (drop rates, SNR, ICA stats).
- **`QCReportGenerator`** — renders HTML report + optional summary CSV.

## Implementation Details

### 1. QCDataCollector

- **Discovery**: Globs for `**/eng03_qc.parquet` under `data/processed/qc/`.
- **Loading**: Reads and concatenates all QC parquets into a single DataFrame.
- **Session access**: `load_session(patient_id, date)` loads a specific session; `get_available_sessions()` lists all (patient_id, date) pairs.
- **Graceful handling**: Returns empty DataFrames with the correct schema when no files or directories exist.

### 2. QCMetricsCalculator

Enriches raw ENG-03 QC data with derived metrics:

- **Drop rates**: `drop_rate = n_epochs_dropped / n_epochs_total` (0.0–1.0 scale).
- **SNR estimates**: `snr_db = 20 * log10(ptp_uv_p50 / ptp_uv_p95)` — ratio of median signal to noise floor. Negative dB; closer to 0 = cleaner data.
- **ICA summaries**: Parses the `ica` JSON column to extract flat fields: `ica_method`, `ica_classification_method`, component exclusion counts by artifact type (EOG, ECG, muscle, line noise, channel noise).
- **Aggregation**: `summary_by_patient()` and `summary_by_trial_type()` provide grouped statistics for dashboard and feature assembly.

All `compute_*` methods return new DataFrames — the original is never mutated.

### 3. QCReportGenerator

Produces a self-contained HTML dashboard:

- **Executive Summary**: Total patients, sessions, epochs, overall drop rate, mean SNR.
- **Trial Type Overview**: Cross-patient comparison table with aggregated metrics.
- **Per-Patient Sections**: Trial-type breakdown tables with drop rates, PTP thresholds, SNR, and ICA component summaries.
- **Inline CSS**: Clean, modern styling using system fonts, responsive layout, and hover effects. No external dependencies.
- **Summary CSV**: Machine-readable output for MOD-01 (Feature Assembly) and VIS-02 (Visualizations).

### 4. Convenience Function

`generate_qc_report()` chains all three classes in a single call:

```python
from src.data_processing.qc_report import generate_qc_report

report_path = generate_qc_report()
```

### 5. Configuration

Added `REPORTS_DIR = PROCESSED_DATA_DIR / "reports"` to `src/data_loading/config.py`.

### 6. Lazy Imports

Updated `src/data_processing/__init__.py` with a table-driven `__getattr__` pattern for all four exports (`QCDataCollector`, `QCMetricsCalculator`, `QCReportGenerator`, `generate_qc_report`). This avoids eagerly importing pandas/numpy at package import time.

## Output Schema

- **HTML report**: `data/processed/reports/qc_report.html`
- **Summary CSV**: `data/processed/reports/qc_summary.csv`
  - All columns from ENG-03 QC parquet, plus: `drop_rate`, `snr_db`, `ica_method`, `ica_classification_method`, `ica_n_components_excluded`, `ica_n_eog`, `ica_n_ecg`, `ica_n_muscle`, `ica_n_line_noise`, `ica_n_channel_noise`

## Dependencies

- `pandas` — data manipulation (already in project)
- `numpy` — numerical computation (already in project)
- No new external dependencies required

## Future Compatibility

- `QCMetricsCalculator.compute_all_metrics()` returns a standardized DataFrame that **MOD-01** (Feature Assembly) can directly ingest.
- `summary_by_patient()` and `summary_by_trial_type()` outputs feed into **VIS-02** (Final Visualizations).
- The HTML report satisfies the **MST-01** milestone requirement: "QC Dashboard: HTML report generated showing artifact rejection rates per patient."
- The modular design allows future tasks to extend with additional metrics (e.g., SCI-03 ITPC QC) by composing with `QCMetricsCalculator`.

## Testing

- 50 unit tests in `tests/test_qc_report.py` covering:
  - **QCDataCollector**: file discovery, concatenation, empty/missing directory handling, session loading, available sessions listing.
  - **QCMetricsCalculator**: drop rate computation, SNR calculation, ICA JSON parsing (valid, null, NaN, invalid, dict input), edge cases (zero epochs, missing PTP columns), summary aggregation, immutability.
  - **QCReportGenerator**: HTML file creation, content verification (patient IDs, metrics, CSS, ICA summary), CSV output, empty DataFrame handling, custom filenames.
  - **Helper functions**: `_empty_qc_dataframe`, `_parse_single_ica_json`, `_format_cell`, `_safe_id`, `_build_html_table`, `_aggregate_summary`.
  - **Integration**: end-to-end from synthetic parquet files to HTML report, empty directory graceful handling.
  - **Lazy imports**: all four exports resolve correctly via `__getattr__`; nonexistent names raise `AttributeError`.
- Full test suite (164 tests) passes with no regressions.

## Demo Notebook

`eda/eng06_qc_report_demo.ipynb` walks through the complete flow:
1. Discover and load QC data
2. Compute enriched metrics
3. Visualize drop rates (bar chart), SNR (box plot), ICA components (stacked bar)
4. Generate HTML report and display inline
5. Inspect summary CSV
