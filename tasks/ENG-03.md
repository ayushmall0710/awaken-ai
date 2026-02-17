# ENG-03: Artifact Rejection (ICA) Implementation

## Overview

Implemented session-level ICA-based artifact rejection for EEG recordings. The pipeline consumes aligned events from ENG-02, applies Independent Component Analysis to remove artifacts (eye blinks, heartbeat, muscle, line noise, channel noise), then exports fixed-window EEG-only epochs per trial type as MNE `.fif` files alongside QC metadata as Parquet.

**Classification strategy (Option B):**
- **Primary**: ICLabel neural-network classifier — detects all 7 artifact types (brain, eye, heart, muscle, line noise, channel noise, other).
- **Fallback**: Correlation-based detection (`find_bads_eog`, `find_bads_ecg`) when ICLabel cannot run (e.g. montage setup fails for non-standard channel names).

## Implementation Details

### 1. Channel Selection

- **Problem**: Clinical EDF files label *all* channels (including EMG, ECG, respiratory, body-position sensors) as type `eeg`. Naively using `mne.pick_types(eeg=True)` includes 50 channels when only 22 are actual scalp EEG.
- **Solution**: Expanded the keyword-based exclusion list (`NON_EEG_CHANNEL_KEYWORDS` in `src/utils/signal_processing.py`) to cover polysomnography channels (EMG, ECG, RESP, ABD, FLOW, SNORE, OSAT, PR, etc.). `_pick_eeg_indices` now always intersects type-based picks with the keyword exclusion, rather than short-circuiting on the type check alone.
- **Result**: ICA is fitted on exactly the 22 true scalp EEG channels (10-20 + FT9/FT10).

### 2. EOG Channel Detection

- **Problem**: Automated blink-component detection (`find_bads_eog`) needs a reference channel that captures eye movement. EDFs in this dataset have dedicated infraorbital electrodes (IO1, IO2) but they aren't typed as EOG.
- **Solution**: `_find_eog_channels` uses a priority cascade: typed EOG → union of name-based "EOG" **and** IO1/IO2 (both can coexist) → Fp1/Fp2 (surrogate). The lower `find_bads_eog` threshold (2.5 vs default 3.0) accounts for weaker cross-correlation from surrogate channels.
- **Result**: IO1/IO2 are correctly identified and used, yielding robust blink-component detection.

### 3. ICA Fitting & Classification

- **ICA method**: Extended Infomax (`method="infomax"`, `fit_params=dict(extended=True)`) — handles both sub-Gaussian (eye blinks) and super-Gaussian (muscle) sources, and is the method ICLabel was trained on.
- **Filter range**: 1–100 Hz bandpass before ICA fitting (broadened from 1–40 Hz to preserve line-noise frequency content for ICLabel classification).
- **Montage setup**: `_try_set_montage` attempts to map channel names to standard 10-20 positions (stripping prefixes like `EEG Fp1` → `Fp1`). If >= 5 channels match, the montage is applied to enable ICLabel's topographic features.
- **ICLabel (primary)**: When montage is set, `mne_icalabel.label_components` classifies each component into 7 categories. Components not labeled as `brain` or `other` (above a configurable probability threshold, default 0.5) are excluded. This catches line noise (60 Hz), channel noise, muscle, eye, and heartbeat artifacts in a single pass.
- **Correlation fallback**: If ICLabel fails (import error, montage insufficient, classification error), the pipeline falls back to `find_bads_eog`, `find_bads_ecg`, and (when sensor positions exist) `find_bads_muscle`.

### 4. Modular Architecture

The monolithic `_apply_ica` (~130 lines) and `run_session` (~95 lines) were decomposed into focused, independently testable sub-functions:

- `_prepare_ica_data` — copy raw, pick EEG indices, bandpass filter
- `_fit_ica` — create and fit ICA (extended infomax)
- `_try_set_montage` — attempt standard 10-20 montage with prefix stripping
- `_classify_components_iclabel` — ICLabel neural-network classifier
- `_classify_components_correlation` — EOG/ECG/muscle correlation fallback
- `_apply_and_summarize` — apply exclusions and build ICASummary

`run_session` delegates to `_load_session_inputs`, `_process_trial_type`, and `_save_qc`.

### 5. Epoch Building & Rejection

- **Vectorized time conversion**: Replaced `iterrows` loop with NumPy vectorized operations for converting Unix timestamps to EDF-relative seconds.
- **Fixed-window epochs**: Each trial type has a configured window length (e.g., language=16s, oddball=35s).
- **EEG-only output**: Epochs contain only the scalp EEG channels — no DC, EMG, or other auxiliary channels.
- **PTP auto-rejection**: Epochs whose max peak-to-peak amplitude exceeds the 95th percentile are automatically dropped, with the threshold and drop indices recorded in QC metadata.

### 6. Logging & Diagnostics

- **`_note()` helper**: Each diagnostic message is both appended to the `notes` list (persisted in QC parquet) and emitted via `logger.debug()` for runtime log files.
- **Section prefixes**: Notes use `[CHANNELS]`, `[FILTER]`, `[EOG]`, `[ECG]`, `[MUSCLE]`, `[ICLABEL]`, `[MONTAGE]`, `[CORRELATION]` tags for easy grep-ability.
- **ICASummary dataclass**: Records `classification_method` (`iclabel` or `correlation`), per-component `iclabel_labels` and `iclabel_probs` for component-level QC, and dedicated fields for each artifact category (`eog_components`, `ecg_components`, `muscle_components`, `line_noise_components`, `channel_noise_components`).

### 7. Shared Utilities & Imports

- **`src/utils/time_utils.py`**: Extracted `detect_timezone_offset`, `unix_to_edf`, and `edf_to_unix` from `TimestampAligner` into standalone functions. Both ENG-02 and ENG-03 delegate to these shared utilities.
- **`UnifiedDataLoader.load_aligned_events`**: Aligned-events loading moved to `UnifiedDataLoader` for cross-module reuse.
- **Lazy imports via `__getattr__`**: Both `src/data_loading/__init__.py` and `src/data_processing/__init__.py` use `__getattr__` for deferred imports, keeping MNE lazy while ensuring `from src.data_processing import ArtifactRejector` works at runtime (not just under `TYPE_CHECKING`).

### 8. Output Schema

- **Epochs**: `data/processed/epochs/{patient_id}/{date}/{trial_type}-epo.fif`
- **QC metadata**: `data/processed/qc/{patient_id}/{date}/eng03_qc.parquet`
  - Columns: `patient_id`, `date`, `trial_type`, `window_sec`, `reject_ptp_percentile`, `reject_ptp_threshold_uv`, `n_epochs_total`, `n_epochs_dropped`, `n_epochs_kept`, `drop_reason`, `ica` (JSON), `notes` (JSON), `ptp_uv_p50/p95/p99/max/mean`

## Dependencies

- `mne>=1.6` — EEG data processing
- `mne-icalabel>=0.7` — ICLabel neural-network component classifier
- `onnxruntime>=1.16` — ONNX backend for ICLabel (lightweight, no PyTorch required)

## Why this approach?

- **ICLabel as primary classifier**: Catches all 7 artifact types (including line noise at 60 Hz and channel noise) that correlation-based methods miss entirely.
- **Correlation fallback**: Robust to clinical EDFs with non-standard channel names where montage cannot be set.
- **Extended Infomax**: Handles both sub-Gaussian and super-Gaussian sources; is the method ICLabel was trained on.
- **Session-level ICA**: One fit per EDF avoids over-fitting to individual trials and keeps computation tractable.
- **Keyword + type intersection for channel selection**: Robust to the common EDF scenario where all channels default to type `eeg`.
- **Modular sub-functions**: Each step is independently testable, reusable by future tasks, and easy to swap out.

## Testing

- 25 unit tests in `tests/test_artifact_rejection.py` covering: window mapping, time conversions, channel exclusion (DC, polysomnography), EOG detection (Fp fallback, IO preference, combined EOG+IO), EEG-only picking, `_note` helper (appends + logs), montage setup (standard names, non-standard, prefix stripping), ICLabel classification (success, import error fallback), correlation classifier, integration fallback path, lazy `__getattr__` imports, and QC stats.
- All tests use `pytest.importorskip("mne")` to gracefully skip in environments without MNE.
- Full test suite (112 tests) passes.
