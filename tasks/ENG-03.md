# ENG-03: Artifact Rejection (ICA) Implementation

## Overview

Implemented session-level ICA-based artifact rejection for EEG recordings. The pipeline consumes aligned events from ENG-02, applies Independent Component Analysis to remove eye-blink and (where sensor positions exist) muscle artifacts, then exports fixed-window EEG-only epochs per trial type as MNE `.fif` files alongside QC metadata as Parquet.

## Implementation Details

### 1. Channel Selection

- **Problem**: Clinical EDF files label *all* channels (including EMG, ECG, respiratory, body-position sensors) as type `eeg`. Naively using `mne.pick_types(eeg=True)` includes 50 channels when only 22 are actual scalp EEG.
- **Solution**: Expanded the keyword-based exclusion list (`NON_EEG_CHANNEL_KEYWORDS` in `src/utils/signal_processing.py`) to cover polysomnography channels (EMG, ECG, RESP, ABD, FLOW, SNORE, OSAT, PR, etc.). `_pick_eeg_indices` now always intersects type-based picks with the keyword exclusion, rather than short-circuiting on the type check alone.
- **Result**: ICA is fitted on exactly the 22 true scalp EEG channels (10-20 + FT9/FT10).

### 2. EOG Channel Detection

- **Problem**: Automated blink-component detection (`find_bads_eog`) needs a reference channel that captures eye movement. EDFs in this dataset have dedicated infraorbital electrodes (IO1, IO2) but they aren't typed as EOG.
- **Solution**: `_find_eog_channels` uses a priority cascade: typed EOG → name contains "EOG" → IO1/IO2 (infraorbital) → Fp1/Fp2 (surrogate). The lower `find_bads_eog` threshold (2.5 vs default 3.0) accounts for weaker cross-correlation from surrogate channels.
- **Result**: IO1/IO2 are correctly identified and used, yielding robust blink-component detection (2 components excluded for CON008).

### 3. ICA Fitting & Application

- **Approach**: Session-level ICA (one fit per EDF recording) using FastICA with `n_components=0.99` (99% explained variance). The EEG channels in a full copy of `raw` are bandpass-filtered (1-40 Hz) before fitting; the ICA is then applied to the original unfiltered data.
- **Muscle detection**: `find_bads_muscle` is only run when digitization/sensor positions are available. Without a montage the slope-only fallback is overly aggressive (flagged 6/12 components on CON008), so it is skipped with a diagnostic note.
- **Safety**: All channels remain in `raw_for_ica` so that EOG reference channels are available for `find_bads_eog`, even though only EEG picks are used for the ICA fit.

### 4. Epoch Building & Rejection

- **Fixed-window epochs**: Each trial type has a configured window length (e.g., language=16s, oddball=35s). Trial start times are converted from Unix to EDF-relative seconds using the shared `unix_to_edf` utility (same formula as ENG-02).
- **EEG-only output**: Epochs contain only the 22 scalp EEG channels — no DC, EMG, or other auxiliary channels.
- **PTP auto-rejection**: Epochs whose max peak-to-peak amplitude exceeds the 95th percentile are automatically dropped, with the threshold and drop indices recorded in QC metadata.

### 5. Shared Utilities Refactoring

- **`src/utils/time_utils.py`** (new): Extracted `detect_timezone_offset`, `unix_to_edf`, and `edf_to_unix` from `TimestampAligner` into standalone functions. Both ENG-02 and ENG-03 now delegate to these shared utilities, eliminating code duplication.
- **Lazy MNE imports**: `src/data_loading/__init__.py`, `src/data_processing/__init__.py`, and `src/data_loading/patient_data.py` now use `TYPE_CHECKING` guards so that `import mne` is deferred, allowing lightweight modules (DAT-03 tests, etc.) to import the package without requiring MNE.

### 6. Output Schema

- **Epochs**: `data/processed/epochs/{patient_id}/{date}/{trial_type}-epo.fif`
- **QC metadata**: `data/processed/qc/{patient_id}/{date}/eng03_qc.parquet`
  - Columns: `patient_id`, `date`, `trial_type`, `window_sec`, `reject_ptp_percentile`, `reject_ptp_threshold_uv`, `n_epochs_total`, `n_epochs_dropped`, `n_epochs_kept`, `drop_reason`, `ica` (JSON), `notes` (JSON), `ptp_uv_p50/p95/p99/max/mean`

## Why this approach?

- **Session-level ICA**: One fit per EDF avoids over-fitting to individual trials and keeps computation tractable.
- **Keyword + type intersection for channel selection**: Robust to the common EDF scenario where all channels default to type `eeg`.
- **Conservative muscle detection**: Skipping `find_bads_muscle` without sensor positions prevents discarding legitimate brain signal; the decision is logged for auditability.

## Testing

- 14 unit tests in `tests/test_artifact_rejection.py` covering: window mapping, time conversions, channel exclusion (DC, polysomnography), EOG detection (Fp fallback, IO preference), EEG-only picking, and QC stats.
- All tests use `pytest.importorskip("mne")` to gracefully skip in environments without MNE.
- Full test suite (101 tests) passes.
