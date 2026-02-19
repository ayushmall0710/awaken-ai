# PR: ENG-05 Language Optimization Pipeline

## Description
This PR implements the end-to-end `LanguageProcessor` pipeline for analyzing neural entrainment to speech in the Language Tracking paradigm. It ensures that data is processed with a 0.5-30 Hz bandpass filter (preserving Delta band sentence information) and reduces the electrode montage to a clinically relevant 19-channel set (Left Hemisphere focus).

## Key Changes
### 1. Language Optimization Logic (`src/data_processing/language_optimization.py`)
- **Trial Isolation**: Loads cleaned epochs directly from `ArtifactRejector` (ENG-03) output.
- **Filtering**: Applies 0.5-30 Hz bandpass filter to capture sentence envelopes.
- **Electrode Selection**: Implements `select_optimal_channels` with a preference for LH language areas (F7, T7, P7, etc.).

### 2. Artifact Rejection Update (`src/data_processing/artifact_rejection.py`)
- **Enhancement**: Changed `DEFAULT_ICA_FILTER_HZ` from `(1.0, 100.0)` to `(0.5, 100.0)` to support low-frequency (Delta) analysis without attenuation.
- **Refactor**: Utilizes shared `normalize_channel_names` logic.

### 3. Shared Utilities (`src/utils/signal_processing.py`)
- **New**: Added `normalize_channel_names` to unify channel cleaning logic across the pipeline.

### 4. Tests
- **Updated**: `tests/test_language_optimization.py` covers initialization, filtering, and channel selection.
- **New Check**: `tests/test_artifact_rejection.py` now explicitly asserts the 0.5 Hz high-pass default.

## Verification
- **Unit Tests**: All 34 tests passed.
- **Integration**: Validated on patient `CON008`.
    - Selected 19 channels successfully.
    - Signal Amplitude: ~8.18 µV (Physiological).
    - PSD: Verified 1/f scaling with intact Delta power.

## Documentation
- Updated `tasks/ENG-05.md` with implementation details and verification results.
- Added `tasks/ENG-05-analysis.md` Detailed ITPC Analysis Plan (renamed from ENG-06).

## Closes
ENG-05
