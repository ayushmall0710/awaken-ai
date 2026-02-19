# PR: ENG-05 Language Optimization Pipeline (Improved)

## Description
Implements the end-to-end `LanguageProcessor` pipeline for analyzing neural entrainment to speech, with improvements: 256 Hz downsampling, DFT-based ITPC cross-validation, expanded test coverage, and a fixed verification script.

## Key Changes

### 1. Language Optimization (`src/data_processing/language_optimization.py`)
- **Trial Isolation**: Loads cleaned epochs from `ArtifactRejector` (ENG-03).
- **Filtering + Downsampling**: Applies 0.5-30 Hz bandpass, then downsamples from 512 Hz to 256 Hz -- halves ITPC computation time while preserving all analysis frequencies (Nyquist 128 Hz >> 30 Hz).
- **Electrode Selection**: `select_optimal_channels` with LH language area preference (F7, T7, P7, etc.).
- **Morlet ITPC**: `compute_itpc` via `tfr_morlet`, targeting 0.05-2.0 Hz.
- **DFT ITPC**: `compute_itpc_dft` using FFT-based phase coherence (Sokoliuk 2021 method) for cross-validation. Frequency resolution ~0.0625 Hz for 16s epochs.

### 2. Artifact Rejection (`src/data_processing/artifact_rejection.py`)
- Changed `DEFAULT_ICA_FILTER_HZ` from `(1.0, 100.0)` to `(0.5, 100.0)` to preserve Delta band.

### 3. Shared Utilities (`src/utils/signal_processing.py`)
- Added `normalize_channel_names` for unified channel cleaning.

### 4. EDA Scripts
- **`eda/run_itpc_analysis.py`**: Batch analysis using both Morlet and DFT, saves combined CSV.
- **`eda/compare_itpc_methods.py`**: Side-by-side comparison script with bar chart visualization.
- **`eda/verify_language_optimization.py`**: Fixed runtime crash (removed invalid `aligned_events` kwarg), removed emojis, simplified to use current API.

## Tests
All **13 tests** pass. New tests added:
- `test_preprocess_signal_no_downsample` -- skip resample when already at target sfreq
- `test_compute_itpc_returns_data_and_itc` -- shape and value range
- `test_compute_itpc_custom_freqs` -- custom freq/cycle override
- `test_compute_itpc_dft_returns_spectrum` -- DFT shape and freq resolution
- `test_extract_itpc_metrics_structure` -- all expected keys present
- `test_extract_itpc_metrics_zero_word` -- division-by-zero safety
- `test_select_optimal_channels_clinical` -- Clinical focus path
- `test_process_patient_no_data` -- returns `None` on all-session failure

## Analysis Results (Morlet, pre-re-analysis)
| Metric | CON008 | CON009 |
| :--- | :--- | :--- |
| **Sentence ITPC (0.065 Hz)** | **0.1251** | **0.1262** |
| **Word ITPC (0.77 Hz)** | 0.1031 | 0.1149 |
| **Hierarchical Ratio** | **1.21** | **1.10** |

> **Interpretation**: Sentence ITPC > Word ITPC in both patients (Ratio > 1.0), consistent with hierarchical speech tracking. DFT cross-validation pending re-analysis with 256 Hz data.

### Visualizations
**CON008 Topomap** (Left-lateralized at 0.065 Hz)
![CON008 Topomap](https://raw.githubusercontent.com/ayushmall0710/awaken-ai/feature/ENG-05-refactor/data/outputs/CON008/CON008_language_ITPC_topomap.png)

**CON008 TFR** (Phase coherence over time)
![CON008 TFR](https://raw.githubusercontent.com/ayushmall0710/awaken-ai/feature/ENG-05-refactor/data/outputs/CON008/CON008_language_ITPC_tfr.png)

## Methodology Notes (vs. Sokoliuk 2021)
| Sokoliuk Step | Our Approach | Reason |
| :--- | :--- | :--- |
| 0.01 Hz high-pass | 0.5 Hz | ICA stability; still captures 0.065 Hz sentence rate |
| Notch 48-52, 98-102 Hz | Not applied | 30 Hz low-pass already eliminates power-line noise |
| Downsample to 256 Hz | Applied | Source EDFs are 512 Hz; downsampling halves compute time |
| Discard first 2.28 s | Not applied | Sokoliuk epochs at tmin=-1.0s; ours at tmin=0.0s (onset) |
| DFT for ITPC | Both Morlet + DFT | Morlet is primary; DFT added for Sokoliuk cross-validation |

## Documentation
- `tasks/ENG-05.md` -- implementation details and verification results.
- `tasks/ENG-05-analysis.md` -- ITPC analysis plan, methodology notes (Sokoliuk divergences), results.

## Closes
ENG-05
