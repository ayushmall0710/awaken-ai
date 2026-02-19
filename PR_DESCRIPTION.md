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

### 5. ITPC Analysis
- **Core Logic**: Added `compute_itpc` to `LanguageProcessor` using Morlet wavelets.
- **Script**: `eda/run_itpc_analysis.py` performs batch analysis and statistical validation.
- **Features**: Computes ITPC at Sentence (0.065 Hz) and Word (0.77 Hz) rates.

## Verification
- **Unit Tests**: All 34 tests passed.
- **Integration**: Validated on patient `CON008` and `CON009`.
    - **CON008**: Selected 19 channels, ~8.18 µV mean amplitude.
    - **CON009**: Selected 19 channels, ~7.48 µV mean amplitude, 68 epochs.
    - **Filter Check**: Confirmed 0.5 Hz high-pass is applied correctly in both `ArtifactRejector` and `LanguageProcessor`.
    - **PSD**: Verified 1/f scaling with intact Delta band power.
- **ITPC Analysis**:
    - **Hierarchical Processing**: Confirmed Sentence Rate ITPC > Word Rate ITPC for both patients (Ratio ~1.1-1.2).
    - **Visuals**: Topomaps show left-hemisphere dominance.

## Analysis Results
### Quantitative Findings
| Metric | CON008 | CON009 |
| :--- | :--- | :--- |
| **Sentence ITPC (0.065 Hz)** | **0.1251** | **0.1262** |
| **Word ITPC (0.77 Hz)** | 0.1031 | 0.1149 |
| **Hierarchical Ratio** | **1.21** | **1.10** |

> **Interpretation**: In both patients, **Sentence ITPC > Word ITPC** (Ratio > 1.0). This suggests that neural entrainment is **not** driven solely by the rapid acoustic envelope of words, but reflects tracking of the slower sentence structure.

### Visualizations
**CON008 Topomap** (Left-lateralized at 0.065 Hz)
![CON008 Topomap](https://raw.githubusercontent.com/ayushmall0710/awaken-ai/feature/ENG-05-refactor/data/outputs/CON008/CON008_language_ITPC_topomap.png)

**CON008 TFR** (Phase coherence over time)
![CON008 TFR](https://raw.githubusercontent.com/ayushmall0710/awaken-ai/feature/ENG-05-refactor/data/outputs/CON008/CON008_language_ITPC_tfr.png)

## Documentation
- Updated `tasks/ENG-05.md` with implementation details and verification results.
- Added `tasks/ENG-05-analysis.md` Detailed ITPC Analysis Plan, Methodology, and Results.

## Closes
ENG-05
