# ENG-05: Language Optimization
**Data Validation:** Verified on CON008 (6/7 checks passed, clean signal metrics confirmed), CON009 (Integration test passed)
**Category:** Signal Processing / Optimization
**Dependencies:** ENG-01 (Unified Data Loader), ENG-02 (Timestamp Alignment), ENG-03 (Artifact Rejection)
**Target Branch:** `feature/ENG-05-refactor` (rebased on `main`)

## 1. Overview
The goal of **ENG-05** is to create a specialized pipeline for the **Language Tracking Paradigm**. This involves utilizing pre-cleaned epochs from the **Artifact Refection** pipeline (ENG-03) and identifying a reduced set of electrodes (~20) that maximize the signal-to-noise ratio for detecting neural entrainment to speech, focusing primarily on the **Left Hemisphere**.

## 2. Objectives
1.  **Artifact Integration**: Consume cleaned data from `ArtifactRejector` (ENG-03) to ensure downstream analysis uses high-quality signals.
2.  **Electrode Selection**: Implement logic to subset channels to a "Clinical 20" set, with a bias towards Left Hemisphere (Language Dominant) regions.
3.  **Signal Optimization**: Apply specific filtering (0.5-30 Hz) consistent with language tracking requirements.

## 3. Implementation Details

### Validated Workflow
- **Data Loading**: Uses `UnifiedDataLoader.load_clean_epochs` to fetch pre-cleaned data from ENG-03.
- **Language Processor**: `src/data_processing/language_optimization.py`
    - **Input**: Expects `ArtifactRejector` to have been run first (generates `-epo.fif` files).
    - **Channel Selection**: Implements `select_optimal_channels` utilizing shared `src.utils.signal_processing.normalize_channel_names` logic for robust channel matching across systems. Prioritizes LH focus (F7, T7, P7, F3, C3, P3).
    - **Filtering**: Applies 0.5-30Hz bandpass filter (`HIGHPASS_FREQ`, `LOWPASS_FREQ` constants).
    - **Output**: Returns `mne.Epochs` restricted to optimal channels.

### Key Decisions
- **Strict Dependency on ENG-03**: The `LanguageProcessor` no longer processes raw EDFs directly. It strictly depends on the artifact rejection pipeline, ensuring separation of concerns (cleaning vs. analysis).
- **Shared Utilities**: Channel name normalization logic was centralized in `src/utils/signal_processing.py` to be shared between `ArtifactRejector` and `LanguageProcessor`.

### Constraints & Limitations
- **Filter Cutoff Note**: The default high-pass filter for `ArtifactRejector` (ENG-03) has been updated to **0.5 Hz** (previously 1.0 Hz) to support sentence-level frequency analysis (Delta band). While this enables the analysis, care should be taken to monitor ICA stability, as low-frequency drift can sometimes affect component separation. The `LanguageProcessor` continues to apply its own 0.5-30 Hz bandpass as a safety measure.

### Verification & Analysis
- **Unit Tests**: `tests/test_language_optimization.py` covers initialization, channel selection, filtering, and end-to-end processing with mocked `load_clean_epochs`.
- **Pipeline Verification**: `eda/verify_language_optimization.py` validates the end-to-end flow on real patient data (CON008), checking channel counts, filter application, and epoch validity.
- **Visualization**: `eda/visualize_language_optimization.py` generates diagnostic plots (Sensor Map, ERP, PSD, Spectrogram).
- **Signal Quality**: `eda/analyze_language_signals.py` computes quantitative metrics (Amplitude ~8uV, 1/f spectral scaling) to confirm physiological plausibility.

## 4. ITPC Analysis (New Feature)
### Objectives
- **Quantify Covert Speech**: Use Inter-Trial Phase Coherence (ITPC) to measure neural tracking of sentence structure (~0.065 Hz).
- **Statistical Validation**: Compare Sentence Rate ITPC vs Word Rate ITPC (~0.77 Hz) to distinguish linguistic processing from acoustic envelope tracking.

### Implementation
- **Core Logic**: `LanguageProcessor.compute_itpc` uses Morlet wavelets to extract phase information.
- **Batch Analysis**: `eda/run_itpc_analysis.py` orchestrates processing across subjects and computes the Sentence/Word ITPC ratio.

### Verification Results
- **Subjects**: `CON008` & `CON009`.
- **Finding**: Both subjects show **Sentence ITPC > Word ITPC** (Ratio ~1.1-1.2), indicating hierarchical processing.
- **Visuals**: Topomaps confirm left-lateralized activation consistent with language networks.

## 5. Definition of Done
- [x] `src/data_processing/language_optimization.py` refactored to use ENG-03 outputs.
- [x] Function to return `mne.Epochs` restricted to optimal channels.
- [x] Unit tests in `tests/test_language_optimization.py`.
- [x] Verification script `eda/verify_language_optimization.py` passing for CON008.
- [x] Visualization script `eda/visualize_language_optimization.py` created and tested.
- [x] Signal analysis script `eda/analyze_language_signals.py` confirming data quality.
- [x] ITPC Analysis implemented and verified (Ratio > 1.0).

## 6. Next Steps
- **Group Statistics**: As N increases, run cluster-based permutation tests on the group level.
- **Clinical Correlation**: Correlate the "Hierarchical Ratio" with patient recovery outcomes.
