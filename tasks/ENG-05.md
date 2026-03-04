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
    - **Channel Selection**: Implements `select_optimal_channels` utilizing shared `src.utils.signal_processing.normalize_channel_names` logic for robust channel matching across systems. Prioritizes LH focus (F7, T7, P7, F3, C3, P3) and strictly validates focus inputs (`LH`, `RH`, `Clinical`).
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

## 4. ITPC Analysis
### Objectives
- **Quantify Covert Speech**: Use Inter-Trial Phase Coherence (ITPC) to measure neural tracking of hierarchial language structure.
- **Statistical Validation**: Compare Sentence Band vs Phrase Band vs Word Band ITPC using trial-level phase-scrambled permutation testing to distinguish true linguistic processing from random 1/f noise characteristics.

### Implementation
- **Core Logic**: `LanguageTrackingAnalysis.compute_itpc` uses Morlet wavelets; `compute_itpc_dft` provides zero-padding DFT cross-validation.
- **Metric Extraction**: Band-averaged ITPC across `SENTENCE_BAND` (0.71-0.85 Hz), `PHRASE_BAND` (1.49-1.63 Hz), and `WORD_BAND` (3.05-3.20 Hz).
- **DFT Zero-Padding**: FFT zero-padded to 0.01 Hz resolution so the sentence band bins align correctly.
- **Permutation Test**: Rigorous chance-level validation by randomly scrambling relative trial phases (0 to 2pi uniform) across iterations, then computing null distribution boundaries at exact subject levels.
- **Batch Analysis**: `awakenai run language` orchestrates processing across subjects.

### Verification Results
- **Subjects**: `CON008` & `CON009` (68-69 trials each, LH focus).
- **Finding**: While previously single-bin misaligned extractions showed false Sentence > Word trends, utilizing the exact acoustic stimulus speeds (Sentence: ~0.78Hz, Phrase: ~1.56Hz, Word: ~3.125Hz) reveals strong entrainment mostly tracking acoustically salient bounds (Word + Phrase frequencies), with Sentence-level synchronization remaining weak or failing the robust mathematical phase-scrambling permutation test (`p > 0.05`).
- **CON008 DFT**: Sentence (0.08, p=0.72), Phrase (0.14, p=0.17), Word (0.41, p<0.01). Strongest entrainment is at the direct acoustic word rate (3.125 Hz).
- **CON009 DFT**: Sentence (0.06, p=0.98), Phrase (0.13, p=0.17), Word (0.16, p=0.06). Marginal significance tracking word acoustic structures, poor hierarchical processing.
- **Visuals**: Topomaps and bar-charts automatically scale, accurately reflecting the localized low-SNR of higher tier hierarchies unless stimulus variations are completely controlled.

## 5. Definition of Done
- [x] `src/pipelines/language_tracking.py` refactored from `LanguageProcessor` to `LanguageTrackingAnalysis`, inheriting `BasePipeline`.
- [x] CLI-integrated standard executions `awakenai run language`.
- [x] Channel selection (`select_optimal_channels`) with LH/RH/Clinical focus (using Fp1/Fp2).
- [x] Unit tests in `tests/test_language_tracking.py` (14 passing).
- [x] ITPC Analysis implemented: Morlet primary + DFT cross-validation.
- [x] Band-averaged ITPC extraction mapped perfectly to stimulation structure (0.78Hz, 1.56Hz, 3.125Hz).
- [x] Formal statistical rigor built into the framework via intra-trial phase-scrambled permutation testing.
- [x] Extracted measurements successfully logged persistently per patient under outputs.

## 6. Next Steps
- **Group Statistics**: As N increases, run cluster-based permutation tests on the group level.
- **Clinical Correlation**: Correlate the "Hierarchical Ratio" with patient recovery outcomes.
