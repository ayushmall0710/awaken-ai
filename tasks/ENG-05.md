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
- **Language Processor**: `src/pipelines/language_tracking.py` (inherits `BasePipeline`)
    - **Input**: Expects `ArtifactRejector` to have been run first (generates `-epo.fif` files).
    - **Channel Selection**: Implements `_select_optimal_channels` via spatial cluster permutation on comprehension-frequency coherence. Supports focus inputs (`LH`, `RH`, `Clinical`, `Optimal`).
    - **Filtering**: Applies 0.02-25.0 Hz bandpass filter (`HIGHPASS_FREQ`, `LOWPASS_FREQ` constants).
    - **Output**: Returns `pd.DataFrame` in long-format with ITPC metrics per focus.

### Key Decisions
- **Strict Dependency on ENG-03**: The `LanguageTrackingAnalysis` strictly depends on the artifact rejection pipeline.
- **Hierarchical Frequency Selection**: Frequencies are precisely aligned to 0.78 Hz (Sentence), 1.56 Hz (Phrase), and 3.125 Hz (Word).
- **Statistical Rigor**: Trial-level random phase scrambling is used for null distributions, ensuring p-values are calibrated against 1/f noise.

### Constraints & Limitations
- **Optimal Focus**: Requires significant spatial clusters to be identified via permutation testing; otherwise, it returns NaN for metrics.

### Verification & Analysis
- **Unit Tests**: `tests/test_language_tracking.py` and `tests/test_language_tracking_morlet_pvals.py`.
- **Visualization**: `src/viz/language_plots.py` provides TFR, Topomaps (with dynamic `vlim`), and per-channel bar plots.

## 4. ITPC Analysis (Sokoliuk 2021 Implementation)
### Objectives
- **Quantify Covert Speech**: Use Inter-Trial Phase Coherence (ITPC) to measure neural tracking of hierarchical language structure.
- **Statistical Validation**: Use trial-level phase-scrambled permutation testing and spatial cluster permutation to identify significant entrainment.

### Implementation
- **Core Logic**: `LanguageTrackingAnalysis.analyze` orchestrates a two-phase architecture (per-channel computation followed by focus aggregation).
- **Metric Extraction**: Band-averaged ITPC across `SENTENCE_BAND` (0.70-0.86 Hz), `PHRASE_BAND` (1.40-1.72 Hz), and `WORD_BAND` (2.81-3.44 Hz) using ±10% bandwidth.
- **New Metrics**: Includes `ratio_sent_phrase` and `ratio_bw_normalized` (bandwidth-normalized ratio).
- **DFT Zero-Padding**: FFT zero-padded to 0.01 Hz resolution to ensure bin alignment.
- **Permutation Test**: n=1000 surrogates per session to calculate p-values for all linguistic levels.

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
