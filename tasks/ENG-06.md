# ENG-06: Language Tracking Analysis

**Status:** Planned
**Dependencies:** ENG-05 (Language Optimization), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Objective:** Quantify the neural entrainment to speech envelopes using Temporal Response Functions (TRF).

## 1. Overview
With the successful isolation of clean, language-specific EEG epochs (ENG-05), the next step is to model the relationship between the auditory stimulus (speech) and the neural response. This analysis will focus on the **Delta band (0.5 - 4 Hz)**, which corresponds to the prosodic rate of speech (sentence/phrase level).

## 2. Inputs
- **Neural Data**: Cleaned `mne.Epochs` from `ArtifactRejector` (via `LanguageProcessor`), filtered 0.5-30 Hz.
- **Stimulus Data**: Audio files corresponding to the sentences presented.
- **Alignment**: Precise event markers from `TimestampAligner` (ENG-02).

## 3. Analysis Plan

### Phase 1: Feature Extraction
- **Speech Envelope**: Extract the amplitude envelope of the stimulus audio (Hilbert transform).
- **Downsampling**: Resample audio envelopes to match EEG variation (e.g., 64 Hz or 128 Hz).

### Phase 2: TRF Modeling (Forward Model)
- **Method**: Regularized Linear Regression (Ridge Regression) or mTRF (multivariate Temporal Response Function).
- **Input**: Speech Envelope (lagged tmin to tmax).
- **Output**: Predicted EEG response.
- **Metric**: Pearson correlation between Predicted vs. Actual EEG.

### Phase 3: Statistical Validation
- **Permutation Testing**: Shuffle stimulus-response pairs to build a null distribution.
- **Topography**: visualize reconstruction accuracy (correlation `r`) across the scalp (expecting LH dominance).

## 4. Implementation Strategy
1.  **Audio Loader**: Create utility to load and process stimulus audio files, matching them to trials in `UnifiedDataLoader`.
2.  **TRF Solver**: Implement or wrap a TRF solver (e.g., `mtrfpy` or `scikit-learn` Ridge).
3.  **Pipeline Script**: `eda/run_trf_analysis.py` to loop through patients (CON008, CON009) and compute models.

## 5. Success Criteria
- [ ] Significant TRF correlations (p < 0.05) over Left Hemisphere channels (e.g., F7, T7).
- [ ] Distinct reconstruction accuracy for "Language" trials vs. "Baseline/Rest" (if available).
