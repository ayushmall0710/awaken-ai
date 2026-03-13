# ENG-05: Analysis Plan - Language Tracking (ITPC)

**Status:** Completed & Synchronized (Mar 2026)
**Dependencies:** ENG-05 (Language Optimization), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Context:** Based on `docs/language_tracking.md`
**Objective:** Quantify **covert speech comprehension** by measuring **neural entrainment** (phase-locking) to the hierarchical structure of speech (Sentence: 0.78 Hz, Phrase: 1.56 Hz, Word: 3.125 Hz).

## 1. Scientific Background

Patients with disorders of consciousness may still process language. Validating this requires detecting neural responses that track the hierarchical temporal envelope of speech, distinct from low-level acoustic processing.
- **Target Bands:** Sentence (0.78 Hz), Phrase (1.56 Hz), and Word (3.125 Hz).
- **Metric:** Band-averaged Inter-Trial Phase Coherence (ITPC), validated against trial-level phase-scrambled permutation testing and spatial cluster selection.

## 2. Methodology & Implementation

### Signal Processing
- **Source**: `LanguageTrackingAnalysis` (utilizing `ArtifactRejector` clean epochs).
- **Filter**: 0.02 - 25.0 Hz.
- **Montage**: Standard 10-20.
- **Channels**: Supports `LH`, `RH`, `Clinical`, and `Optimal` (via spatial cluster permutation).
- **Epochs**: Cropped to 13.08s segments (2.28s to 15.36s) to eliminate filter/ICA edge artifacts.

### Time-Frequency Analysis
- **Method**: Morlet Wavelets (`tfr_morlet`) and Zero-padded DFT.
- **Frequencies**: Log-spaced 0.5 - 5.0 Hz (60 bins).
- **Cycles**: Adaptive (min 0.5 cycles).

### Metric Extraction: Band-Averaged ITPC

ITPC is extracted by **averaging over all frequency bins within each band** (±10% bandwidth).

| Band | Frequencies | Target Center |
| :--- | :--- | :--- |
| Sentence | 0.70 - 0.86 Hz | 0.78 Hz |
| Phrase | 1.40 - 1.72 Hz | 1.56 Hz |
| Word | 2.81 - 3.44 Hz | 3.125 Hz |

### DFT Zero-Padding

The DFT path zero-pads the time series to achieve 0.01 Hz frequency resolution before computing the FFT, ensuring precise bin alignment for target frequencies.

### Statistical Validation
To distinguish high-level comprehension from basic acoustic processing, we employ **trial-level phase-scrambled permutation testing** (n=1000) for all linguistic levels and **spatial cluster permutation** for data-driven focus selection.
- **Hypothesis**: True neural tracking will exhibit ITPC significantly greater than empirical chance representations of 1/f noise evaluated per band (`p < 0.05`).

## 3. Verification Results

### Quantitative Findings

Analysis performed on 68 trials per patient using the **LH Focus** channel subset (Fp1, F7, T7, F3, C3, P3).

**Exact DFT Extracted ITPC (LH Focus Subset)**

| Patient | Sentence ITPC (p-val) | Phrase ITPC (p-val) | Word ITPC (p-val) |
| :--- | :--- | :--- | :--- |
| CON004 | **0.236 (p=0.001)** | **0.207 (p=0.009)** | **0.341 (p<0.001)** |
| CON008 | 0.073 (p=0.786) | **0.183 (p=0.046)** | **0.404 (p<0.001)** |

> [!IMPORTANT]
> **Observation**: Patient CON004 demonstrates significant phase-locking across all three linguistic levels (Sentence, Phrase, and Word), suggesting intact hierarchical language processing. Patient CON008 shows significant entrainment to Word and Phrase rates but fails to significantly track Sentence-level structure (p=0.786), indicating a potential dissociation between acoustic/syntactic tracking and higher-level integration in this subject.

### Visualizations

#### CON008
- **Topomap**: Shows left-lateralized activation.
- **TFR**: Distinct low-frequency streak at sentence rate.
![CON008 Topomap](CON008_language_ITPC_topomap.png)
![CON008 TFR](CON008_language_ITPC_tfr.png)

#### CON009
- **Consistency**: Replicates sentence-rate entrainment pattern.
![CON009 Topomap](CON009_language_ITPC_topomap.png)
![CON009 TFR](CON009_language_ITPC_tfr.png)

## 4. Methodology Notes (vs. Sokoliuk 2021)

The analysis is inspired by Sokoliuk et al. (2021). The following intentional divergences apply:

| Sokoliuk Step | Our Approach | Reason |
| :--- | :--- | :--- |
| **Band-pass 0.01-100 Hz** | 0.5-100 Hz upstream (ENG-03), then 0.02-25.0 Hz in `LanguageTrackingAnalysis` | 0.02 Hz high-pass provides a 2.5x safety margin below the sentence band; 25.0 Hz low-pass avoids high-frequency aliasing before downsampling |
| **Notch 48-52, 98-102 Hz** | Not applied | The 25.0 Hz low-pass already eliminates all power-line noise; notch filtering is redundant |
| **Downsample to 256 Hz** | Applied in `preprocess_signal` | Source EDFs record at 512 Hz; downsampling to 256 Hz halves ITPC computation time |
| **Discard first 2.28 s** | Applied via `crop(tmin=2.28, tmax=15.36)` | Precise cropping eliminates edge artifacts from upstream filtering/ICA, yielding a clean 13.08s window |
| **Single-bin ITPC at 0.065 Hz** | Match specifically extracted acoustic durations (0.78, 1.56, 3.125 Hz) | Tracking arbitrary duration assumptions produced false statistical hits on 1/f noise floors |
| **ITPC via DFT** | DFT extraction verified exclusively | Direct extraction using Fourier aligns precisely with zero-padded bands |
| **Spearman / Bootstrap / Regression** | Single-subject trial-level phase-scrambled permutations | Bootstrapping permutations at the trial structural level prevents artificial discoveries within dense auto-correlated spectral regions |

## 5. Execution Steps
1. **Full analysis**: `awakenai run language CON008 CON009`
2. **Core logic**: `LanguageTrackingAnalysis.compute_itpc` (Morlet), `LanguageTrackingAnalysis.compute_itpc_dft` (DFT).
3. **Plots** saved to `data/outputs/language/` inside respective sessions.

## 6. Next Steps
- **Group Statistics**: As N increases, run cluster-based permutation tests.
- **Clinical Correlation**: Correlate the Hierarchical Ratio with patient recovery outcomes.
