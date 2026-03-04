# ENG-05: Analysis Plan - Language Tracking (ITPC)

**Status:** Completed & Verified
**Dependencies:** ENG-05 (Language Optimization), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Context:** Based on `docs/language_tracking.md`
**Objective:** Quantify **covert speech comprehension** by measuring **neural entrainment** (phase-locking) to the hierarchical structure of speech (specifically sentence rate ~0.065 Hz).

## 1. Scientific Background

Patients with disorders of consciousness may still process language. Validating this requires detecting neural responses that track the **slow temporal envelope** of sentences, distinct from low-level acoustic processing (word rate).
- **Target Bands:** Sentence (~0.78 Hz), Phrase (~1.56 Hz), and Word (~3.125 Hz).
- **Metric:** Band-averaged Inter-Trial Phase Coherence (ITPC), validated against trial-level phase-scrambled permutation testing.

## 2. Methodology & Implementation

### Signal Processing
- **Source**: `LanguageTrackingAnalysis` (utilizing `ArtifactRejector` clean epochs).
- **Filter**: 0.5 - 30 Hz (Preserves low-frequency Delta band).
- **Montage**: Standard 10-20.
- **Channels**: Left Hemisphere Focus (6 channels: F7, T7, P7, F3, C3, P3) is default, with explicit validation of focus parameter (`LH`, `RH`, or `Clinical`).
- **Epochs**: ~16s segments covering full sentence duration.

### Time-Frequency Analysis
- **Method**: Morlet Wavelets (`tfr_morlet`).
- **Frequencies**: Log-spaced 0.05 - 2.0 Hz (40 bins).
- **Cycles**: Adaptive (min 0.5 cycles for low freq to fit 16s epoch).

### Metric Extraction: Band-Averaged ITPC

ITPC is extracted by **averaging over all frequency bins within each band**, allowing robustness against slight biological shifting.

| Band | Frequencies | Target Center |
| :--- | :--- | :--- |
| Sentence | 0.71 - 0.85 Hz | 0.78 Hz |
| Phrase | 1.49 - 1.63 Hz | 1.56 Hz |
| Word | 3.05 - 3.20 Hz | 3.125 Hz |

### DFT Zero-Padding

The DFT path zero-pads the time series to achieve 0.01 Hz frequency resolution before computing the FFT. Without padding, the 16s epoch raw resolution (0.0625 Hz) causes the 0.065 Hz sentence target to fall between bins (4% error). Zero-padding allows band-selection to correctly include the appropriate bins in both bands.

### Statistical Validation
To distinguish high-level comprehension from basic acoustic processing, we employ **trial-level phase-scrambled permutation testing**.
- **Hypothesis**: True neural tracking will exhibit ITPC significantly greater than empirical chance representations of 1/f noise evaluated per band (`p < 0.05`).

## 3. Verification Results

### Quantitative Findings

68-69 trials per patient (LH focus: Fp1, F7, T7, F3, C3, P3).

**Global Exact DFT Extracted ITPC (Zero-padded)**

| Patient | Sentence ITPC (p-val) | Phrase ITPC (p-val) | Word ITPC (p-val) |
| :--- | :--- | :--- | :--- |
| CON008 | 0.083 (p=0.718) | 0.143 (p=0.172) | **0.415 (p<0.01)** |
| CON009 | 0.057 (p=0.984) | 0.135 (p=0.165) | 0.157 (p=0.063) |

> [!WARNING]
> **Paradigm Shift Result**: After correcting the stimulus presentation frequency tracking values, both patients show failure to phase-lock significantly at the hierarchical sentence level. Patient CON008 shows powerful, massive entrainment to the acoustic structural word boundaries (p=0.000). The earlier artificial ratios were driven primarily by extracting very low-frequency 1/f red noise without permutation-level correction, rendering them invalid.

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
| **Band-pass 0.01-100 Hz** | 0.5-100 Hz upstream (ENG-03), then 0.5-30 Hz in `LanguageTrackingAnalysis` | 0.5 Hz high-pass improves ICA stability while still capturing 0.065 Hz sentence rate; 30 Hz low-pass scopes analysis to language frequencies |
| **Notch 48-52, 98-102 Hz** | Not applied | The 30 Hz low-pass already eliminates all power-line noise; notch filtering is redundant |
| **Downsample to 256 Hz** | Applied in `preprocess_signal` | Source EDFs record at 512 Hz; downsampling to 256 Hz halves ITPC computation time |
| **Discard first 2.28 s** | Not applied | Our epochs start at `tmin=0.0` (stimulus onset) with 16s window -- no pre-stimulus period to discard |
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
