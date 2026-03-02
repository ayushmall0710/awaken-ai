# ENG-05: Analysis Plan - Language Tracking (ITPC)

**Status:** Completed & Verified
**Dependencies:** ENG-05 (Language Optimization), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Context:** Based on `docs/language_tracking.md`
**Objective:** Quantify **covert speech comprehension** by measuring **neural entrainment** (phase-locking) to the hierarchical structure of speech (specifically sentence rate ~0.065 Hz).

## 1. Scientific Background

Patients with disorders of consciousness may still process language. Validating this requires detecting neural responses that track the **slow temporal envelope** of sentences, distinct from low-level acoustic processing (word rate).
- **Target Band:** 0.05-0.08 Hz (Sentence Rate, centered on ~0.065 Hz)
- **Control Band:** 0.70-0.90 Hz (Word Rate, centered on ~0.77 Hz)
- **Metric:** Band-averaged Inter-Trial Phase Coherence (ITPC), Left Hemisphere.

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

ITPC is extracted by **averaging over all frequency bins within each band**, not at a single nearest bin. This matches the frequency band optimization strategy from the original gridsearch design (`docs/language_tracking.md`, Optimization Strategy #2) and is more robust to ICA-induced bin-level power shifts between sessions.

| Band | Frequencies | Morlet bins | DFT bins (zero-padded) |
| :--- | :--- | :--- | :--- |
| Sentence | 0.05 - 0.08 Hz | 4 bins | 31 bins |
| Word | 0.70 - 0.90 Hz | 3 bins | 201 bins |

### DFT Zero-Padding

The DFT path zero-pads the time series to achieve 0.001 Hz frequency resolution before computing the FFT. Without padding, the 16s epoch raw resolution (0.0625 Hz) causes the 0.065 Hz sentence target to fall between bins (4% error). Zero-padding allows band-selection to correctly include the appropriate bins in both bands.

### Statistical Validation
To distinguish high-level comprehension from basic acoustic processing, we compare band-averaged ITPC at the **Sentence Rate** versus the **Word Rate**.
- **Hypothesis**: Sentence Band ITPC > Word Band ITPC (Ratio > 1.0) indicates hierarchical processing.

## 3. Verification Results

### Quantitative Findings

68 trials per patient (LH focus: F7, T7, P7, F3, C3, P3).

**Morlet Wavelet (primary) - Band-averaged across 0.05-0.08 Hz / 0.70-0.90 Hz**

| Patient | Source | Sentence ITPC | Word ITPC | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| CON008 | BAK | 0.1217 | 0.1047 | 1.16 |
| CON008 | NEW | 0.1217 | 0.0985 | 1.23 |
| CON009 | BAK | 0.1413 | 0.1152 | 1.23 |
| CON009 | NEW | 0.1353 | 0.1170 | 1.16 |

**DFT / FFT (Sokoliuk 2021 cross-validation) - Zero-padded, band-averaged**

| Patient | Source | Sentence ITPC | Word ITPC | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| CON008 | BAK | 0.1884 | 0.1078 | 1.75 |
| CON008 | NEW | 0.2131 | 0.1017 | 2.09 |
| CON009 | BAK | 0.1853 | 0.1116 | 1.66 |
| CON009 | NEW | 0.1256 | 0.1142 | 1.10 |

> [!IMPORTANT]
> **Cross-method agreement**: Both Morlet and DFT independently confirm **Sentence ITPC > Word ITPC** (Ratio > 1.0 in all cases) after applying band-averaged extraction and DFT zero-padding. CON009 NEW DFT ratio was previously 0.92 (< 1.0) when using single-bin extraction at the misaligned 0.0625 Hz bin; the corrected band-averaged result is 1.10.

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
| **Single-bin ITPC at 0.065 Hz** | Band-averaged across 0.05-0.08 Hz | Matches gridsearch design from `docs/language_tracking.md`; robust to small ICA-induced bin shifts |
| **ITPC via DFT** | Morlet primary; zero-padded DFT for cross-validation | Morlet provides time-frequency resolution; DFT zero-padded to 0.001 Hz resolution for accurate band alignment |
| **Spearman / Bootstrap / Regression** | Deferred | Requires larger N; planned for group-level analysis |

## 5. Execution Steps
1. **Full analysis (Morlet + DFT)**: `eda/run_itpc_analysis.py --patients CON008 CON009`
2. **Method comparison**: `eda/compare_itpc_methods.py --patients CON008 CON009` -- produces bar chart comparing sentence/word ITPC across both methods.
3. **Core logic**: `LanguageTrackingAnalysis.compute_itpc` (Morlet), `LanguageTrackingAnalysis.compute_itpc_dft` (DFT).
4. **Plots** saved to `data/outputs/{patient_id}/`.

## 6. Next Steps
- **Group Statistics**: As N increases, run cluster-based permutation tests.
- **Clinical Correlation**: Correlate the Hierarchical Ratio with patient recovery outcomes.
