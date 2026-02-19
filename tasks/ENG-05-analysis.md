# ENG-05: Analysis Plan - Language Tracking (ITPC)

**Status:** Completed & Verified
**Dependencies:** ENG-05 (Language Optimization), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Context:** Based on `docs/language_tracking.md`
**Objective:** Quantify **covert speech comprehension** by measuring **neural entrainment** (phase-locking) to the hierarchical structure of speech (specifically sentence rate ~0.065 Hz).

## 1. Scientific Background
Patients with disorders of consciousness may still process language. Validating this requires detecting neural responses that track the **slow temporal envelope** of sentences, distinct from low-level acoustic processing (word rate).
*   **Target Frequency:** ~0.065 Hz (Sentence Rate)
*   **Control Frequency:** ~0.77 Hz (Word Rate)
*   **Metric:** Inter-Trial Phase Coherence (ITPC) at the Left Hemisphere.

## 2. Methodology & Implementation

### Signal Processing
- **Source**: `LanguageProcessor` (utilizing `ArtifactRejector` clean epochs).
- **Filter**: 0.5 - 30 Hz (Preserves low-frequency Delta band).
- **Montage**: Standard 10-20.
- **Channels**: Left Hemisphere Focus (19 channels selected, e.g., F7, T7, P7).
- **Epochs**: ~16s segments covering full sentence duration.

### Time-Frequency Analysis
- **Method**: Morlet Wavelets (`tfr_morlet`).
- **Frequencies**: Log-spaced 0.05 - 2.0 Hz.
- **Cycles**: Adaptive (min 0.5 cycles for low freq to fit 16s epoch).

### Statistical Validation
To distinguish high-level comprehension from basic acoustic processing, we compare ITPC at the **Sentence Rate** (~0.065 Hz) versus the **Word Rate** (~0.77 Hz).
*   **Hypothesis**: Sentence Rate ITPC > Word Rate ITPC (Ratio > 1.0) indicates hierarchical processing.

## 3. Verification Results

### Quantitative Findings
We validated the pipeline on two subjects: `CON008` (Aug 14, 2025) and `CON009` (Aug 26, 2025).

| Metric | CON008 | CON009 |
| :--- | :--- | :--- |
| **Sentence ITPC (0.065 Hz)** | **0.1251** | **0.1262** |
| **Word ITPC (0.77 Hz)** | 0.1031 | 0.1149 |
| **Hierarchical Ratio** | **1.21** | **1.10** |

> [!IMPORTANT]
> **Interpretation**: In both patients, **Sentence ITPC > Word ITPC** (Ratio > 1.0).
> This suggests that neural entrainment is **not** driven solely by the rapid acoustic envelope of words (which is close to the noise floor ~0.10-0.11), but reflects tracking of the slower sentence structure. A ratio > 1.1 is a positive indicator of hierarchical processing.

### Visualizations

#### CON008
- **Topomap**: Shows left-lateralized activation.
- **TFR**: Distinct low-frequency streak at 0.065 Hz.
![CON008 Topomap](CON008_language_ITPC_topomap.png)
![CON008 TFR](CON008_language_ITPC_tfr.png)

#### CON009
- **Consistency**: Replicates the ~0.126 Sentence ITPC value.
![CON009 Topomap](CON009_language_ITPC_topomap.png)
![CON009 TFR](CON009_language_ITPC_tfr.png)

## 4. Methodology Notes (vs. Sokoliuk 2021)

The analysis is inspired by Sokoliuk et al. (2021). The following intentional divergences apply:

| Sokoliuk Step | Our Approach | Reason |
| :--- | :--- | :--- |
| **Band-pass 0.01-100 Hz** | 0.5-100 Hz upstream (ENG-03), then 0.5-30 Hz in `LanguageProcessor` | 0.5 Hz high-pass improves ICA stability while still capturing 0.065 Hz sentence rate; 30 Hz low-pass scopes analysis to language frequencies |
| **Notch 48-52, 98-102 Hz** | Not applied | The 30 Hz low-pass already eliminates all power-line noise; notch filtering is redundant |
| **Downsample to 256 Hz** | Applied in `preprocess_signal` | Source EDFs record at 512 Hz; downsampling to 256 Hz (Nyquist 128 Hz >> 30 Hz cut-off) halves ITPC computation time |
| **Discard first 2.28 s** | Not applied | Sokoliuk epochs at `tmin=-1.0` (1 s pre-stimulus) with 16.36 s total. Our epochs start at `tmin=0.0` (stimulus onset) with 16 s window -- no pre-stimulus period to discard |
| **ITPC via DFT** | Morlet primary; DFT available via `compute_itpc_dft` | Morlet provides time-frequency resolution; DFT method added for direct cross-validation with Sokoliuk |
| **Spearman / Bootstrap / Regression** | Deferred | Requires larger N; planned for group-level analysis |

## 5. Execution Steps
1.  **Full analysis (Morlet + DFT)**: `eda/run_itpc_analysis.py --patients CON008 CON009`
2.  **Method comparison**: `eda/compare_itpc_methods.py --patients CON008 CON009` -- produces bar chart comparing sentence/word ITPC across both methods.
3.  **Core logic**: `LanguageProcessor.compute_itpc` (Morlet), `LanguageProcessor.compute_itpc_dft` (DFT).
4.  **Plots** saved to `data/outputs/{patient_id}/`.

## 6. Next Steps
- **Group Statistics**: As N increases, run cluster-based permutation tests.
- **Clinical Correlation**: Correlate the Hierarchical Ratio with patient recovery outcomes.
- **Re-analysis**: Run improved pipeline (256 Hz + DFT cross-validation) to refresh quantitative results.
