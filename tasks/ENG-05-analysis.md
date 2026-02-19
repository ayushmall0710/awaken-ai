# ENG-05: Analysis Plan - Language Tracking (ITPC)

**Status:** Planned
**Dependencies:** ENG-05 (Language Optimization - Completed), ENG-03 (Artifact Rejection), ENG-02 (Timestamp Alignment)
**Context:** Based on `docs/language_tracking.md`
**Objective:** Quantify **covert speech comprehension** by measuring **neural entrainment** (phase-locking) to the hierarchical structure of speech (specifically sentence rate ~0.065 Hz).

## 1. Scientific Background
Patients with disorders of consciousness may still process language. Validating this requires detecting neural responses that track the **slow temporal envelope** of sentences, distinct from low-level acoustic processing (word rate).
*   **Target Frequency:** ~0.065 Hz (1 sentence / 15.5s)
*   **Control Frequency:** ~0.77 Hz (Word rate)
*   **metric:** Inter-Trial Phase Coherence (ITPC)

## 2. Inputs (from ENG-05 Pipeline)
- **Neural Data**: `mne.Epochs`
    - **Source**: `LanguageProcessor.process_patient`
    - **Filter**: 0.5 - 30 Hz (Retains Delta band)
    - **Channels**: Reduced "Clinical 20" set with Left Hemisphere focus (e.g., F7, T7, P7).
    - **Epoch Length**: ~16s (covering full sentence duration).

## 3. Implementation Strategy

### Phase 1: Time-Frequency Representation (TFR)
Decompose EEG signals into time-frequency components using **Morlet Wavelets**.
*   **Frequencies**: Log-spaced from 0.05 Hz to 2.0 Hz.
*   **Cycles**: Adaptive (e.g., `freq / 2`) to balance temporal/spectral resolution.
*   **Key Interest**: The bin closest to **0.065 Hz**.

```python
# Pseudocode
freqs = np.logspace(np.log10(0.05), np.log10(2), num=30)
tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=freqs/2, return_itc=False)
```

### Phase 2: Compute ITPC
Calculate phase consistency across trials at each time-frequency point.
*   **Formula**: `ITPC(f, t) = |1/N * Σ exp(i * φ_n(f, t))|`
*   **Interpretation**:
    *   **0**: Random phase (no tracking).
    *   **1**: Perfect phase locking (strong entrainment).

### Phase 3: Statistical Validation
Determine if the observed ITPC is significantly different from chance/baseline.
1.  **Baseline Comparison**: Compare Sentence Rate ITPC vs. specific baseline window or shuffled data.
2.  **Permutation Testing**:
    *   Shuffle trial phases to build a null distribution.
    *   Cluster-based permutation test (MNE) for robust statistical thresholding.

## 4. Expected Outputs & Visualization

### A. ITPC Topomap
*   **Goal**: Visualize spatial distribution of entrainment.
*   **Expectation**: High ITPC over **Left Temporal/Frontal** regions (T7, F7, F3) for language processing.
*   **Output**: `outputs/{patient_id}_language_ITPC_topomap.png`

### B. Time-Frequency Plot
*   **Goal**: Show ITPC evolution over the trial duration.
*   **Expectation**: Sustained ITPC at 0.065 Hz band throughout the sentence presentation.
*   **Output**: `outputs/{patient_id}_language_ITPC_tfr.png`

### C. Feature Table
*   **Format**: CSV
*   **Columns**: `patient_id`, `n_trials`, `itpc_sentence_mean`, `itpc_word_mean`, `p_value`.

## 5. Execution Steps
1.  **Create Script**: `eda/run_itpc_analysis.py`
    *   Load epochs using `LanguageProcessor`.
    *   Compute TFR & ITPC.
    *   Run permutation stats.
    *   Generate plots.
2.  **Run on CON008**: Validate LH dominance and sentence tracking.
3.  **Run on CON009**: Replicate findings.

## 6. Success Criteria
- [ ] **Data Quality**: 0.5 Hz filter preserved enough low-frequency signal for 0.065 Hz analysis (Requires checking effective resolution after Morlet transform).
- [ ] **Significance**: `p < 0.05` for ITPC at sentence rate in LH channels.
- [ ] **differentiation**: Sentence Rate ITPC > Word Rate ITPC (evidence of higher-level comprehension).
