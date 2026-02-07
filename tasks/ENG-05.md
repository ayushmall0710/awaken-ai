# ENG-05: Language Optimization
**Data Validation:** Verified on CON008 (199/210 epochs, 6/7 checks), CON009 (553/574 epochs, 5/7 checks)
**Category:** Signal Processing / Optimization
**Dependencies:** ENG-01 (Unified Data Loader), ENG-02 (Timestamp Alignment)
**Target Branch:** `feature/ENG-05-lang-opt` (rebased on `main`)

## 1. Overview
The goal of **ENG-05** is to create a specialized pipeline for the **Language Tracking Paradigm**. This involves isolating pertinent trial segments and, crucially, identifying a reduced set of electrodes (~20) that maximize the signal-to-noise ratio for detecting neural entrainment to speech, focusing primarily on the **Left Hemisphere**.

This optimization is critical for eventual clinical deployment where high-density caps are impractical.

## 2. Objectives
1.  **Trial Isolation**: Accurately extract language trials using the `UnifiedDataLoader`.
2.  **Electrode Selection**: Implement logic to subset channels to a "Clinical 20" set, with a bias towards Left Hemisphere (Language Dominant) regions.
3.  **Signal Optimization**: Apply specific filtering or re-referencing (e.g., CAR within LH) to enhance speech envelope tracking.

## 3. Implementation Details

### Validated Workflow
- **Data Loading**: Uses `UnifiedDataLoader` to fetch patient trials and load EDFs.
- **Timestamp Alignment**: Relies on `src/data_processing/timestamp_aligner.py` to detect timezone offsets and align events. `TimestampAligner` outputs `event_start_edf` (EDF-relative seconds) in enriched events so downstream consumers use aligned times directly without any timezone conversion.
- **Language Processor**: `src/data_processing/language_optimization.py`
    - **Channel Selection**: Implements `select_optimal_channels` prioritizing LH focus (F7, T7, P7, F3, C3, P3) + Clinical 20 fallback.
    - **Filtering**: Applies 0.5-30Hz bandpass filter to remove drift and high-freq noise.
    - **Epoching**: Extracts 16s epochs using `event_start_edf` from TimestampAligner output. Converts directly to MNE sample indices without timezone logic.

### Key Decisions
- **No Duplicate Timezone Logic**: `LanguageProcessor` is a pure consumer of `TimestampAligner` output. Timezone detection and Unix-to-EDF conversion are handled entirely by the aligner, which exposes `event_start_edf` in its enriched events. The processor reads this field directly to compute epoch sample offsets.

### Verification
- **Unit Tests**: `tests/test_language_optimization.py` covers initialization, channel selection, filtering, and end-to-end processing.
- **Demo Script**: `eda/demo_language_optimization.py` validates the pipeline on real patient data (CON008, CON009), confirming successful channel reduction (e.g., 64 -> 19 channels) and correct epoch alignment.

## 4. Definition of Done
- [x] `src/data_processing/language_optimization.py` created.
- [x] Function to return `mne.Epochs` or `mne.Raw` array restricted to optimal channels.
- [x] Unit tests in `tests/test_language_optimization.py` using `conftest.py` mock data.
- [x] Demonstration notebook or script outputting the channel selection results for a sample patient (CON008).

## 5. Next Steps
- **Integration**: The `LanguageProcessor` is ready to be integrated into the broader analysis pipeline for feature extraction (e.g., TRF analysis).
- **Deployment**: The optimized channel set reduces data volume, facilitating faster processing for potential real-time applications.
