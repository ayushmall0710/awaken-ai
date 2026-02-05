# ENG-02: Timestamp Alignment Implementation

## Overview

Implemented timestamp alignment to synchronize experimental event logs (stimuli) with EEG recordings (EDF files) using a DC channel signal. The alignment corrects for clock drift and start-time discrepancies between the recording device and the stimulus presentation computer.

## Implementation Details

### 1. Audio Envelope Correlation (Language Trials)

- **Problem**: Default cross-correlation between DC signal and raw audio files yielded very low scores (~15%) because the DC channel records the _amplitude envelope_ of the audio, not the waveform itself.
- **Solution**: Compute the **Hilbert envelope** of the source audio files before correlation.
- **Result**: Correlation scores improved drastically (15% -> ~80-100% for high-confidence matches).

### 2. Peak Detection (Oddball/Beep Trials)

- **Problem**: Beeps are short pulses mixed with instruction audio, requiring precise separation.
- **Solution**:
  - **Instruction Masking**: Implemented `_detect_instruction_end` to find and mask the instruction using cross-correlation, effectively isolating the beep sequence.
  - **Highpass Filtering**: Applied a 50Hz highpass filter to remove baseline drift and low-frequency noise.
  - **Envelope Peak Detection**: Used `scipy.signal.find_peaks` on the signal envelope with adaptive prominence (0.5 \* std_dev) to robustly identify beep onsets.
- **Result**: ~99-100% alignment success for oddball trials, with robust handling of instruction prompts and noise.

### 3. Data Schema & Validation

- **Output**: Per-trial parquet files with enriched `sentences` list containing:
  - `event_start`, `event_end` (aligned unix timestamps)
  - `correlation_score` (confidence metric)
  - `peak_amplitude` (for beeps)
- **Validation**: Added a robust `validate()` method that provides alignment rates broken down by trial type and identifies specific trials with poor performance.

### 4. Code Optimization

- **Audio Caching**: Implemented `lru_cache` in `UnifiedDataLoader` to cache stimulus audio files, eliminating redundant disk I/O.

## Why this approach?

- **Robustness**: Aligning _events_ individually (within a trial window) is more robust to drift than aligning just the trial start.
- **Verification**: We store the correlation score for every single event, allowing us to filter out low-confidence alignments downstream.

## Remaining Issues & Next Steps

1. **Event ID Mismatch**:
   - Analysis revealed that the `event_id` in logs does not always match the played audio file (e.g., Event 18 -> `lang13.wav`).
   - **Fix**: Implement a "brute-force" correlation check (compare signal against all 34 audio templates) or a hash-based lookup to identify the true audio file played.

2. **Missing Audio Files**:
   - Audio files for Event IDs > 34 are missing from the `sentences` directory in 3 patients - `CON003`, `CON006`, `test_new`.
   - **Observation**: These `sentences` match with other available `lang` files in no particular order.

3. **Low Correlation Events**:
   - Some events still show low correlation (<30%), likely due to the ID mismatch issue mentioned above.

4. **Timezone Offset Logic**:
   - The current implementation of `_detect_timezone_offset` in `TimestampAligner` uses a heuristic approach.
   - It compares the EDF start timestamp (converted to unix) with the first trial's start timestamp.
   - If the difference is greater than 1 hour, it rounds to the nearest hour and applies it as a correction.
   - **Note**: This logic is simplistic but sufficient for our current data. Corner cases (DST changes, etc.) would result in **failed alignment** (0 correlation) rather than incorrect data, so they are easily detectable if they occur.
