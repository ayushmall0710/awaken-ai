# ENG-04: Command Following Analysis

**Task ID:** ENG-04  
**Assignee:** Aaditya Chopra

---

## Objective

Develop a complete analysis pipeline to detect **Covert Command Following (CMD)** in acute brain injury patients using Event-Related Desynchronization (ERD) in motor cortex.

## Physiological Basis

### The ERD Phenomenon

**Alpha (8-13 Hz) and Beta (13-30 Hz)** oscillations are the "idle rhythms" of the sensorimotor cortex:

- **At rest:** High power in alpha/beta → cortex is "idling", neurons firing synchronously
- **During motor imagery/execution:** Power **DROPS** → neurons desynchronize to perform computation
- **After movement:** Beta power rebounds above baseline (post-movement beta rebound)

**Analogy:**

```
Synchronized neurons (high power) → Can't process information independently
Desynchronized neurons (low power) → Individual neurons compute different aspects of movement
```

### Motor Command Paradigm

**Task:** Patients hear alternating audio commands within a ~200s trial:

- **"Keep opening and closing your right/left hand"** (~13s each) — motor imagery
- **"Stop"** (~13s each) — rest

**Neural Response:**
When you imagine moving your right hand:

- **Left motor cortex (C3)** activates → contralateral control
- **Premotor areas (SMA, PMC)** show bilateral activation
- **Parietal cortex** integrates sensory feedback (even imagined)

**Signal:** Motor imagery causes **Event-Related Desynchronization (ERD)** (power drop) in:

- **Alpha (8-13 Hz)** - PRIMARY ERD MARKER
- **Beta (13-30 Hz)** - SECONDARY ERD MARKER

**Goal:** Detect significantly lower Alpha/Beta power during "keep" (imagery) blocks vs "stop" (rest) blocks.

### Methodology Adaptation

**Claassen et al. (2019)** used **64-128 channel high-density EEG** systems.

**This implementation** adapts the methodology for **~20 channel clinical EEG** (standard ICU montage):

- **Same core channels:** C3, C4, Cz (present in both systems)
- **Same analysis approach:** ERD detection in Alpha/Beta bands
- **Trade-off:** Lower spatial resolution, but preserves ERD detection capability
- **Advantage:** More feasible for routine clinical use in ICU settings

---

## Implementation Design

### Data Flow

```
.fif (ENG-03)  ──→  mne.Epochs  ──→  .crop()  ──→  .filter()  ──→  Welch PSD  ──→  ERD (dB)
                        ↑                ↑
              Aligned events      Unix → EDF → epoch-relative
              (ENG-02 sentences)   time conversion
```

### Why Paired Epochs (Not Separate Baselines)

Earlier designs used separate control trials or pre-stimulus periods as baselines. The current design uses **paired keep-stop epochs** from within the same trial:

- **Same brain state:** Both epochs come from the same ~200s trial, so non-stationarities (electrode drift, arousal changes) are matched.
- **Built-in baseline:** The "stop" (rest) command IS the baseline for its adjacent "keep" (imagery) command.
- **No external dependency:** No need for separate control trials or arbitrary pre-stimulus windows.
- **Per-pair statistics:** Each keep-stop pair yields one ERD sample, enabling proper paired statistical testing.

### Epoch Structure

```
Trial epoch (~200s):
|-- keep cmd --|-- stop cmd --|-- keep cmd --|-- stop cmd --| ...
    ~13s           ~13s           ~13s           ~13s

What we CROP for each command segment:
|audio|======= usable brain signal ========|trim|
 0.5s           ~12.4s analyzed              0.1s
 (skip onset)                            (skip tail)
```

- **EPOCH_TRIM_START = 0.5s:** Skips the audio command onset — the brain needs time to process the instruction before motor imagery begins.
- **EPOCH_TRIM_END = 0.1s:** Trims tail transition artifacts.
- **MIN_EPOCH_DURATION = 1.5s:** Segments shorter than this after trimming are dropped (insufficient data for reliable PSD).

### ERD in dB Scale

```
ERD_dB = 10 * log10(keep_power) - 10 * log10(stop_power)
```

- **Negative ERD_dB** → desynchronization during imagery → patient is following commands
- **dB scale** → normalizes across patients with different absolute power levels
- **Per-pair computation** → each keep-stop pair yields one ERD value

### Why mne.Epochs (Not Raw or NumPy Arrays)

The pipeline works directly with `mne.Epochs` objects loaded from ENG-03 `.fif` files:

- **`.crop(tmin, tmax)`** — extracts individual command segments without converting to Raw or NumPy
- **`.filter()`** — applies bandpass using MNE's built-in FIR (zero-phase, proper edge handling)
- **`.get_data()`** — called only at PSD computation time (Welch needs the numpy array)
- **`.pick(channels)`** — selects ROI channels natively

No intermediate format conversions. The Epochs object carries all metadata (sfreq, ch_names, meas_date) so no need to pass these around separately.

---

## Implementation Requirements

### 1. Data Selection (UnifiedDataLoader)

- Load `right_command` and `left_command` trials from aligned events.
- Load clean epochs per `(date, trial_type)` from ENG-03 `.fif` files via `load_clean_epochs()`.
- `groupby(["date", "trial_type"])` ensures each `.fif` is loaded only once.
- Command positions (keep/stop boundaries) come from the `sentences` field in aligned events (ENG-02 audio correlation output).

### 2. Event Deduplication

Each ~13s audio position produces TWO detections in ENG-02 (both "keep" and "stop" templates match the same audio). `deduplicate_and_label()` handles this:

1. Merge events within 1s of each other into a single position
2. Assign alternating keep/stop labels
3. Filter out entries missing `event_start`/`event_end`

> **TODO:** The alternating keep/stop assumption is a placeholder. Actual command sequence needs to be confirmed with Prof/Alex.

### 3. Preprocessing

**Bandpass Filter:** 8-30 Hz (FIR filter) to isolate Alpha/Beta bands.

**Why 8-30 Hz?**

- **< 4 Hz:** Eye blinks, slow drifts (artifacts)
- **4-8 Hz (Theta):** Not primarily motor-related
- **8-13 Hz (Alpha):** Sensorimotor idle rhythm — **PRIMARY ERD MARKER**
- **13-30 Hz (Beta):** Motor maintenance and rebound — **SECONDARY ERD MARKER**
- **> 30 Hz (Gamma):** Muscle artifacts, not relevant

### 4. Spectral Analysis (Welch PSD)

For each ~12.4s segment, Welch's method provides stable power estimates by averaging overlapping periodograms.

### 5. Statistical Testing

**Three-layer testing:**

| Test                        | Purpose                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Paired one-sided t-test** | H₁: keep_power < stop_power (ERD < 0). Paired because each keep has a matched stop. |
| **Benjamini-Hochberg FDR**  | Controls false discovery rate across multiple channels × bands per side.            |
| **Mixed effects model**     | `ERD ~ 1 + (1                                                                       | trial)` — accounts for within-trial correlation (epochs from the same trial share brain state). |

**Effect size:** Cohen's d quantifies how large the desynchronization is (not just whether it's significant).

### 6. Classification (CMD+/CMD-)

**Contralateral-first approach:**

| Criterion     | Threshold                                                                   |
| ------------- | --------------------------------------------------------------------------- |
| Channel       | Must be **contralateral** to the commanded hand (C3 for right, C4 for left) |
| p-value       | FDR-corrected p < 0.05                                                      |
| ERD magnitude | ERD_dB < -1.0 dB                                                            |
| Effect size   | \|Cohen's d\| > 0.5                                                         |

**CMD+** if any contralateral channel × band combination meets all criteria.

---

## Usage

```python
from src.pipelines.command_following_analysis import CommandFollowingAnalysis

analysis = CommandFollowingAnalysis()

# Single call runs the full pipeline: load → preprocess → ERD → classify
erd_df = analysis.run("CON008")

# With CMD+/- classification summary
erd_df, summary = analysis.run("CON008", summary=True)
print(summary["cmd_status"])  # "CMD+" or "CMD-"
```

**ERD DataFrame columns:**
`side`, `channel`, `band`, `erd_dB`, `erd_std`, `n_pairs`, `p_value_raw`, `cohens_d`, `p_mixed`, `is_contralateral`, `p_value`, `significant`

---

## Clinical Relevance

### Covert Command Following Detection

**Problem:** 15% of unresponsive ICU patients can follow commands with their brain but not behaviorally (Claassen 2019)

**Solution:** EEG-based ERD detection reveals hidden consciousness

**Impact:**

- 44% of CMD+ patients recover vs. 14% of CMD-
- Informs life-support and rehabilitation decisions
- Bedside-compatible (no fMRI needed)

### EEG System Configuration

**Channel Count:** ~20 EEG channels (standard clinical montage)

- Total channels: 50 (includes EMG, ECG, DC, respiratory sensors)
- EEG channels: C3, C4, Cz, F3, F4, F7, F8, Fz, Fp1, Fp2, Fpz, P3, P4, Pz, T7, T8, P7, P8, O1, O2

**Primary Motor Channels (ROI):**

- **C3:** Left motor cortex (controls right hand) - **Primary**
- **C4:** Right motor cortex (controls left hand) - **Primary**
- **Cz:** Midline supplementary motor area - **Primary**

**Extended Motor Network (Optional):**

- **FC3, FC4:** Frontal-central (premotor) - _Not available in current system_
- **CP3, CP4:** Central-parietal (sensorimotor integration) - _Not available in current system_
- **F3, F4:** Frontal (motor planning) - Available
- **P3, P4:** Parietal (sensory processing) - Available

**Comparison to Claassen (2019):**

- Claassen used 64-128 channels for higher spatial resolution
- Our 20-channel system captures the essential motor cortex signals
- C3/C4/Cz are gold-standard channels present in both systems
- Lower density reduces spatial detail but preserves ERD detection capability

---

## References

- **Claassen et al. (2019)** - Detection of brain activation in unresponsive patients with acute brain injury. _New England Journal of Medicine_
- **Pfurtscheller & Lopes da Silva (1999)** - Event-related EEG/MEG synchronization and desynchronization
- **MNE-Python Documentation** - Cluster-based permutation testing for EEG
