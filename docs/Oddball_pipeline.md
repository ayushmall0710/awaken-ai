# Event-Related Potential (ERP) Pipeline: Oddball Paradigm

**Project:** EEG Prognostic Data Pipeline - AwakenAI Capstone  
**Priority:** 1 (Critical Path)  
**Timeline:** Jan 10 - Feb 13, 2026 (Milestone Deliverable)  
**Lead:** Member C (Signal Processing)

---

## 📖 Table of Contents
1. [Background & Clinical Context](#background--clinical-context)
2. [The Oddball Paradigm](#the-oddball-paradigm)
3. [Data Structures & Inputs](#data-structures--inputs)
4. [Technical Implementation](#technical-implementation)
5. [Expected Outputs](#expected-outputs)
6. [Quality Control & Validation](#quality-control--validation)
7. [Usage Examples](#usage-examples)
8. [References](#references)

---

## 🧠 Background & Clinical Context

### What is an Event-Related Potential (ERP)?

An **Event-Related Potential (ERP)** is a measured brain response that is the direct result of a specific sensory, cognitive, or motor event. ERPs are extracted from the electroencephalogram (EEG) by time-locked averaging of brain responses to repeated stimuli.

**Key Properties:**
- **Time-Locked:** ERPs are synchronized to the onset of a specific stimulus.
- **Signal Averaging:** By averaging hundreds of trials, random EEG noise cancels out, revealing the consistent neural response.
- **Clinical Value:** ERPs can detect cognitive processing even in patients who are unable to respond behaviorally.

### The P300 Component

The **P300** (or P3) is a specific ERP component that appears as a positive deflection in the EEG signal approximately **300-600 milliseconds** after a rare, unexpected, or task-relevant stimulus.

**Clinical Significance:**
- **Attention & Awareness:** The P300 reflects cognitive processes such as attention allocation, working memory updating, and stimulus evaluation.
- **Prognostic Marker:** In severe brain injury patients, the presence of a P300 response suggests preserved cortical function and may predict better recovery outcomes.
- **Objective Measurement:** Unlike behavioral assessments, the P300 does not require the patient to move or speak.

### Why the Oddball Paradigm?

The **Oddball Paradigm** is the gold-standard method for eliciting the P300 response. It presents two types of auditory stimuli:
1. **Standard Stimuli:** Frequent, predictable tones (~80% of trials).
2. **Deviant Stimuli:** Rare, unpredictable tones (~20% of trials) that differ in pitch.

The P300 is elicited specifically by the **deviant (rare)** stimuli, as the brain detects the "oddball" event.

---

## 🎯 The Oddball Paradigm

### Experimental Design

**Our Implementation:**
- **Stimulus Type:** Auditory beeps delivered via headphones.
- **Standard Tone:** 1000 Hz (frequent).
- **Deviant Tone:** 2000 Hz (rare) - presented ~20% of the time.
- **Inter-Stimulus Interval (ISI):** Approximately 1.3 seconds.
- **Trial Duration:** ~32 seconds per trial block.
- **Total Stimuli per Trial:** 25 beeps (typically 20 standard, 5 deviant).

### Data from CON008 (Example Session)

In the dataset `CON008_2025-08-14_stimulus_results.csv`, we have **4 oddball trials**:

| Trial | Start Time | Duration | Standard | Rare | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1755207543.24 | 32.02s | 23 | 2 | 25 |
| 2 | 1755209507.58 | 31.99s | 22 | 3 | 25 |
| 3 | 1755209771.39 | 31.99s | 19 | 6 | 25 |
| 4 | 1755209805.06 | 32.02s | 20 | 5 | 25 |

**Example Trial Structure:**
```
['standard', 'standard', 'standard', 'standard', 'standard', 'standard', 
 'standard', 'standard', 'standard', 'standard', 'standard', 'rare', 
 'standard', 'standard', 'standard', 'standard', 'standard', 'standard', 
 'standard', 'rare', 'standard', 'standard', 'standard', 'standard', 'rare']
```

**Key Observation:** The sequence is pseudo-randomized to prevent the patient from predicting when the deviant will occur.

---

## 📂 Data Structures & Inputs

### Input 1: CSV Stimulus Log

**Location:** `EEG Project Data/EEG/*_stimulus_results.csv`

**Schema:**
```csv
patient_id,date,trial_type,sentences,start_time,end_time,duration
CON008,2025-08-14,oddball+p,"['standard','standard',...,'rare']",1755207543.24,1755207575.27,32.02
```

**Relevant Fields:**
- `trial_type`: Must equal `"oddball+p"` to identify oddball trials.
- `sentences`: Despite the column name, this contains the stimulus sequence as a list of strings (`'standard'` or `'rare'`).
- `start_time` / `end_time`: Unix timestamps (seconds since epoch) marking the trial boundaries.

### Input 2: EDF File (EEG Recording)

**Location:** `EEG Project Data/EEG/edf/CON008_clipped.EDF`

**Key Channels:**
1. **EEG Channels:** 16-64 channels recording brain activity (e.g., `Fp1`, `Fp2`, `C3`, `C4`, `Pz`, etc.).
2. **DC Audio Channel:** A dedicated channel recording the **audio waveform** played to the patient. This is critical for synchronization.

**Technical Details:**
- **Sampling Rate:** Typically 250-2000 Hz.
- **Format:** EDF/EDF+ (European Data Format) - standard for clinical neurophysiology.
- **Software:** Use **MNE-Python** to load and parse EDF files.

### Synchronization Strategy

**Challenge:** The CSV log provides stimulus timing in Unix timestamps, but the EDF file uses its own internal clock (starting at 0 or the recording start time).

**Solution (Peter's Recommendation):**
1. Use the **DC Audio Channel** in the EDF to identify the exact time of each beep.
2. Cross-reference the audio waveform peaks with the `'rare'` labels in the CSV.
3. Align the two time bases to achieve **sub-50ms precision**.

---

## 🔧 Technical Implementation

### Phase 1: Data Loading & Synchronization

#### Step 1.1: Load EDF File
```python
import mne

# Load the EDF file
edf_path = 'EEG Project Data/EEG/edf/CON008_clipped.EDF'
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Print channel names to identify the DC audio channel
print(raw.info['ch_names'])
```

#### Step 1.2: Load CSV Stimulus Log
```python
import pandas as pd
import ast

# Load CSV
csv_path = 'EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv'
stim_df = pd.read_csv(csv_path)

# Filter for oddball trials
oddball_trials = stim_df[stim_df['trial_type'] == 'oddball+p'].copy()

# Parse the stimulus sequence (stored as a string)
oddball_trials['stim_sequence'] = oddball_trials['sentences'].apply(ast.literal_eval)
```

#### Step 1.3: Synchronize Timestamps
```python
# Extract the DC audio channel
audio_channel_name = 'DC1'  # Adjust based on actual channel name
audio_data = raw.copy().pick_channels([audio_channel_name]).get_data()[0]
sfreq = raw.info['sfreq']  # Sampling frequency

# Detect beeps in the audio channel (example using threshold)
from scipy.signal import find_peaks

# Normalize and threshold to find beep onsets
audio_norm = (audio_data - audio_data.mean()) / audio_data.std()
peaks, _ = find_peaks(audio_norm, height=3, distance=int(sfreq * 0.8))  # Min 0.8s apart

# Convert EDF sample indices to timestamps
edf_times = peaks / sfreq

# Align with CSV Unix timestamps
# (This requires calculating the offset between EDF time 0 and Unix time)
csv_trial_start = oddball_trials.iloc[0]['start_time']
edf_recording_start = raw.info['meas_date'].timestamp()  # Convert to Unix
time_offset = csv_trial_start - edf_recording_start

# Adjust EDF times to Unix
edf_times_unix = edf_times + edf_recording_start
```

**Note:** This is a simplified example. Robust synchronization may require cross-correlation or manual validation.

---

### Phase 2: Epoching & ERP Construction

#### Step 2.1: Identify Deviant Stimuli

For each oddball trial, use the CSV `stim_sequence` to identify which beeps are "rare":

```python
# Example for Trial 1
trial_1 = oddball_trials.iloc[0]
stim_sequence = trial_1['stim_sequence']

# Find indices of 'rare' stimuli
rare_indices = [i for i, stim in enumerate(stim_sequence) if stim == 'rare']
print(f"Rare stimuli at positions: {rare_indices}")  # e.g., [11, 19, 24]
```

#### Step 2.2: Extract Epochs

For each "rare" beep, extract a segment of EEG data from **-200ms to +700ms** relative to the beep onset:

```python
# Define epoch window
tmin = -0.2  # 200ms before stimulus
tmax = 0.7   # 700ms after stimulus

# Select EEG channels of interest (e.g., midline: Fz, Cz, Pz)
picks = mne.pick_types(raw.info, eeg=True, exclude=[audio_channel_name])

# Create events array: [sample_index, 0, event_id]
rare_events = []
for rare_idx in rare_indices:
    # Find the corresponding EDF sample for this beep
    beep_time_unix = trial_1['start_time'] + (rare_idx * 1.3)  # Approx ISI
    beep_sample = int((beep_time_unix - edf_recording_start) * sfreq)
    rare_events.append([beep_sample, 0, 1])  # event_id=1 for 'rare'

rare_events = np.array(rare_events)

# Create epochs
event_id = {'rare': 1}
epochs = mne.Epochs(raw, rare_events, event_id, tmin, tmax, 
                    baseline=(None, 0), picks=picks, preload=True)
```

#### Step 2.3: Average to Create ERP

```python
# Average all epochs to create the ERP
erp = epochs.average()

# Plot the ERP
erp.plot(titles='P300 ERP - Oddball Paradigm')
```

---

### Phase 3: P300 Detection & Quantification

#### Step 3.1: Identify the P300 Peak

The P300 appears as a **positive peak** between **300-600ms** after the stimulus. In neurophysiology convention, positive is typically plotted downward.

```python
# Extract data from a midline electrode (e.g., Pz - parietal)
pz_data = erp.copy().pick_channels(['Pz']).data[0]  # Shape: (n_timepoints,)

# Find peak in the 300-600ms window
time_vector = erp.times  # e.g., [-0.2, -0.19, ..., 0.7]
p300_window = (time_vector >= 0.3) & (time_vector <= 0.6)

# Find the maximum (or minimum if plotting convention inverted)
p300_amplitude = pz_data[p300_window].max()
p300_latency_idx = pz_data[p300_window].argmax()
p300_latency = time_vector[p300_window][p300_latency_idx]

print(f"P300 Amplitude: {p300_amplitude:.2f} µV")
print(f"P300 Latency: {p300_latency * 1000:.0f} ms")
```

#### Step 3.2: Store Features

```python
# Create a feature table
features = {
    'patient_id': 'CON008',
    'trial': 1,
    'p300_amplitude_uV': p300_amplitude,
    'p300_latency_ms': p300_latency * 1000,
    'n_epochs': len(epochs)
}

# Append to a DataFrame
feature_df = pd.DataFrame([features])
feature_df.to_csv('processed/features/p300_features.csv', index=False)
```

---

## 📊 Expected Outputs

### 1. ERP Waveform Plots

**Grand Average ERP** across all deviant stimuli:
- **X-axis:** Time (ms) relative to stimulus onset.
- **Y-axis:** Amplitude (µV).
- **Key Feature:** Clear positive peak around 300-500ms.

**Example Visualization:**
```
Amplitude (µV)
      |
   +5 |                    ____ P300 Peak (~400ms)
      |                   /    \
    0 |__________________|______\___________________
      |                           \__N400?
   -5 |
      +---------------------------------------------> Time (ms)
       -200    0    200   400   600
```

### 2. Feature Table

| patient_id | trial | p300_amplitude_uV | p300_latency_ms | n_epochs |
| :--- | :--- | :--- | :--- | :--- |
| CON008 | 1 | 4.32 | 387 | 2 |
| CON008 | 2 | 3.98 | 412 | 3 |
| CON009 | 1 | 5.12 | 365 | 2 |

### 3. QC Dashboard

**Metrics to Include:**
- Number of deviant stimuli detected per trial.
- Number of epochs rejected due to artifacts.
- Signal-to-Noise Ratio (SNR) estimate.
- Peak-to-peak amplitude consistency across trials.

---

## ✅ Quality Control & Validation

### Validation Checklist

- [ ] **Synchronization Accuracy:** Verify that the audio channel peaks align with CSV timestamps within ±50ms.
- [ ] **Epoch Count:** Confirm that the number of extracted epochs matches the number of "rare" stimuli in the CSV.
- [ ] **Baseline Correction:** Ensure that the pre-stimulus baseline (-200ms to 0ms) is flat (mean ≈ 0 µV).
- [ ] **Artifact Rejection:** Remove epochs with excessive noise (e.g., >100 µV peak-to-peak).
- [ ] **P300 Presence:** For control subjects (awake patients), expect a clear positive peak between 300-600ms.

### Expected Results (Control Data)

For **awake, healthy control subjects**, the P300 should be:
- **Robust:** Present in >80% of control subjects.
- **Amplitude:** 3-10 µV at midline parietal electrodes (Pz, Cz).
- **Latency:** 300-500ms (may vary with age and attention).
- **Topography:** Maximal at **parietal** (back of head) electrodes.

### Red Flags (Indicates Pipeline Error)

- **No Peak Detected:** Check synchronization and epoching logic.
- **Multiple Peaks:** May indicate incorrect baseline or filtering.
- **Negative Peak:** Verify plotting convention (positive up vs. down).
- **Latency <200ms or >700ms:** Likely not the P300.

---

## 💻 Usage Examples

### Example 1: Full Pipeline for One Patient

```python
from erp_pipeline import OddballPipeline

# Initialize pipeline
pipeline = OddballPipeline(
    edf_path='EEG Project Data/EEG/edf/CON008_clipped.EDF',
    csv_path='EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv',
    audio_channel='DC1'
)

# Run full pipeline
pipeline.load_data()
pipeline.synchronize_timestamps()
pipeline.extract_epochs()
pipeline.compute_erp()

# Extract P300 features
features = pipeline.quantify_p300(electrode='Pz')
print(features)

# Save outputs
pipeline.save_erp_plot('outputs/CON008_oddball_ERP.png')
pipeline.save_features('processed/features/CON008_p300.csv')
```

### Example 2: Batch Processing

```python
import os
import pandas as pd

patients = ['CON008', 'CON009', 'CON010']
all_features = []

for patient in patients:
    edf_path = f'EEG Project Data/EEG/edf/{patient}_clipped.EDF'
    csv_files = [f for f in os.listdir('EEG Project Data/EEG/') 
                 if f.startswith(patient) and f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = f'EEG Project Data/EEG/{csv_file}'
        
        try:
            pipeline = OddballPipeline(edf_path, csv_path)
            pipeline.run()
            features = pipeline.quantify_p300()
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {patient}: {e}")

# Compile all features
feature_df = pd.concat(all_features, ignore_index=True)
feature_df.to_csv('processed/features/all_p300_features.csv', index=False)
```

---

## 📚 References

### Scientific Background

1.  **Polich, J. (2007).** "Updating P300: An integrative theory of P3a and P3b." *Clinical Neurophysiology*, 118(10), 2128-2148.
    - Comprehensive review of P300 theory and clinical applications.

2.  **Donchin, E., & Coles, M. G. (1988).** "Is the P300 component a manifestation of context updating?" *Behavioral and Brain Sciences*, 11(3), 357-374.
    - Classic paper on the cognitive significance of P300.

3.  **Luck, S. J. (2014).** *An Introduction to the Event-Related Potential Technique* (2nd ed.). MIT Press.
    - Essential textbook for ERP methodology.

### Clinical Context (Severe Brain Injury)

4.  **Claassen, J., et al. (2019).** "Detection of brain activation in unresponsive patients with acute brain injury." *New England Journal of Medicine*, 380(26), 2497-2505.
    - Demonstrates prognostic value of EEG-based cognitive assessments.

5.  **Sokoliuk, R., et al. (2021).** "Covert speech comprehension predicts recovery from acute unresponsive states." *Annals of Neurology*, 89(4), 646-656.
    - Shows that language processing EEG markers predict recovery.

### MNE-Python Documentation

6.  **MNE-Python Tutorials:** https://mne.tools/stable/auto_tutorials/index.html
    - Official tutorials for EEG/ERP analysis.

7.  **Epoching Guide:** https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html
    - Detailed guide on creating epochs from raw data.

---

## 🚀 Next Steps

1.  **Week 1-2 (Jan 10-24):** Implement synchronization logic and validate on CON008.
2.  **Week 3-4 (Jan 25-Feb 08):** Scale to all patients, implement QC dashboard.
3.  **Week 5 (Feb 09-13):** **MILESTONE DELIVERABLE** - Present Grand Average ERPs and feature table.

---

**Last Updated:** December 10, 2025  
**Author:** AwakenAI Capstone Team  
**Contact:** [Team Communication Channel]
