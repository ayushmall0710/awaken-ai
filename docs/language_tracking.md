# Language Tracking Optimization: Speech Comprehension Paradigm

**Project:** EEG Prognostic Data Pipeline - AwakenAI Capstone  
**Priority:** 2 (Building on Prior Work)  
**Timeline:** Jan 25 - Mar 15, 2026  
**Lead:** Member B (Data Engineer) + Member D (QC & Vis)

---

## 📖 Table of Contents
1. [Background & Clinical Context](#background--clinical-context)
2. [The Language Tracking Paradigm](#the-language-tracking-paradigm)
3. [Neural Entrainment & ITPC](#neural-entrainment--itpc)
4. [Data Structures & Inputs](#data-structures--inputs)
5. [Technical Implementation](#technical-implementation)
6. [Optimization Strategies](#optimization-strategies)
7. [Expected Outputs](#expected-outputs)
8. [Quality Control & Validation](#quality-control--validation)
9. [Building on Tricia's Work](#building-on-tricias-work)
10. [Usage Examples](#usage-examples)
11. [References](#references)

---

## 🧠 Background & Clinical Context

### Covert Speech Comprehension

In severe brain injury patients, **behavioral assessments** (e.g., "squeeze my hand") often fail to detect consciousness due to:
- Motor deficits (inability to move).
- Sedation or metabolic disruptions.
- Fluctuating arousal states.

However, the brain may still be **processing language** at the cortical level, even without overt responses. This phenomenon is called **covert speech comprehension**.

### Why Language Tracking Matters

**Key Clinical Insight:**
> Patients who show evidence of neural speech tracking have significantly better recovery outcomes than those who do not, even when behavioral exams are equivalent.

**Sokoliuk et al. (2021)** demonstrated that EEG signatures of language processing in the acute phase predict recovery from disorders of consciousness.

### The Challenge: Limited Electrodes

Traditional language tracking studies use **high-density EEG arrays** (128-256 electrodes) to capture fine-grained neural responses. However:
- **Clinical Reality:** Most ICUs use only **16-20 electrodes** (standard 10-20 system).
- **Our Goal:** Develop a robust language tracking pipeline that works with **sparse electrode setups**, making it clinically viable.

---

## 🎯 The Language Tracking Paradigm

### Experimental Design

**Stimulus Structure:**
- **Monosyllabic Words:** Short, one-syllable words (e.g., "cat", "dog", "red", "ball").
- **Sentences:** Words are organized into 12-word sentences with semantic structure.
- **Presentation Rate:** ~1.3 seconds per word (rapid serial presentation).
- **Trial Duration:** 15-16 seconds (12 words/trial).

**Audio Generation:**
- Originally generated with **Apple Text-to-Speech** (or similar).
- All audio files **normalized** to consistent duration and amplitude.
- Stored as `lang0.wav` through `lang34.wav` (35 files total, `lang28.wav` missing).

### Hierarchical Structure

**Key Concept:** The stimulus has **multiple temporal frequencies**:
1. **Word Rate:** ~0.77 Hz (1 word every 1.3 seconds).
2. **Phrase Rate:** ~0.38 Hz (2-word phrases).
3. **Sentence Rate:** ~0.065 Hz (12-word sentences over 15.5 seconds).

**Hypothesis:** If the brain is **comprehending** the speech (not just hearing sounds), neural activity should **entrain** (synchronize) to the **sentence-level structure**, not just the word rate.

---

### Data from Our Dataset

**CON008 Session (Aug 14, 2025):**
- **72 language trials** recorded.
- Each trial: 12 sentences presented over ~15.5 seconds.
- Total language stimulation time: **~18.6 minutes**.

**CON009 Session (Aug 26, 2025):**
- **72 language trials** recorded.
- Mixed with control and emotional voice trials (more varied protocol).

**Example Trial Structure (CON008, Trial 1):**
```python
Sentence IDs: [10, 29, 19, 25, 15, 24, 12, 22, 16, 1, 8, 5]
Start: 1755207296.46 (Unix timestamp)
Duration: 15.62 seconds
```

**Randomization:** The order of sentences is pseudo-randomized across trials to prevent predictability.

---

## 🧬 Neural Entrainment & ITPC

### What is Neural Entrainment?

**Neural entrainment** (or phase locking) occurs when brain oscillations **synchronize** to the temporal structure of an external stimulus. For speech:
- The brain's neural activity "locks" to the rhythm of sentences.
- This reflects **hierarchical processing** of linguistic information.

### Inter-Trial Phase Coherence (ITPC)

**ITPC** measures the **consistency of phase** across multiple trials at a specific frequency.

**Mathematical Definition:**
```
ITPC(f, t) = |1/N * Σ exp(i * φ_n(f, t))|
```

Where:
- `φ_n(f, t)` = phase of trial `n` at frequency `f` and time `t`.
- `N` = total number of trials.
- ITPC ranges from 0 (random phase) to 1 (perfect phase coherence).

**Interpretation:**
- **High ITPC at Sentence Frequency (~0.065 Hz):** The brain is consistently tracking sentence-level structure across trials.
- **Low ITPC:** Neural responses are variable or absent.

### Why ITPC (Not Just Power)?

**Power Spectral Density (PSD)** measures the strength of oscillations, but **not** whether they are time-locked to the stimulus.

**ITPC** specifically measures **stimulus-locked activity**, which is the signature of **active processing** rather than spontaneous brain rhythms.

---

## 📂 Data Structures & Inputs

### Input 1: CSV Stimulus Log

**Location:** `EEG Project Data/EEG/*_stimulus_results.csv`

**Schema:**
```csv
patient_id,date,trial_type,sentences,start_time,end_time,duration
CON008,2025-08-14,language,"[10, 29, 19, 25, 15, 24, 12, 22, 16, 1, 8, 5]",1755207296.46,1755207312.08,15.62
```

**Relevant Fields:**
- `trial_type`: Must equal `"language"`.
- `sentences`: Array of 12 integers (IDs mapping to `langX.wav` files).
- `start_time` / `end_time`: Unix timestamps marking trial boundaries.

### Input 2: EDF File (EEG Recording)

**Location:** `EEG Project Data/EEG/edf/CON008_clipped.EDF`

**Key Channels for Language:**
- **Frontal:** `Fp1`, `Fp2`, `F3`, `F4`, `F7`, `F8` (early auditory processing).
- **Temporal:** `T7`, `T8` (auditory cortex, language processing).
- **Central:** `C3`, `C4` (motor/language interface).
- **Parietal:** `P3`, `P4`, `Pz` (higher-order integration).

**Left Hemisphere Focus:** For right-handed patients, language processing is typically **left-lateralized** (e.g., `T7`, `F7`, `P3` more relevant).

### Input 3: Audio Stimulus Files

**Location:** `EEG Project Data/Audio/sentences/`

**Files:**
- `lang0.wav`, `lang1.wav`, ..., `lang34.wav` (35 files total).
- **Missing:** `lang28.wav` (gap in sequence).

**Metadata Needed:**
- Word-level transcripts (e.g., "The cat sat on the mat").
- Audio duration per word (~1.3s).
- Onset times of each word within the trial.

**Action Item:** Create `stimulus_manifest.csv` mapping IDs to text content.

---

## 🔧 Technical Implementation

### Phase 1: Data Segmentation & Alignment

#### Step 1.1: Load and Filter Language Trials

```python
import pandas as pd
import mne

# Load CSV
csv_path = 'EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv'
stim_df = pd.read_csv(csv_path)

# Filter for language trials
lang_trials = stim_df[stim_df['trial_type'] == 'language'].copy()

print(f"Total language trials: {len(lang_trials)}")
# Output: Total language trials: 72
```

#### Step 1.2: Load EDF and Extract Language Epochs

```python
# Load EDF
edf_path = 'EEG Project Data/EEG/edf/CON008_clipped.EDF'
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Select channels (focus on language-relevant areas)
picks = mne.pick_types(raw.info, eeg=True, exclude=['DC1', 'EOG'])

# For each trial, extract the corresponding EEG segment
sfreq = raw.info['sfreq']
edf_start_time = raw.info['meas_date'].timestamp()

epochs_list = []

for idx, trial in lang_trials.iterrows():
    # Calculate segment in EDF
    trial_start = trial['start_time'] - edf_start_time
    trial_end = trial['end_time'] - edf_start_time
    
    # Crop the raw data for this trial
    trial_segment = raw.copy().crop(tmin=trial_start, tmax=trial_end)
    
    epochs_list.append(trial_segment)
```

---

### Phase 2: Time-Frequency Analysis & ITPC

#### Step 2.1: Compute Time-Frequency Representation (TFR)

Use **Morlet wavelets** to decompose the EEG signal into time-frequency components:

```python
from mne.time_frequency import tfr_morlet

# Define frequency range of interest
# Focus on sentence-level frequency: 0.065 Hz (1 sentence / 15.5s)
freqs = np.logspace(np.log10(0.05), np.log10(2), num=30)  # 0.05 - 2 Hz
n_cycles = freqs / 2  # Number of cycles for each frequency

# Compute TFR for each epoch
tfr_list = []
for epoch in epochs_list:
    tfr = tfr_morlet(epoch, freqs=freqs, n_cycles=n_cycles, 
                     use_fft=True, return_itc=False, picks=picks)
    tfr_list.append(tfr)
```

#### Step 2.2: Calculate ITPC

```python
# Extract phase information across trials
phases = []

for tfr in tfr_list:
    # Get complex time-frequency representation
    tfr_complex = tfr.data  # Shape: (n_channels, n_freqs, n_times)
    phase = np.angle(tfr_complex)  # Extract phase
    phases.append(phase)

phases = np.array(phases)  # Shape: (n_trials, n_channels, n_freqs, n_times)

# Compute ITPC: Average of unit vectors
itpc = np.abs(np.mean(np.exp(1j * phases), axis=0))
# Shape: (n_channels, n_freqs, n_times)

print(f"ITPC shape: {itpc.shape}")
```

#### Step 2.3: Extract Sentence-Frequency ITPC

```python
# Find the frequency bin closest to sentence rate (~0.065 Hz)
sentence_freq = 0.065  # Hz
sentence_freq_idx = np.argmin(np.abs(freqs - sentence_freq))

# Extract ITPC at this frequency
itpc_sentence = itpc[:, sentence_freq_idx, :]  # Shape: (n_channels, n_times)

# Average over time to get a single value per channel
itpc_sentence_avg = np.mean(itpc_sentence, axis=1)  # Shape: (n_channels,)
```

---

### Phase 3: Statistical Thresholding & Significance

#### Step 3.1: Baseline Comparison

To determine if the ITPC is **significant**, compare it to a baseline (e.g., pre-stimulus period or shuffled trials):

```python
from scipy.stats import ttest_1samp

# Hypothesis: ITPC at sentence frequency > chance (0.1)
threshold = 0.1
t_stat, p_value = ttest_1samp(itpc_sentence_avg, threshold)

print(f"T-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant language tracking detected!")
else:
    print("No significant language tracking.")
```

#### Step 3.2: Cluster-Based Permutation Test (Advanced)

For more robust statistics, use MNE's cluster permutation test to account for multiple comparisons:

```python
from mne.stats import permutation_cluster_1samp_test

# Reshape for cluster test: (n_trials, n_channels * n_times)
itpc_reshaped = itpc_sentence.reshape(len(epochs_list), -1)

# Run cluster test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    itpc_reshaped, threshold={'start': 0, 'step': 0.2}, n_permutations=1000
)

print(f"Significant clusters: {np.sum(cluster_p_values < 0.05)}")
```

---

## 🔍 Optimization Strategies

### Challenge: Sparse Electrode Array

**Problem:** Prior studies (e.g., Ding & Simon, 2012) used **128-256 electrodes**. We have only **16-20**.

**Our Optimization Approaches:**

#### 1. **Electrode Selection (Left-Hemisphere Focus)**

**Strategy:** Focus on electrodes over **left-hemisphere language areas** for right-handed patients.

```python
# Define left-hemisphere language channels
left_lang_channels = ['F7', 'T7', 'P7', 'F3', 'C3', 'P3']

# Filter ITPC to these channels only
left_lang_idx = [raw.ch_names.index(ch) for ch in left_lang_channels if ch in raw.ch_names]
itpc_left = itpc_sentence_avg[left_lang_idx]

# Average over left-hemisphere channels
itpc_left_mean = np.mean(itpc_left)
print(f"Left-hemisphere ITPC: {itpc_left_mean:.3f}")
```

#### 2. **Frequency Band Optimization**

**Strategy:** Test multiple frequency bands to find the optimal window for sentence tracking.

```python
# Test different frequency ranges
freq_bands = {
    'sentence': (0.05, 0.08),   # Sentence-level
    'phrase': (0.3, 0.5),        # Phrase-level (2-word chunks)
    'word': (0.7, 0.9)           # Word-level
}

for band_name, (fmin, fmax) in freq_bands.items():
    band_idx = (freqs >= fmin) & (freqs <= fmax)
    itpc_band = np.mean(itpc[:, band_idx, :], axis=1)  # Average over freq band
    itpc_band_avg = np.mean(itpc_band)
    print(f"{band_name.capitalize()} ITPC: {itpc_band_avg:.3f}")
```

#### 3. **Artifact Rejection & Preprocessing**

**Strategy:** Language trials are sensitive to high-frequency noise. Apply strict preprocessing.

```python
# Bandpass filter: 0.05 - 2 Hz (low-frequency envelope)
raw_filtered = raw.copy().filter(l_freq=0.05, h_freq=2, method='iir')

# Apply ICA to remove ocular artifacts
from mne.preprocessing import ICA

ica = ICA(n_components=15, random_state=42)
ica.fit(raw_filtered, picks=picks)

# Automatically detect eye blinks (if EOG channel available)
ica.exclude = [0, 1]  # Typically components 0-1 are artifacts
raw_clean = ica.apply(raw_filtered.copy())
```

#### 4. **Trial Averaging vs. Single-Trial Analysis**

**Strategy:** Compare **grand average** (across all 72 trials) vs. **single-trial** ITPC.

```python
# Grand Average: High statistical power, but may miss variability
itpc_grand_avg = np.mean(itpc_sentence_avg)

# Single-Trial: Lower SNR, but captures individual variability
itpc_trials = [np.mean(tfr_list[i].data) for i in range(len(tfr_list))]

print(f"Grand Average ITPC: {itpc_grand_avg:.3f}")
print(f"Single-Trial ITPC (std): {np.std(itpc_trials):.3f}")
```

---

## 📊 Expected Outputs

### 1. ITPC Topographic Map

**Visualization:** Heatmap showing ITPC values across all electrodes.

```python
import matplotlib.pyplot as plt
from mne.viz import plot_topomap

# Plot ITPC topography
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
plot_topomap(itpc_sentence_avg, raw.info, axes=ax, show=False, 
             cmap='RdBu_r', vlim=(0, 0.5), contours=6)
ax.set_title('ITPC at Sentence Frequency (0.065 Hz)', fontsize=14)
plt.colorbar(ax.images[0], ax=ax, label='ITPC')
plt.tight_layout()
plt.savefig('outputs/CON008_language_ITPC_topomap.png', dpi=300)
```

**Expected Pattern:**
- **High ITPC** (red) over **left temporal/frontal** regions (T7, F7) for right-handed patients.
- **Low ITPC** (blue) over occipital (visual) regions.

### 2. Time-Frequency ITPC Plot

**Visualization:** Spectrogram showing how ITPC evolves over time and frequency.

```python
# Average ITPC over left-hemisphere channels
itpc_left_tfr = np.mean(itpc[left_lang_idx, :, :], axis=0)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
im = ax.imshow(itpc_left_tfr, aspect='auto', origin='lower',
               extent=[0, trial_duration, freqs[0], freqs[-1]], cmap='hot')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Frequency (Hz)', fontsize=12)
ax.set_title('ITPC Time-Frequency Representation (Left Hemisphere)', fontsize=14)
plt.colorbar(im, ax=ax, label='ITPC')
plt.savefig('outputs/CON008_language_ITPC_tfr.png', dpi=300)
```

### 3. Feature Table

| patient_id | n_trials | itpc_sentence | itpc_phrase | itpc_word | left_hem_itpc | p_value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CON008 | 72 | 0.32 | 0.18 | 0.12 | 0.38 | 0.002 |
| CON009 | 72 | 0.29 | 0.21 | 0.15 | 0.35 | 0.008 |

---

## ✅ Quality Control & Validation

### Validation Checklist

- [ ] **Trial Count:** Verify that all language trials are successfully loaded (72 expected).
- [ ] **Missing Stimuli:** Flag trials that reference `lang28.wav` (missing file).
- [ ] **Baseline ITPC:** Check that pre-stimulus ITPC is near 0 (no spurious coherence).
- [ ] **Frequency Resolution:** Ensure that 0.065 Hz is within the analyzed frequency range.
- [ ] **Lateralization:** For right-handed patients, expect left-hemisphere dominance.

### Expected Results (Control Data)

For **awake, healthy control subjects**:
- **ITPC at Sentence Frequency:** 0.25 - 0.45 (moderate to strong coherence).
- **ITPC at Word Frequency:** 0.10 - 0.20 (weak; we want sentence-level tracking).
- **Topography:** Peak ITPC at **T7** (left temporal) and **F7** (left frontal).

### Red Flags (Indicates Pipeline Error)

- **Uniform ITPC Across Frequencies:** Suggests artifact or improper filtering.
- **ITPC > 0.7:** Unrealistically high; likely due to insufficient trial averaging or bad channels.
- **Right-Hemisphere Dominance (Right-Handed Patient):** May indicate incorrect channel labeling.
- **No Significant Difference from Baseline:** Could indicate poor synchronization or noisy data.

---

## 🔧 Building on Tricia's Work

### What Tricia Accomplished

**Previous Student:** Tricia (now in industry) implemented an initial language tracking pipeline for this project.

**Her Contributions:**
1. **Proof-of-Concept:** Successfully detected language tracking in 2-3 control subjects.
2. **Code Base:** Well-commented Python scripts using MNE.
3. **Poster Submission:** Results submitted to Society for Neuroscience (Peter will provide).

**Limitations Identified:**
- **Inconsistent Results:** Signal detection varied significantly across sessions.
- **Electrode Averaging:** Used **global averaging** (all electrodes), which dilutes the signal.
- **Limited Optimization:** Did not explore frequency band tuning or electrode selection.

### Our Objectives (Building on Tricia)

1. **Reproduce Her Results:** Validate her pipeline on the same data.
2. **Optimize Electrode Selection:** Test left-hemisphere-only vs. global averaging.
3. **Frequency Tuning:** Systematically explore 0.05-0.1 Hz for optimal sentence tracking.
4. **Scale to Full Dataset:** Apply to all 10 patients (not just 2-3).
5. **Clinical Validation:** Correlate ITPC with patient outcomes (if outcome data available).

### Accessing Tricia's Code

**Location:** GitHub repository (Peter will provide access).

**Key Files to Review:**
- `language_tracking.py` - Main pipeline.
- `itpc_analysis.ipynb` - Jupyter notebook with visualizations.
- `README.md` - Documentation of her approach.

**Action Item:** Review Tricia's code in **Week 1** (Jan 10-17) to understand baseline methodology.

---

## 💻 Usage Examples

### Example 1: Full Pipeline for One Patient

```python
from language_tracker import LanguageTrackingPipeline

# Initialize pipeline
pipeline = LanguageTrackingPipeline(
    edf_path='EEG Project Data/EEG/edf/CON008_clipped.EDF',
    csv_path='EEG Project Data/EEG/CON008_2025-08-14_stimulus_results.csv',
    focus_channels=['F7', 'T7', 'P7', 'F3', 'C3', 'P3']  # Left hemisphere
)

# Run full pipeline
pipeline.load_data()
pipeline.segment_trials()
pipeline.compute_itpc(freq_band=(0.05, 0.08))

# Extract features
features = pipeline.quantify_itpc()
print(features)

# Save outputs
pipeline.save_topomap('outputs/CON008_language_topomap.png')
pipeline.save_tfr_plot('outputs/CON008_language_tfr.png')
pipeline.save_features('processed/features/CON008_language.csv')
```

### Example 2: Compare Left vs. Global Averaging

```python
# Test 1: Left-hemisphere only
itpc_left = pipeline.compute_itpc(channels=['F7', 'T7', 'P7'])

# Test 2: Global averaging (all channels)
itpc_global = pipeline.compute_itpc(channels='all')

# Compare
print(f"Left-Hemisphere ITPC: {np.mean(itpc_left):.3f}")
print(f"Global ITPC: {np.mean(itpc_global):.3f}")
print(f"Improvement: {(np.mean(itpc_left) - np.mean(itpc_global)) / np.mean(itpc_global) * 100:.1f}%")
```

### Example 3: Batch Processing with Optimization Grid

```python
# Define hyperparameter grid
freq_bands = [(0.05, 0.08), (0.06, 0.09), (0.04, 0.10)]
channel_sets = [
    ['F7', 'T7', 'P7'],           # Left temporal
    ['F3', 'C3', 'P3'],           # Left central
    ['all']                       # Global
]

results = []

for freq_band in freq_bands:
    for channels in channel_sets:
        itpc_val = pipeline.compute_itpc(freq_band=freq_band, channels=channels)
        results.append({
            'freq_band': freq_band,
            'channels': str(channels),
            'itpc_mean': np.mean(itpc_val),
            'itpc_max': np.max(itpc_val)
        })

# Find optimal configuration
results_df = pd.DataFrame(results)
best_config = results_df.loc[results_df['itpc_mean'].idxmax()]
print("Optimal Configuration:")
print(best_config)
```

---

## 📚 References

### Foundational Papers

1.  **Ding, N., & Simon, J. Z. (2012).** "Neural coding of continuous speech in auditory cortex during monaural and dichotic listening." *Journal of Neurophysiology*, 107(1), 78-89.
    - Original paper demonstrating neural entrainment to speech structure.

2.  **Luo, H., & Poeppel, D. (2007).** "Phase patterns of neuronal responses reliably discriminate speech in human auditory cortex." *Neuron*, 54(6), 1001-1010.
    - Phase-locking mechanisms in speech perception.

3.  **Zoefel, B., & VanRullen, R. (2015).** "The role of high-level processes for oscillatory phase entrainment to speech sound." *Frontiers in Human Neuroscience*, 9, 651.
    - Hierarchical processing and sentence-level tracking.

### Clinical Applications

4.  **Sokoliuk, R., et al. (2021).** "Covert speech comprehension predicts recovery from acute unresponsive states." *Annals of Neurology*, 89(4), 646-656.
    - **Key paper** showing language tracking as a prognostic marker in brain injury.

5.  **Claassen, J., et al. (2019).** "Detection of brain activation in unresponsive patients with acute brain injury." *NEJM*, 380(26), 2497-2505.
    - Context for using EEG-based cognitive assessments in ICU.

### Methodological Guides

6.  **Lachaux, J. P., et al. (1999).** "Measuring phase synchrony in brain signals." *Human Brain Mapping*, 8(4), 194-208.
    - ITPC mathematical foundations.

7.  **MNE Time-Frequency Tutorial:** https://mne.tools/stable/auto_tutorials/time-freq/index.html
    - Practical guide to implementing TFR and ITPC in Python.

---

## 🚀 Next Steps

1.  **Week 3 (Jan 25-31):** Review Tricia's code, reproduce her results on 1-2 subjects.
2.  **Week 4-5 (Feb 01-13):** Implement electrode selection and frequency optimization.
3.  **Week 6-7 (Feb 14-28):** **MILESTONE CHECK:** Present optimized ITPC results with topomap and statistics.
4.  **Week 8-9 (Mar 01-15):** Scale to full dataset, generate final feature tables and publication-quality plots.

---

**Last Updated:** December 10, 2025  
**Author:** AwakenAI Capstone Team  
**Contact:** [Team Communication Channel]
