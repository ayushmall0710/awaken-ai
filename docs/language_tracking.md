# Language Tracking Pipeline: Speech Comprehension Paradigm

**Project:** EEG Prognostic Data Pipeline — AwakenAI Capstone
**Status:** Implemented — `src/pipelines/language_tracking.py` (`LanguageTrackingAnalysis`), wired into the CLI as `awakenai run --pipeline language`.
**Code:** `src/pipelines/language_tracking.py`, `src/cli/runners/language.py`, `src/viz/language_plots.py`, `src/reports/language_tracking_report.py`

> This document describes the *implemented* pipeline. For the original
> design brief this was built from, see `tasks/ENG-05.md` and
> `tasks/ENG-05-analysis.md`.

---

## Table of Contents
1. [Background & Clinical Context](#background--clinical-context)
2. [The Language Tracking Paradigm](#the-language-tracking-paradigm)
3. [Neural Entrainment & ITPC](#neural-entrainment--itpc)
4. [Pipeline Inputs](#pipeline-inputs)
5. [Implementation](#implementation)
6. [Outputs](#outputs)
7. [Quality Control & Validation](#quality-control--validation)
8. [Usage](#usage)
9. [References](#references)

---

## Background & Clinical Context

### Covert Speech Comprehension

In severe brain injury patients, **behavioral assessments** (e.g., "squeeze my hand") often fail to detect consciousness due to:
- Motor deficits (inability to move).
- Sedation or metabolic disruptions.
- Fluctuating arousal states.

However, the brain may still be **processing language** at the cortical level, even without overt responses. This phenomenon is called **covert speech comprehension**.

### Why Language Tracking Matters

**Sokoliuk et al. (2021)** demonstrated that EEG signatures of language processing in the acute phase predict recovery from disorders of consciousness — patients who show evidence of neural speech tracking have significantly better recovery outcomes than those who do not, even when behavioral exams are equivalent.

### The Challenge: Limited Electrodes

Traditional language tracking studies use **high-density EEG arrays** (128–256 electrodes). Our clinical setup uses **16–20 electrodes** (standard 10-20 system), so the pipeline is built around sparse-array focus selection rather than assuming dense coverage.

---

## The Language Tracking Paradigm

### Experimental Design

- **Monosyllabic Words:** Short, one-syllable words (e.g., "cat", "dog", "red", "ball").
- **Sentences:** Words organized into 12-word sentences with semantic structure.
- **Presentation Rate:** ~1.3 seconds per word (rapid serial presentation).
- **Trial Duration:** ~15–16 seconds (12 words/trial).
- **Audio:** Stored as `lang0.wav` through `lang34.wav` under `data/Audio/sentences/` (35 files; `lang28.wav` is missing from the set).

### Hierarchical Structure

Following Sokoliuk et al. (2021), the stimulus has multiple temporal frequencies:

| Level    | Target frequency | Cycle length            |
|----------|-------------------|--------------------------|
| Sentence | 0.78 Hz           | 12 sentences / ~15.36s (1.28s/sentence) |
| Phrase   | 1.56 Hz           | One phrase every 0.64s (2 words/phrase) |
| Word     | 3.125 Hz          | One word every 0.32s |

**Hypothesis:** If the brain is *comprehending* the speech (not just hearing sounds), neural activity should entrain to the sentence-level (0.78 Hz) and phrase-level (1.56 Hz) structure, in addition to the acoustic word rate.

---

## Neural Entrainment & ITPC

**Neural entrainment** (phase locking) occurs when brain oscillations synchronize to the temporal structure of a stimulus. **Inter-Trial Phase Coherence (ITPC)** measures the consistency of phase across trials at a given frequency:

```
ITPC(f, t) = |1/N * Σ exp(i * φ_n(f, t))|
```

ITPC ranges from 0 (random phase) to 1 (perfect phase coherence). Unlike power spectral density, ITPC specifically captures **stimulus-locked** activity — the signature of active processing rather than spontaneous rhythms.

The pipeline computes ITPC two ways (see [Implementation](#implementation) below): a **DFT** method (matching Sokoliuk et al. 2021's approach) and a **Morlet wavelet** method, and reports both.

---

## Pipeline Inputs

Unlike the earlier raw-CSV/EDF design, `LanguageTrackingAnalysis` does **not** read raw stimulus logs or EDFs directly. It consumes the outputs of the setup prerequisites (`awakenai setup <patient>`):

1. **Aligned events** — `data/processed/aligned_events/{patient_id}_events.parquet`, produced by `TimestampAligner` (ENG-02). Used by `BasePipeline.run()` to resolve session IDs.
2. **Clean language epochs** — `data/processed/epochs/{patient_id}/{session_id}/language-epo.fif`, produced by `ArtifactRejector` (ENG-03): ICA-cleaned, per-session, per-trial-type epochs.

If these don't exist for a patient/session, `awakenai run` reports the pipeline as blocked and tells you to run `awakenai setup` first.

---

## Implementation

### Configuration — `LanguageConfig`

A dataclass (`src/pipelines/language_tracking.py`) holds all tunable constants:

- **Filtering:** highpass 0.02 Hz, lowpass 25 Hz, resample to 256 Hz.
- **Epoch cropping:** `2.28s`–`16.36s` (removes filter edge artifacts, per Sokoliuk et al. 2021).
- **Target frequencies:** sentence 0.78 Hz, phrase 1.56 Hz, word 3.125 Hz, each with a ±10% bandwidth for the Morlet band-averaged metrics.
- **DFT frequency resolution:** 0.01 Hz.
- **Morlet frequency axis:** 60 log-spaced points from 0.5–5.0 Hz (used for the TFR/topomap visualization, not the target-frequency phase extraction).

### Pipeline stages (`BasePipeline` template: `load → preprocess → analyze`)

1. **`load()`** — loads and concatenates clean language epochs across sessions (or a single session if `session_id` is given).
2. **`preprocess()`** — applies the bandpass filter + resample + crop (`_preprocess_signal`) and sets a standard 10-20 montage for topomap plotting.
3. **`analyze()`** — a two-phase computation:
   - **Phase 1 (per-channel, computed once):**
     - Pick the `CLINICAL_20` channel subset.
     - **DFT ITPC** (`ITPCProcessor.compute_dft_itpc`): FFT the epoch data, take the phase, average unit vectors across trials, per channel per frequency bin.
     - **Morlet ITPC**: compute per-trial phase at the three target frequencies (`_compute_morlet_target_phases`, 5-cycle wavelets) and derive per-channel ITPC (`_compute_per_channel_itpc_morlet`).
     - **Null distributions** for both methods via trial-level random phase scrambling (`PermutationEngine.generate_null_distribution`, n=1000 by default) — adds an identical random phase offset per trial across all channels, preserving 1/f noise and spatial covariance while destroying stimulus-locking.
   - **Phase 2 (per-focus aggregation):**
     - **Optimal channel selection** (`_select_optimal_channels`) via spatial cluster permutation (`src/utils/signal_processing.select_optimal_channels`) on comprehension-frequency (avg of sentence + phrase) ITPC: vet clusters at α<0.05, exclude `Fp1`/`Fp2` (eye-artifact-prone), then take the top 3 electrodes by comprehension ITPC. Returns an empty list if no cluster survives.
     - Four **focuses** are resolved (`_resolve_focuses`): `clinical` (all 19 available `CLINICAL_20` channels), `lh`/`rh` (fixed hemisphere channel sets), `optimal` (data-driven).
     - For each focus, `_build_focus_row` averages the per-channel metrics over that focus's channels and computes permutation p-values by subsetting the per-channel null distributions — so the p-value for a channel subset is drawn from the correctly-scoped null, not recomputed from scratch.

The result is one row per focus per patient/session, held in `self.results` (a `pandas.DataFrame`).

### Derived summary metrics (`generate_summary()`)

From the per-focus rows, computes:
- **Lateralization index** per band: `LI = (LH − RH) / (LH + RH)`, for word/phrase/sentence/comprehension. Positive = left-lateralized (expected in right-handed patients for language).
- **`ratio_cognitive_acoustic`**: comprehension ITPC / word ITPC (clinical focus) — how much stronger the sentence/phrase-level tracking is relative to the acoustic word-rate response.
- **`morlet_ratio`**: Morlet sentence ITPC / Morlet word ITPC (clinical focus).

---

## Outputs

Per session, `awakenai run --pipeline language --report` writes to
`data/reports/{patient_id}/{session_id}/language_tracking/`:

- **`features.csv`** — one row per focus (`clinical`/`lh`/`rh`/`optimal`), appended across runs. Columns include `itpc_word`, `itpc_phrase`, `itpc_sentence`, `itpc_comprehension` (DFT), `morlet_itpc_*` (Morlet), `dft_p_*`/`morlet_p_*` (permutation p-values), `ratio_sent_word`, `ratio_sent_phrase`, `ratio_bw_normalized` (bandwidth-normalized sentence/word density ratio), and the peak frequency found near each target (`freq_sentence_hz`, etc.).
- **`features.npz`** — the full per-channel DFT spectrum (`dft_spectrum_full`, `dft_freqs`, `ch_names`) for re-plotting without recomputation.
- **`report.html`** (when `--report` is passed) — built by `LanguageTrackingReport` / `src/viz/language_plots.py`, including:
  - ITPC topomap + TFR with dynamic color scale (vlim = 1.2× the 95th percentile) and target-frequency overlay lines.
  - Per-channel ITPC bar chart (sentence/phrase/word) with a `1/√N` chance-level reference line.
  - Focus comparison bar chart (`clinical`/`lh`/`rh`/`optimal`) with p-value annotations.
- A **combined patient report** (`data/reports/{patient_id}/combined/language_tracking/report.html`) is generated automatically when a patient has more than one session with results.

---

## Quality Control & Validation

### Sanity checks
- **Trial count:** expect ~72 language trials per session (dataset-dependent).
- **Missing stimulus file:** `lang28.wav` is absent from the audio set — trials referencing it should be flagged upstream during stimulus manifest creation.
- **Frequency resolution:** the DFT resolution (0.01 Hz) must resolve 0.78/1.56/3.125 Hz distinctly from neighboring bins.
- **Lateralization:** for right-handed patients, expect left-hemisphere dominance (positive lateralization index).

### Expected results (awake, healthy controls)
- **ITPC at sentence frequency (~0.78 Hz):** 0.25–0.45 (moderate to strong coherence).
- **ITPC at phrase frequency (~1.56 Hz):** some entrainment expected from 0.64s acoustic boundaries.
- **ITPC at word frequency (~3.125 Hz):** weak to moderate.
- **Topography:** peak ITPC typically at **T7** (left temporal) and **F7** (left frontal).

### Red flags
- **Uniform ITPC across frequencies** — suggests artifact or improper filtering.
- **ITPC > 0.7** — unrealistically high; likely insufficient trial averaging or bad channels.
- **Right-hemisphere dominance in a right-handed patient** — may indicate incorrect channel labeling.
- **No significant difference from baseline** — poor synchronization or noisy data.

---

## Usage

### CLI

```bash
awakenai setup CON008                                    # prerequisite: alignment + ICA epochs
awakenai run CON008 --pipeline language --report          # single patient
awakenai run CON008 CON009 --pipeline language --session 2025-08-14
awakenai run --all --pipeline language                    # every patient with language trials
```

### Python API

```python
from src.data_loading import UnifiedDataLoader
from src.pipelines.language_tracking import LanguageTrackingAnalysis

loader = UnifiedDataLoader()
pipeline = LanguageTrackingAnalysis(loader=loader)

results = pipeline.run("CON008")          # DataFrame: one row per focus
summary = pipeline.generate_summary()     # lateralization indices, ratios

# Restrict to one session:
pipeline_session = LanguageTrackingAnalysis(loader=loader, session_id="2025-08-14")
results_session = pipeline_session.run("CON008", session_id="2025-08-14")
```

---

## References

### Foundational papers

1. **Ding, N., & Simon, J. Z. (2012).** "Neural coding of continuous speech in auditory cortex during monaural and dichotic listening." *Journal of Neurophysiology*, 107(1), 78–89.
2. **Luo, H., & Poeppel, D. (2007).** "Phase patterns of neuronal responses reliably discriminate speech in human auditory cortex." *Neuron*, 54(6), 1001–1010.
3. **Zoefel, B., & VanRullen, R. (2015).** "The role of high-level processes for oscillatory phase entrainment to speech sound." *Frontiers in Human Neuroscience*, 9, 651.

### Clinical applications

4. **Sokoliuk, R., et al. (2021).** "Covert speech comprehension predicts recovery from acute unresponsive states." *Annals of Neurology*, 89(4), 646–656. — Key methodological reference for this pipeline's DFT ITPC approach and epoch cropping.
5. **Claassen, J., et al. (2019).** "Detection of brain activation in unresponsive patients with acute brain injury." *NEJM*, 380(26), 2497–2505.

### Methodological guides

6. **Lachaux, J. P., et al. (1999).** "Measuring phase synchrony in brain signals." *Human Brain Mapping*, 8(4), 194–208.
7. **MNE Time-Frequency Tutorial:** https://mne.tools/stable/auto_tutorials/time-freq/index.html
