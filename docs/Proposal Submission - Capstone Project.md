# Proposal Submission

## Awaken AI

**Sponsor:** Dr. Peter Schwab, Harborview Medical Center  
**Clinical Collaborator:** Alex Diamond  
**Team** ([Bio Link](https://docs.google.com/document/d/11WY9jCLHoR54mv7cqQY2hqpbg_DZX2KhBzQizQg-ld4/edit?usp=sharing))**:** Aaditya Chopra, Arnav Dixit, Ayush Mall, Riddhesh Sawant

## **1\. Problem Statement**

Severe brain injury patients in the ICU can retain meaningful auditory and cognitive processing even when they appear unresponsive at the bedside. Current prognostication relies heavily on behavioral scales (e.g., GCS) and structural imaging, which can be insensitive to “covert” cognition and often produce uncertain predictions about recovery.

Our sponsor’s study collects EEG while patients passively listen to **oddball tones**, **language stimuli**, and **command prompts**, to extract neural signatures of residual cognition that might predict later awakening and functional outcomes. However, the existing data are fragmented across EDF files, stimulus logs, audio files, and partially processed notebooks, with no unified, reproducible analysis pipeline.

**Our project goal** is to design and implement a **robust EEG prognostic data pipeline** that:

1. Ingests raw EDF EEG files and stimulus timing logs.  
2. Aligns neural data precisely with auditory stimuli across multiple paradigms.  
3. Extracts interpretable neural markers such as:  
   * Event-related potentials (ERPs), especially auditory **P300** from the oddball paradigm.  
   * **Language tracking** measures (e.g., sentence-rate ITPC) from speech stimuli.  
   * Spectral features for command-following paradigms.  
4. Produces well-documented, reusable data products (epochs, feature tables, QC reports) that future clinical collaborators can apply to larger cohorts.

In the long term, this pipeline will support research on whether these EEG markers can improve prognostication beyond standard clinical scores. In the short term (this capstone), success means a **validated end-to-end analysis pipeline** running on the existing Harborview dataset.

## ---

## **2\. Background and Prior Work**

### **2.1 Scientific Context**

Our individual background reviews focused on two key strands of literature:

* **Auditory P300 in coma** – Gott et al. showed that an auditory P300 ERP can be detected in a subset of non-traumatic coma patients and that its presence is associated with higher GCS scores and a greater probability of awakening. This reframes P300 from a purely experimental measure into a **prognostic marker**, albeit with limited sensitivity and a small sample size.   
* **Cortical tracking of speech** – Sokoliuk et al. used a frequency-tagged language paradigm where isochronous words form phrases and sentences. They quantified **inter-trial phase coherence (ITPC)** at word, phrase, and sentence frequencies and showed that “comprehension responses” at higher-level linguistic rates significantly improved 3–6 month outcome prediction over standard clinical predictors in unresponsive traumatic brain injury patients. 

Both papers support a shared thesis: **residual auditory cognition is measurable with EEG and carries independent prognostic information**, even in the absence of overt command following.

### **2.2 Methodological Lessons for Our Project**

From this literature, we draw several design principles for our pipeline:

* Treat EEG as **feature vectors**, not just binary “present/absent” markers.  
* Combine multiple paradigms (P300, language tracking, command-following) rather than relying on a single ERP.  
* Use **event-locked averaging** for ERPs and **frequency-domain features** such as ITPC for language paradigms.  
* Plan for **regression or classification models** that incorporate both clinical variables and EEG features, even if we do only exploratory modeling this quarter due to limited time.

We view our project as a step toward an integrated, multimodal prognostic tool that is compatible with standard 19–20 channel clinical EEG systems, rather than high-density research arrays.

---

## 

## **3\. Data Description and EDA Summary ([Link](https://github.com/ayushmall0710/awaken-ai/blob/main/eda/EDA.pdf))**

### **3.1 Data Assets**

Our exploratory data analysis identified the following main assets from the Harborview study: 

* **EEG Recordings**  
  * 17 EDF/EDF+ files from 10 patients (CON001–CON010 \+ test).  
  * Some have “clipped” versions focusing on experimental windows.  
  * 16–64 EEG channels with typical clinical sampling rates (250–2000 Hz).  
* **Trial/Event Logs (CSV)**  
  * 21 CSV files, including multiple `patient_df*.csv` variants and session-specific stimulus logs.  
  * Core columns include: `patient_id`, `date`, `trial_type`, `sentences`, `start_time`, `end_time`, `duration`.  
  * Precision: millisecond-resolution Unix timestamps.  
* **Audio Stimuli**  
  * \~35 sentence WAV files (`lang0.wav`–`lang34.wav`, missing `lang28.wav`).  
  * Prompt and static command files (e.g., motor command prompts, beep stimuli).  
  * Trimmed voice clips (control and “loved one” voices).  
* **Patient Metadata & Notes**  
  * `patient_history*.csv`, `patient_notes*.csv` with sparse clinical notes such as “audio on left side” or “interrupted by monitor briefly.”

### **3.2 Trial Types and Paradigms**

From the EDA, we identified several paradigms encoded via `trial_type`: 

* `language`: \~12-sentence blocks (\~15–16 s) for speech tracking.  
* `left_command+p` / `right_command+p`: \~200–212 s motor command blocks.  
* `oddball+p`: \~30–34 s beep oddball tasks.  
* `control`: short baseline trials (\~2.6 s).  
* `loved_one_voice`: \~3.5 s emotional voice clips.

**Example:** CON009 has 184 trials spanning language, commands, oddball, control, and loved-one voice, with an estimated total recording duration \>3 hours. 

### **3.3 Data Challenges Identified**

The EDA and data pipeline report highlighted several key challenges that shape our proposed approach:

* **Schema drift** across CSV variants (`patient_df_043025`, `patient_df_052225`, etc.) with inconsistent columns and duplicate versions.  
* **Protocol drift**: earlier sessions differ (e.g., absence of loved-one voice trials), making cross-patient pooling non-trivial.  
* **Missing data**: missing audio files (e.g., `lang28.wav`) and at least one EDF (`CON006.EDF`) flagged as not downloaded.  
* **Temporal alignment risk**: EDF and CSV timestamps must be aligned carefully, using both Unix time and the audio DC channel.  
* **Sparse clinical outcomes**: current files lack formal outcome scales (GOSE, CRS-R), limiting what prognostic modeling we can do this quarter.

### **3.4 Planned Data Products**

Our EDA document already outlines a target output structure: processed epochs (HDF5/NumPy), feature tables (Parquet/CSV), QC logs, and visualizations stored under a `processed/` hierarchy.

Our proposal formalizes this structure as one of the main deliverables (Section 5).

---

## **4\. Proposed Technical Approach**

We organize the technical work into **four layers**:

1. **Infrastructure & harmonization**  
2. **Paradigm-specific analysis pipelines** (P300, language tracking, command-following)  
3. **Feature aggregation & exploratory statistics**  
4. **Documentation, QC, and reproducibility**

### **4.1 Infrastructure & Data Harmonization**

**Objectives**

* Create a reliable entry point from raw data (EDF/CSV/audio) to analysis-ready epochs.  
* Resolve schema drift and generate a single, authoritative trial log per patient.

**Planned Components**

1. **Repository & environment setup**  
   * Use sponsor’s GitHub repository as upstream.  
   * Standard Python environment with MNE, NumPy, SciPy, pandas, and plotting libraries.  
2. **CSV schema unification**  
   * Script that ingests all `patient_df*` files, harmonizes column names, resolves duplicates, and outputs a **canonical `patient_trials.csv`** with one row per trial.  
   * Explicitly logs skipped or ambiguous rows and missing files (e.g., `lang28.wav`).   
     annotated-EDA  
3. **Stimulus manifest**  
   * Generate `stimulus_manifest.csv` mapping sentence IDs (`lang0`–`lang34`) to audio files and transcripts (when available), including duration and basic linguistic features (e.g., word count).  
4. **EEGDataLoader class**  
   * Abstraction for:  
     * Loading EDF (original or clipped) with MNE.  
     * Pulling trials from the unified CSV for a given patient and paradigm.  
     * Returning aligned, trial-locked epochs based on timestamps.  
5. **Timestamp alignment using the DC audio channel**  
   * Use the audio waveform channel present in EDF to align stimulus times from CSV with EEG sample indices.  
   * Implement validation routines (cross-check expected trial durations against EDF segments; flag drift).

This layer directly addresses risks around misalignment, inconsistent CSVs, and missing files.

### **4.2 Oddball / P300 ERP Pipeline (Primary Technical Deliverable)**

**Goal:** Given EDF \+ stimulus logs, output a clean **event-related potential (ERP)** for deviant tones in the oddball paradigm and automatically quantify P300 amplitude and latency.

**Key Steps**

1. **Trial selection & epoching**  
   * Filter unified trial log to `trial_type == 'oddball+p'`.  
   * Use DC audio channel \+ CSV timestamps to mark onsets of standard vs deviant beeps (the CSV specifies which are “deviant”).  
   * Epoch EEG from \-100 ms to 700 ms around each stimulus; baseline correct using the pre-stimulus interval.  
2. **Preprocessing & artifact rejection**  
   * Bandpass filter (e.g., 0.1–30 Hz) to isolate ERP frequencies.  
   * Apply ICA or related techniques to remove eye-blink and muscle artifacts, with automatic rejection thresholds and optional manual inspection for the first subjects.  
3. **ERP computation**  
   * Separately average deviant-beep epochs per subject and per relevant electrode cluster (e.g., parietal sites).  
   * Optionally compute difference waves (deviant – standard).  
4. **Peak detection & quantification**  
   * Detect the maximal positive peak within a 250–600 ms window for P300 (or negative, depending on sign convention).  
   * Record peak latency (ms) and amplitude (µV) for each subject and electrode cluster.  
   * Store these metrics in a central feature table.  
5. **Validation**  
   * Evaluate on **awake control data** (as suggested by sponsor) where robust P300 is expected; failure to see a peak triggers QC review.  
   * Produce “grand average” ERP plots for at least one control subject and across all controls.

This pipeline aligns directly with the Gott et al. logic of treating P300 presence and magnitude as a marker of residual cognition, while exposing richer continuous features.

### **4.3 Language Tracking Pipeline**

**Goal:** Reuse and extend prior work (Trisha’s code) to compute **sentence- and phrase-rate ITPC** as a measure of covert speech comprehension using only standard clinical EEG montages.

**Key Steps**

1. **Trial selection & segmentation**  
   * Filter `trial_type == 'language'` from the unified trial log.  
   * Use sentence IDs from the `sentences` column to associate trials with specific audio sequences.  
2. **Frequency-tagging analysis**  
   * Compute ITPC at word, phrase, and sentence frequencies using Fourier transforms over the trial epochs, mirroring the Sokoliuk et al. approach summarized in our background research.   
   * Average ITPC across trials to derive a per-subject “comprehension response”.  
3. **Electrode selection strategies**  
   * Compare naive averaging over all electrodes vs. subsets likely to capture language (e.g., left temporal/parietal sites), especially in right-handed patients.  
4. **Stability and consistency checks**  
   * Bootstrap across non-linguistic frequencies to verify that peaks at phrase/sentence frequencies are not generic SNR artifacts (as in Sokoliuk et al.).   
     annotated-DATA 590\_ Background …  
5. **Feature extraction**  
   * For each subject, generate features such as:  
     * Mean sentence-rate ITPC over selected electrodes.  
     * Ratio of phrase/sentence ITPC to baseline word-rate ITPC.  
   * Store in the central feature table next to P300 metrics.

### **4.4 Command-Following and Other Paradigms (Stretch Goals)**

**Goal:** Characterize motor command and emotional voice paradigms where feasible, primarily through spectral and time–frequency features.

1. **Motor command paradigms (`left_command+p`, `right_command+p`)**  
   * Extract long epochs (\~200 s).  
   * Compute bandpower (e.g., µ/β bands) over motor-related channels during “keep” vs “stop” periods.  
   * Explore simple contrasts to see whether any volitional modulation is detectable at the single-subject level.  
2. **Loved-one voice & control voices**  
   * Compare ERPs and spectral responses to familiar vs neutral voices.  
   * Given time constraints and small N, we treat this as exploratory.

These analyses will likely produce fewer features but may become important in future cohorts.

### **4.5 Feature Aggregation and Exploratory Modeling (Stretch Goals)**

Once paradigm-specific pipelines run reliably, we will:

1. **Assemble feature table**  
   * One row per subject (and possibly per session).  
   * Columns: P300 metrics, language ITPC metrics, command-following bandpower, plus basic data-quality indicators (artifact rates, number of usable trials).   
     annotated-EDA  
2. **Exploratory statistics**  
   * Correlation matrices to understand feature relationships.  
   * Cluster analysis to see if subjects naturally group by feature profile (e.g., “strong language tracking vs. weak P300”).  
3. **Preliminary outcome modeling (if metadata available in time)**  
   * If clinical outcomes (e.g., GOSE, CRS-R) become available this quarter, we will fit simple regression models analogous to the Sokoliuk framework; otherwise, we will focus on within-subject feature characterization. 

---

## 

## **5\. Proposed Deliverables**

We break deliverables into **Minimal Viable Product (MVP)** and **Stretch Goals.**

### **5.1 MVP Deliverables (Committed)**

1. **Unified Trial Log & Stimulus Manifest**  
   * Canonical CSV for all trials across patients with a harmonized schema.  
   * Separate manifest linking sentence IDs to audio content and duration.   
2. **EEGDataLoader Library**  
   * Python package (module) that:  
     * Loads EDF \+ stimulus logs.  
     * Aligns timestamps using the DC audio channel.  
     * Returns MNE `Epochs` objects for specified paradigms.  
3. **Oddball/P300 ERP Pipeline**  
   * Script/notebook producing subject-level and grand-average ERPs for deviant tones.  
   * Automatic extraction of P300 latency and amplitude for at least one control subject and one patient.  
4. **Language Tracking Pipeline (Baseline Version)**  
   * Working reimplementation or extension of Trisha’s methods to compute linguistic-frequency ITPC using the current dataset.  
   * At least one set of plots demonstrating sentence-rate ITPC for a selected subject.  
5. **Processed Data Products**  
   * `processed/epochs/`: HDF5/NumPy epochs for each subject and paradigm.  
   * `processed/features/`: feature tables with P300 and ITPC metrics.  
   * `processed/qc/`: artifact rejection logs and trials-per-subject summaries.   
6. **Documentation**  
   * `DATA_PIPELINE_DOCUMENTATION.md`: full description of preprocessing, alignment, and feature extraction steps.   
     annotated-EDA  
   * `HOW_TO_RUN.md`: step-by-step instructions to reproduce analyses from raw EDF/CSV.

### **5.2 Stretch Deliverables (Time-Dependent)**

1. **Command-Following Feature Set**  
   * Bandpower features for left/right command paradigms with basic contrasts.  
2. **Integrated Feature Table for Exploratory Modeling**  
   * Combined table ready for regression or machine learning.  
3. **Prototype Prognostic Analyses**  
   * Simple correlation/regression analyses relating EEG features to any available clinical outcomes.  
4. **Visualization Suite**  
   * Publication-quality ERP plots, language ITPC spectra, topomaps, and QC dashboards (e.g., artifact rejection rates per subject).

---

## **6\. Project Schedule ([Link](https://docs.google.com/document/d/11wXvqzYv1DSrU1DBrrf_1hrRkZd1hdonsX7wnNpqExc/edit?tab=t.0#heading=h.ae6dqmvveeq8))**

We follow a **9-week timeline (January 10 – March 15\)** with 2-week sprints, aligned with our uploaded project schedule. 

### **6.1 Major Milestones**

* **Milestone 1 – Infrastructure & Data Harmonization (Weeks 1–2)**  
  * Repo setup, Teams channel, GitHub access.  
  * CSV schema unification and stimulus manifest.  
  * `EEGDataLoader` and timestamp alignment logic.  
* **Milestone 2 – End-to-End Pipeline & QC Validation (Target: Feb 13\)**  
  * Automatic preprocessing pipeline (raw EDF \+ CSV → artifact-rejected epochs).  
  * Initial oddball ERP extraction and language tracking plots on at least one control subject.  
  * QC HTML report summarizing trial counts and artifact rates per subject.   
* **Milestone 3 – Feature Extraction & Analysis (Weeks 6–7)**  
  * P300 and ITPC features extracted for all processed subjects.  
  * Aggregated feature table and exploratory statistics.  
* **Milestone 4 – Final Polish & Handoff (Weeks 8–9)**  
  * Final visualizations and documentation.  
  * Repository cleanup and packaging of data products.  
  * Final presentation and written report.

---

## **7\. Risks and Mitigation Strategies**

We explicitly incorporate risks identified during EDA and in our schedule.

1. **Schema Drift and Messy CSVs**  
   * *Risk:* Data harmonization may take longer than expected, delaying downstream work.  
   * *Mitigation:* Start `EEGDataLoader` development using mock or partial data while schema unification is underway; treat canonical CSV as a rolling contract.  
2. **Temporal Misalignment**  
   * *Risk:* Misaligned EDF–CSV timestamps could invalidate ERPs and ITPC analyses.  
   * *Mitigation:* Use multiple alignment checks (DC audio channel, expected durations) and flag suspicious trials for exclusion.  
3. **Small Sample Size & Heterogeneous Protocols**  
   * *Risk:* Limited ability to generalize or do robust statistical modeling; session-to-session differences.  
   * *Mitigation:* Focus on within-subject signal quality; document protocol differences; treat modeling as exploratory rather than confirmatory.  
4. **Artifact-Heavy EEG**  
   * *Risk:* Aggressive artifact rejection could yield too few usable trials for some paradigms.  
   * *Mitigation:* Start with permissive thresholds (for milestone validation), then tighten; always track and report the number of retained trials.  
5. **Compute and Data Governance**  
   * *Risk:* Large EDFs may strain local machines; improper cloud configuration could risk privacy.  
   * *Mitigation:* Use AWS only under sponsor guidance within healthcare-compliant settings; keep all PHI de-identified and within approved storage.  
6. **Dependency on External Code (Trisha’s Repository)**  
   * *Risk:* Legacy code may require major adaptation for the current dataset.  
   * *Mitigation:* Treat prior work as a reference implementation, not a dependency; re-implement critical components in our own pipeline where needed.

---

## **8\. References**

* Gott, P. S., Rabinowicz, A. L., & DeGiorgio, C. M. (1991). *P300 auditory event-related potentials in nontraumatic coma: Association with Glasgow Coma Score and awakening.* Archives of Neurology.  
* Sokoliuk, R., et al. (2021). *Covert speech comprehension predicts recovery from acute unresponsive states.* Annals of Neurology.  
* Awaken AI Capstone Team. (2025). *EEG Prognostic Data Pipeline & Characterization Report.* Internal project EDA document.  
* Awaken AI Capstone Team. (2025). *EEG Prognostic Data Pipeline: Project Schedule.* Internal planning document.

## 