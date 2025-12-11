# 📅 EEG Prognostic Data Pipeline: Project Schedule

**Project Duration:** January 10, 2026 – March 15, 2026 (9 Weeks)  
**Methodology:** Agile / GitHub Projects (2-week Sprints)  
**Team Size:** 5 Members (Aaditya Chopra, B, C, D, + Alex Diamond)

---

## 🎯 Intermediate Milestone
**Target Date:** **Friday, February 13, 2026**  
**Milestone Name:** **End-to-End Pipeline & QC Validation**  
**Measurable Success Criteria:**
1.  **Unified Dataset:** All disjointed CSV logs merged into a single, conflict-free master schema.
2.  **Automated Preprocessing:** Pipeline accepts raw `.EDF` + `.CSV` and outputs clean, artifact-rejected epochs (HDF5).
3.  **QC Dashboard:** HTML report generated showing artifact rejection rates per patient.
4.  **Proof-of-Concept Plots:** "Grand Average" plots confirming presence of P300 peaks (Oddball) and Alpha suppression (Command) in at least one healthy subject (or best dataset).

---

## 🗓️ Phase 1: Infrastructure & Data Harmonization (Weeks 1-2)
*Focus: Setting up the repo and fixing the critical "Schema Drift" and file inventory issues.*

**Sprint 1 Dates:** Jan 10 – Jan 24

| ID | Task Name | Assignee | Due Date | Dep | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INF-01** | **Repo Init & Environment** | Aaditya Chopra | Jan 12 | - | Git setup, send GitHub usernames to Peter for write access. Setup Teams channel (invite Peter & Alex Diamond). Assess AWS compute needs. |
| **DAT-01** | **Inventory & Sync Script** | Aaditya Chopra | Jan 15 | INF-01 | Script to verify local OneDrive files against master list. Log missing files (e.g., `lang28.wav`). |
| **DAT-02** | **Stimulus Manifest** | Ayush Mall | Jan 17 | - | Create `stimulus_manifest.csv` mapping audio filenames to transcript/duration. |
| **DAT-03** | **CSV Schema Unification** | Ayush Mall | Jan 20 | DAT-02 | **CRITICAL.** Script to merge `patient_df*.csv` variants. Standardize column names (`trial_type`, `start_time`). |
| **ENG-01** | **Base Data Loader** | Arnav Dixit | Jan 22 | DAT-03 | Python class `EEGDataLoader` that reads specific EDFs and links them to the harmonized CSV. |
| **ENG-02** | **Timestamp Alignment** | Arnav Dixit | Jan 24 | ENG-01 | Logic to sync CSV Unix timestamps with EDF internal clocks using the **DC audio input channel** for precise alignment. |
| **ENG-02b** | **ERP/Oddball Pipeline** | Arnav Dixit | Jan 30 | ENG-02 | Construct pipeline to identify "deviant" beeps, extract 500-700ms epochs, and average segments to reveal P300 ERP. |
| **DAT-04** | **Clinical Metadata Ingest** | Riddhesh Sawant | Jan 24 | - | Digitize `patient_notes.csv` and `history` into a structured JSON/Pandas format. |

---

## 🗓️ Phase 2: Signal Processing & Pipeline Core (Weeks 3-5)
*Focus: Building the engine that processes signals. Goal is to hit the Feb 13 Milestone.*

**Sprint 2 Dates:** Jan 25 – Feb 13

| ID | Task Name | Assignee | Due Date | Dep | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ENG-03** | **Artifact Rejection (ICA)** | Riddhesh Sawant | Feb 01 | ENG-02 | Implement MNE ICA to remove eye blinks/muscle noise. Automated rejection thresholds. |
| **ENG-04** | **Command Epoching** | Aaditya Chopra | Feb 04 | ENG-02 | Slicing logic for Motor Command blocks (200s duration). Bandpass filtering (8-30Hz). |
| **ENG-05** | **Language Optimization** | Ayush Mall | Feb 04 | ENG-02 | Isolate language trial segments. Analyze neural activity at sentence frequency. optimize for ~20 electrode setup (focus on left-hemisphere). |
| **ENG-06** | **QC Report Generation** | Riddhesh Sawant | Feb 08 | ENG-03 | Script generating HTML table: Dropped epochs count, SNR metrics per file. |
| **VIS-01** | **Validation Plotting** | Arnav Dixit | Feb 10 | ENG-05 | Generate ERP waveforms (Grand Average) for P300 validation. |
| **MST-01** | **Milestone Review** | **ALL** | **Feb 13** | **ALL** | **Code freeze.** Run full pipeline on CON008/CON009. Verify outputs match Success Criteria. |

---

## 🗓️ Phase 3: Feature Extraction & Analysis (Weeks 6-7)
*Focus: Deriving meaning from the clean signals.*

**Sprint 3 Dates:** Feb 14 – Feb 28

| ID | Task Name | Assignee | Due Date | Dep | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SCI-01** | **P300 Peak Features** | Arnav Dixit | Feb 18 | MST-01 | Extract amplitude/latency of max peak in 300-600ms window (Oddball). |
| **SCI-02** | **Spectral Power (PSD)** | Ayush Mall | Feb 20 | MST-01 | Compute Alpha/Beta power changes for Command Following (Left vs Right). |
| **SCI-03** | **Language Tracking (ITPC)** | Riddhesh Sawant | Feb 22 | MST-01 | Calculate Inter-Trial Phase Coherence for language trials. |
| **MOD-01** | **Feature Assembly** | Aaditya Chopra | Feb 25 | SCI-01+ | Aggregate all features into `features.parquet` (Rows=Trials, Cols=Features). |
| **MOD-02** | **Exploratory Stats** | Aaditya Chopra | Feb 28 | MOD-01 | Correlation analysis: Do features cluster by patient? By outcome? |

---

## 🗓️ Phase 4: Reporting & Final Polish (Weeks 8-9)
*Focus: wrapping up for the March 15 deadline.*

**Sprint 4 Dates:** Mar 01 – Mar 15

| ID | Task Name | Assignee | Due Date | Dep | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VIS-02** | **Final Visualizations** | Riddhesh Sawant | Mar 05 | MOD-02 | Publication-quality plots: Topomaps, PSD spectra, ITPC polar plots. |
| **DOC-01** | **Documentation Update** | Aaditya Chopra | Mar 08 | - | Update `DATA_PIPELINE_DOCUMENTATION.md` with final processing steps. |
| **DOC-02** | **Pipeline Usage Guide** | Ayush Mall | Mar 10 | - | Write `HOW_TO_RUN.md` for future researchers (reproducibility). |
| **SUB-01** | **Final Package Handoff** | **ALL** | **Mar 15** | **ALL** | Clean repo, zip results, submit final proposal/project. |

---

## ⚠️ Risk Register

1.  **Time Crunch (Weeks 1-2):** The schedule allows only 2 weeks for data harmonization. If CSVs are messier than expected, **ENG-01** will be delayed.
    *   *Mitigation:* Arnav Dixit starts building the Loader using *dummy data* while Ayush Mall fixes the CSVs.
2.  **Milestone Failure (Feb 13):** If artifact rejection is too aggressive, we may end up with 0 usable epochs for the milestone.
    *   *Mitigation:* Set permissive thresholds for the milestone; refine strictness in Sprint 3.
3.  **Missing Files:** `CON006` or `lang28.wav` might break the loop.
    *   *Mitigation:* **DAT-01** must include a "skip on missing" flag in the config immediately.

---

## 📋 Team Roles

*   **Aaditya Chopra (Team Lead):** Architecture, Integration, Risk Management.
*   **Ayush Mall (Data Engineer):** Wrangling CSVs, Spectral Analysis.
*   **Arnav Dixit (Signal Processing):** ERP Pipeline (P300), Synchronization logic (DC channel alignment).
*   **Riddhesh Sawant (QC & Vis):** Artifact Rejection, QC Reports, Language Optimization.
*   **Alex Diamond (Clinical Lead):** Clinical relevance, validation of results against clinical expectations.

PS: The tasks assigned to team members are placeholders and can be changed at a later date.