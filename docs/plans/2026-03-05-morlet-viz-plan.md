# Morlet ITPC Visualizations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Morlet ITPC frequency-spectrum and topographic plots alongside existing DFT plots, with proper permutation p-values derived from per-trial Morlet phase data.

**Architecture:** Three self-contained changes applied in order — (1) rename and generalize the existing DFT visualization functions to be method-agnostic, (2) store Morlet per-trial phase information in the pipeline and compute Morlet permutation p-values, (3) generate Morlet spectrum/topomap plots and add them to the report HTML.

**Tech Stack:** MNE-Python (`tfr_morlet`, `AverageTFR`), NumPy, Matplotlib, Python 3.10

---

## Context

All commands run from `Repository/awaken-ai/`. Tests: `pytest tests/ -v`. Lint: `ruff check --fix . && ruff format .`.

Key files:
- `src/viz/language_plots.py` — plot functions
- `src/pipelines/language_tracking.py` — pipeline class (`LanguageTrackingAnalysis`)
- `src/reports/language_tracking_report.py` — HTML report class
- `tests/test_language_tracking_report.py` — report tests
- `tasks/report.md` — pipeline documentation

Key constants in `LanguageTrackingAnalysis`:
- `ITPC_FREQS = np.logspace(np.log10(0.5), np.log10(5.0), num=60)` — 60 log-spaced frequency bins used for Morlet
- `ITPC_CYCLES = np.array([max(0.5, f * 2.0) for f in ITPC_FREQS])`
- `TARGET_SENTENCE_FREQ = 0.78`, `TARGET_PHRASE_FREQ = 1.56`, `TARGET_WORD_FREQ = 3.125`

---

## Task 1: Rename `plot_dft_spectrum` → `plot_itpc_spectrum` and `plot_dft_topomap` → `plot_itpc_topomap`

**Files:**
- Modify: `src/viz/language_plots.py:157-308`
- Modify: `src/reports/language_tracking_report.py:13`
- Modify: `tests/test_language_tracking_report.py:65-94`

### Step 1: Write the failing import test

Add at top of `tests/test_language_tracking_report.py` (before existing imports), then run to verify failure:

```python
from src.viz.language_plots import plot_itpc_spectrum, plot_itpc_topomap  # noqa: F401
```

Run: `pytest tests/test_language_tracking_report.py -v`
Expected: `ImportError: cannot import name 'plot_itpc_spectrum'`

### Step 2: Rename functions in `src/viz/language_plots.py`

In `language_plots.py`, find the two functions:
- `def plot_dft_spectrum(` at line 157 → rename to `def plot_itpc_spectrum(`
- `def plot_dft_topomap(` at line 233 → rename to `def plot_itpc_topomap(`

Add `method_label: str = "DFT"` as a new parameter to **both** functions (insert after `output_dir: str` in the signature).

In `plot_itpc_spectrum`, update:
- Title: `f"{patient_id}: {method_label} ITPC Frequency Spectrum"` (was `f"{patient_id}: DFT ITPC Frequency Spectrum"`)
- `out_path` filename: `f"{patient_id}_lang_{method_label.lower()}_spectrum.png"` (was `f"{patient_id}_lang_dft_spectrum.png"`)

In `plot_itpc_topomap`, update:
- `out_path` filename: `f"{patient_id}_lang_topomap_{method_label.lower()}_{safe_label}.png"` (was `f"{patient_id}_lang_topomap_{safe_label}.png"`)

Full updated signatures:
```python
def plot_itpc_spectrum(
    itpc_spectrum: np.ndarray,
    freqs: np.ndarray,
    patient_id: str,
    output_dir: str,
    metrics: dict,
    method_label: str = "DFT",
) -> Path:

def plot_itpc_topomap(
    itpc_spectrum: np.ndarray,
    freqs: np.ndarray,
    info: "mne.Info",
    target_freq: float,
    label: str,
    patient_id: str,
    output_dir: str,
    vlim: tuple = None,
    method_label: str = "DFT",
) -> Path:
```

### Step 3: Update the import in `src/reports/language_tracking_report.py`

Change line 13:
```python
# Before:
from src.viz.language_plots import plot_dft_spectrum, plot_dft_topomap, plot_itpc_results

# After:
from src.viz.language_plots import plot_itpc_results, plot_itpc_spectrum, plot_itpc_topomap
```

In the same file, update `_save_plots` method (~line 96):
```python
# Before:
paths["dft_spectrum"] = plot_dft_spectrum(spectrum, freqs, pid, self.output_dir, metrics)
# ...
paths[f"topomap_{label.lower()}"] = plot_dft_topomap(
    spectrum, freqs, info, freq, label, pid, self.output_dir, vlim=vlim
)

# After:
paths["dft_spectrum"] = plot_itpc_spectrum(spectrum, freqs, pid, self.output_dir, metrics)
# ...
paths[f"topomap_{label.lower()}"] = plot_itpc_topomap(
    spectrum, freqs, info, freq, label, pid, self.output_dir, vlim=vlim
)
```

### Step 4: Update test patches in `tests/test_language_tracking_report.py`

In all `patch(...)` calls, update the patched names:
```python
# Before (3 places):
patch("src.reports.language_tracking_report.plot_dft_spectrum", ...)
patch("src.reports.language_tracking_report.plot_dft_topomap", ...)

# After:
patch("src.reports.language_tracking_report.plot_itpc_spectrum", ...)
patch("src.reports.language_tracking_report.plot_itpc_topomap", ...)
```

### Step 5: Run tests and lint

```bash
cd "Repository/awaken-ai" && pytest tests/test_language_tracking_report.py -v
```
Expected: 4 tests pass.

```bash
ruff check --fix . && ruff format .
```

### Step 6: Commit

```bash
git add src/viz/language_plots.py src/reports/language_tracking_report.py tests/test_language_tracking_report.py
git commit -m "refactor: rename plot_dft_spectrum/topomap to plot_itpc_spectrum/topomap with method_label param"
```

---

## Task 2: Store Morlet per-trial phases and compute Morlet permutation p-values

**Files:**
- Modify: `src/pipelines/language_tracking.py:93-260` (init, analyze, compute_trial_shuffled_null_itpc)

### Step 1: Write the failing test

Add a new test file `tests/test_language_tracking_morlet_pvals.py`:

```python
"""Tests for Morlet permutation p-values in LanguageTrackingAnalysis."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import mne

from src.pipelines.language_tracking import LanguageTrackingAnalysis


@pytest.fixture
def pipeline_with_phases():
    """Pipeline instance with synthetic _morlet_phases stored."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    rng = np.random.default_rng(0)
    # Shape: (n_trials=10, n_channels=5, n_target_freqs=3)
    lt._morlet_phases = rng.uniform(-np.pi, np.pi, size=(10, 5, 3))
    return lt


def test_compute_null_morlet_returns_correct_shape(pipeline_with_phases):
    """compute_trial_shuffled_null_itpc(method='morlet') returns (n_permutations,)."""
    null = pipeline_with_phases.compute_trial_shuffled_null_itpc(
        epochs=None, n_permutations=50, metric="word", seed=0, method="morlet"
    )
    assert null.shape == (50,)
    assert np.all(null >= 0) and np.all(null <= 1)


def test_compute_null_morlet_all_metrics(pipeline_with_phases):
    """All metric types work for morlet method."""
    for metric in ("word", "phrase", "sentence", "comprehension"):
        null = pipeline_with_phases.compute_trial_shuffled_null_itpc(
            epochs=None, n_permutations=20, metric=metric, seed=0, method="morlet"
        )
        assert null.shape == (20,)


def test_compute_null_morlet_raises_if_no_phases():
    """Raises ValueError when _morlet_phases is not stored."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    lt._morlet_phases = None
    with pytest.raises(ValueError, match="_morlet_phases"):
        lt.compute_trial_shuffled_null_itpc(
            epochs=None, n_permutations=10, metric="word", seed=0, method="morlet"
        )


def test_compute_null_invalid_method():
    """Raises ValueError for unknown method."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    with pytest.raises(ValueError, match="method"):
        lt.compute_trial_shuffled_null_itpc(
            epochs=None, n_permutations=10, metric="word", seed=0, method="unknown"
        )
```

Run: `pytest tests/test_language_tracking_morlet_pvals.py -v`
Expected: `AttributeError` or `TypeError` — function doesn't have `method` param yet.

### Step 2: Add `_morlet_phases` to `__init__`

In `LanguageTrackingAnalysis.__init__` (around line 116), add after `self._morlet_itc = None`:

```python
self._morlet_phases: Optional[np.ndarray] = None  # (n_trials, n_channels, 3) at target freq bins
```

### Step 3: Store Morlet phases in `analyze()`

In the `analyze` method (around line 173–176), after the existing `compute_itpc` call that stores `self._morlet_itc`, add a second `tfr_morlet` call to get per-trial complex data at the three target bins only.

Find this block (around line 173):
```python
# 2. Compute Morlet ITPC on clinical epochs
logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
itpc_data_morlet, itc_obj = self.compute_itpc(clinical_epochs)
self._morlet_itc = itc_obj
morlet_metrics = self.extract_itpc_metrics(itpc_data_morlet)
```

Replace with:
```python
# 2. Compute Morlet ITPC on clinical epochs
logger.info(f"[{self.patient_id}] Computing Morlet ITPC...")
itpc_data_morlet, itc_obj = self.compute_itpc(clinical_epochs)
self._morlet_itc = itc_obj
morlet_metrics = self.extract_itpc_metrics(itpc_data_morlet)

# Store per-trial Morlet phases at the 3 target bins for permutation tests.
# A second tfr_morlet call with average=False, output='complex' at only 3
# frequencies is cheap — far fewer frequency bins than the full 60-bin call above.
self._morlet_phases = self._compute_morlet_target_phases(clinical_epochs)
```

Add the private helper method `_compute_morlet_target_phases` to the class (place it just before `compute_trial_shuffled_null_itpc`):

```python
def _compute_morlet_target_phases(self, epochs: mne.Epochs) -> np.ndarray:
    """
    Compute per-trial Morlet phase angles at the three target frequency bins.

    Runs tfr_morlet with average=False, output='complex' restricted to only
    three frequencies (word, phrase, sentence) to avoid storing the full
    60-frequency complex array.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs.

    Returns
    -------
    np.ndarray, shape (n_trials, n_channels, 3)
        Phase angles (radians) at [word, phrase, sentence] frequencies.
        Axis-2 order: [0]=word, [1]=phrase, [2]=sentence.
    """
    from mne.time_frequency import tfr_morlet

    target_freqs = np.array(
        [self.TARGET_WORD_FREQ, self.TARGET_PHRASE_FREQ, self.TARGET_SENTENCE_FREQ]
    )
    n_cycles = np.array([max(0.5, f * 2.0) for f in target_freqs])

    # Returns EpochsTFR with shape (n_trials, n_channels, 3, n_times)
    epoch_tfr = tfr_morlet(
        epochs,
        freqs=target_freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        output="complex",
        average=False,
        n_jobs=-1,
    )
    # epoch_tfr.data: (n_trials, n_channels, 3, n_times)
    # Time-average the angle (take mean complex vector, then angle)
    # Simpler: just average the phases across time for each (trial, channel, freq)
    complex_data = epoch_tfr.data  # (n_trials, n_channels, 3, n_times)
    # Mean complex across time -> (n_trials, n_channels, 3)
    mean_complex = np.mean(complex_data, axis=-1)
    return np.angle(mean_complex)
```

### Step 4: Generalize `compute_trial_shuffled_null_itpc` with `method` parameter

Find the function signature at line 647:
```python
def compute_trial_shuffled_null_itpc(
    self,
    epochs: mne.Epochs,
    n_permutations: int = 1000,
    metric: str = "word",
    seed: int = 42,
) -> np.ndarray:
```

Replace with:
```python
def compute_trial_shuffled_null_itpc(
    self,
    epochs: mne.Epochs,
    n_permutations: int = 1000,
    metric: str = "word",
    seed: int = 42,
    method: str = "dft",
) -> np.ndarray:
```

Add a routing block at the top of the method body (before the existing DFT logic), after the `rng = np.random.default_rng(seed)` line:

```python
if method == "morlet":
    return self._compute_morlet_null_itpc(n_permutations, metric, rng)
elif method != "dft":
    raise ValueError(f"Unknown method '{method}'. Use 'dft' or 'morlet'.")
```

Add the private helper method `_compute_morlet_null_itpc` just after `compute_trial_shuffled_null_itpc`:

```python
def _compute_morlet_null_itpc(
    self,
    n_permutations: int,
    metric: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate null Morlet ITPC via trial-level random phase scrambling.

    Uses stored `_morlet_phases` (set by `_compute_morlet_target_phases`).
    Axis-2 order in _morlet_phases: [0]=word, [1]=phrase, [2]=sentence.

    Parameters
    ----------
    n_permutations : int
        Number of surrogates.
    metric : str
        "word", "phrase", "sentence", or "comprehension".
    rng : np.random.Generator
        Seeded random generator.

    Returns
    -------
    np.ndarray, shape (n_permutations,)
    """
    if self._morlet_phases is None:
        raise ValueError(
            "_morlet_phases not set. Call analyze() before running Morlet permutation tests."
        )

    phases = self._morlet_phases  # (n_trials, n_channels, 3)
    n_trials, n_channels, _ = phases.shape

    # FREQ_IDX: axis-2 index for each target
    FREQ_IDX = {"word": 0, "phrase": 1, "sentence": 2}

    def surrogate_itpc(freq_idx: int) -> np.ndarray:
        """Compute null ITPC for a single frequency bin."""
        unit_vectors = np.exp(1j * phases[:, :, freq_idx])  # (n_trials, n_channels)
        # Random phase offsets per trial, identical across channels
        rand_phase = rng.uniform(0, 2 * np.pi, size=(n_permutations, n_trials, 1))
        shifted = unit_vectors * np.exp(1j * rand_phase)  # (n_permutations, n_trials, n_channels)
        return np.mean(np.abs(np.mean(shifted, axis=1)), axis=1)  # (n_permutations,)

    if metric in FREQ_IDX:
        return surrogate_itpc(FREQ_IDX[metric])
    elif metric == "comprehension":
        return (surrogate_itpc(FREQ_IDX["sentence"]) + surrogate_itpc(FREQ_IDX["phrase"])) / 2.0
    else:
        raise ValueError(f"Unknown metric '{metric}'")
```

### Step 5: Add Morlet p-values to `analyze()` and result DataFrame

After the existing permutation block (around line 207–222 in `analyze()`), add Morlet permutation calls and then add the columns to `result_dict`.

After the existing p-value computation lines, add:
```python
# Morlet permutation tests
logger.info(f"[{self.patient_id}] Running Morlet permutation tests ({n_permutations} surrogates)...")
morlet_null_sentence = self.compute_trial_shuffled_null_itpc(
    None, n_permutations, metric="sentence", seed=46, method="morlet"
)
morlet_null_phrase = self.compute_trial_shuffled_null_itpc(
    None, n_permutations, metric="phrase", seed=47, method="morlet"
)
morlet_null_word = self.compute_trial_shuffled_null_itpc(
    None, n_permutations, metric="word", seed=48, method="morlet"
)
morlet_null_comp = self.compute_trial_shuffled_null_itpc(
    None, n_permutations, metric="comprehension", seed=49, method="morlet"
)

morlet_p_sentence = self.compute_permutation_pvalue(morlet_metrics["itpc_sentence"], morlet_null_sentence)
morlet_p_phrase = self.compute_permutation_pvalue(morlet_metrics["itpc_phrase"], morlet_null_phrase)
morlet_p_word = self.compute_permutation_pvalue(morlet_metrics["itpc_word"], morlet_null_word)
morlet_p_comprehension = self.compute_permutation_pvalue(
    (morlet_metrics["itpc_sentence"] + morlet_metrics["itpc_phrase"]) / 2.0, morlet_null_comp
)
```

In `result_dict`, add after `"dft_p_comprehension": p_comprehension`:
```python
"morlet_p_word": morlet_p_word,
"morlet_p_phrase": morlet_p_phrase,
"morlet_p_sentence": morlet_p_sentence,
"morlet_p_comprehension": morlet_p_comprehension,
```

### Step 6: Run tests and lint

```bash
pytest tests/test_language_tracking_morlet_pvals.py -v
```
Expected: 4 tests pass.

```bash
ruff check --fix . && ruff format .
```

### Step 7: Commit

```bash
git add src/pipelines/language_tracking.py tests/test_language_tracking_morlet_pvals.py
git commit -m "feat: store Morlet per-trial phases and compute morlet permutation p-values"
```

---

## Task 3: Add Morlet spectrum and topomaps to the report

**Files:**
- Modify: `src/reports/language_tracking_report.py`
- Modify: `tests/test_language_tracking_report.py`
- Modify: `tasks/report.md`

### Step 1: Write the failing test

Add to `tests/test_language_tracking_report.py`:

```python
def test_report_html_contains_morlet_section(mock_pipeline):
    """Report HTML contains a Morlet ITPC section when _morlet_itc is set."""
    # Set up a minimal AverageTFR-like object
    morlet_itc = MagicMock()
    n_ch, n_freqs, n_times = 7, 60, 50
    morlet_itc.data = np.random.rand(n_ch, n_freqs, n_times) * 0.1
    morlet_itc.freqs = np.logspace(np.log10(0.5), np.log10(5.0), num=n_freqs)
    mock_pipeline._morlet_itc = morlet_itc
    mock_pipeline.results.iloc[0]["morlet_p_word"] = 0.01
    mock_pipeline.results.iloc[0]["morlet_p_phrase"] = 0.05
    mock_pipeline.results.iloc[0]["morlet_p_sentence"] = 0.02
    mock_pipeline.results.iloc[0]["morlet_p_comprehension"] = 0.03

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reports.language_tracking_report.plot_itpc_spectrum", return_value=Path(tmpdir) / "s.png"),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            path = rpt.generate()
            html = Path(path).read_text()
    assert "morlet" in html.lower()
```

Also update the existing `mock_pipeline` fixture to add `morlet_p_*` columns to `pipeline.results`:

```python
# Add to the dict in mock_pipeline fixture (existing tests have _morlet_itc=None, so Morlet section is skipped):
"morlet_p_word": 0.01,
"morlet_p_phrase": 0.04,
"morlet_p_sentence": 0.02,
"morlet_p_comprehension": 0.03,
```

Run: `pytest tests/test_language_tracking_report.py::test_report_html_contains_morlet_section -v`
Expected: fail — Morlet section not yet in HTML.

### Step 2: Add Morlet plots to `_save_plots` in the report

In `src/reports/language_tracking_report.py`, inside `_save_plots`, after the existing Morlet TFR block (around line 107), add:

```python
if self.lt_obj._morlet_itc is not None:
    try:
        # Time-average to get (n_channels, n_freqs) spectrum for Morlet
        morlet_data = self.lt_obj._morlet_itc.data  # (n_channels, n_freqs, n_times)
        morlet_spectrum = np.mean(morlet_data, axis=-1)  # (n_channels, n_freqs)
        morlet_freqs = self.lt_obj._morlet_itc.freqs

        morlet_metrics = {
            "itpc_word": float(row.get("morlet_itpc_word", 0)),
            "itpc_phrase": float(row.get("morlet_itpc_phrase", 0)),
            "itpc_sentence": float(row.get("morlet_itpc_sentence", 0)),
            "dft_p_word": float(row.get("morlet_p_word", 1)),
            "dft_p_phrase": float(row.get("morlet_p_phrase", 1)),
            "dft_p_sentence": float(row.get("morlet_p_sentence", 1)),
        }
        paths["morlet_spectrum"] = plot_itpc_spectrum(
            morlet_spectrum, morlet_freqs, pid, self.output_dir, morlet_metrics, method_label="Morlet"
        )

        morlet_word_idx = int(np.argmin(np.abs(morlet_freqs - LanguageTrackingAnalysis.TARGET_WORD_FREQ)))
        morlet_vmax = float(np.percentile(morlet_spectrum[:, morlet_word_idx], 95)) * 1.2 or 0.1
        morlet_vlim = (0.0, morlet_vmax)

        for freq, label in _TARGET_FREQS:
            paths[f"morlet_topomap_{label.lower()}"] = plot_itpc_topomap(
                morlet_spectrum,
                morlet_freqs,
                info,
                freq,
                label,
                pid,
                self.output_dir,
                vlim=morlet_vlim,
                method_label="Morlet",
            )
    except Exception as e:
        logger.warning(f"Morlet spectrum/topomap plots failed: {e}")
```

Note: `morlet_metrics` uses `"dft_p_word"` etc. as keys because `plot_itpc_spectrum` reads those keys from the metrics dict. This is intentional — the plot function is method-agnostic about key names.

### Step 3: Add Morlet section to `_build_plots_section`

In `_build_plots_section` (around line 237), after the existing `if "morlet_tfr"` block, add:

```python
if "morlet_spectrum" in plot_paths:
    img = self._embed_image(plot_paths["morlet_spectrum"], "Morlet ITPC Frequency Spectrum")
    sections.append(
        "<h3>Cortical Tracking Frequency Spectrum (Morlet)</h3>"
        f"<div class='plot-card'>{img}"
        "<figcaption>Time-averaged Morlet ITPC across 0.5&ndash;4 Hz. "
        "Dashed lines mark word (3.125 Hz), phrase (1.56 Hz), sentence (0.78 Hz). "
        "Broader peaks than DFT reflect the Morlet wavelet's time-frequency trade-off."
        "</figcaption></div>"
    )

morlet_topo_html = ""
for freq, label in _TARGET_FREQS:
    key = f"morlet_topomap_{label.lower()}"
    if key in plot_paths:
        img = self._embed_image(plot_paths[key], f"Morlet {label} Topomap")
        morlet_topo_html += (
            f"<div class='plot-card'>{img}"
            f"<figcaption>Morlet ITPC Topomap @ {freq} Hz ({label} rate).</figcaption></div>"
        )
if morlet_topo_html:
    sections.append(
        f"<h3>ITPC Topographic Maps (Morlet)</h3><div class='plot-grid'>{morlet_topo_html}</div>"
    )
```

### Step 4: Run all report tests and lint

```bash
pytest tests/test_language_tracking_report.py -v
```
Expected: all 5 tests pass.

```bash
ruff check --fix . && ruff format .
```

### Step 5: Update `tasks/report.md` documentation

In the "Language Tracking Report — Sections" table (around line 186 of `tasks/report.md`), add rows for Morlet frequency spectrum and Morlet topomap plots:

| Section | Content |
|---|---|
| ... existing rows ... |
| Morlet frequency spectrum | Time-averaged Morlet ITPC vs 0.5–4 Hz with annotated target lines and Morlet p-values |
| Morlet topomap plots | Spatial Morlet ITPC maps at word (3.125 Hz), phrase (1.56 Hz), sentence (0.78 Hz) |

Also update the `LanguageTrackingReport` constructor docs to note `_morlet_phases` as a new optional pipeline attribute.

### Step 6: Run full test suite and lint

```bash
pytest tests/ -v
ruff check --fix . && ruff format .
```
Expected: all tests pass, no lint errors.

### Step 7: Commit

```bash
git add src/reports/language_tracking_report.py tests/test_language_tracking_report.py tasks/report.md
git commit -m "feat: add Morlet ITPC frequency spectrum and topomap sections to language tracking report"
```

---

## Final validation

Run the pipeline end-to-end for one patient (requires local EEG data):

```bash
python -m src.cli.main language --patient CON008 --report
```

Confirm the output HTML contains both DFT and Morlet ITPC sections.

Run the full test suite one last time:

```bash
pytest tests/ -v && ruff check . && ruff format --check .
```
