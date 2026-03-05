# Morlet ITPC Visualizations Design

Date: 2026-03-05

## Background

The Language Tracking pipeline currently produces two analysis paths:

- **DFT ITPC**: per-trial zero-padded FFT -> phase angles -> averaged spectrum, shape `(n_channels, n_freqs)`. Already has a frequency-spectrum line plot and per-frequency topomaps in the report.
- **Morlet ITPC**: `tfr_morlet` with `average=True` -> `AverageTFR`, shape `(n_channels, 60_log_freqs, n_times)`. Currently only used for a time-frequency plot.

The goal is to give Morlet the same visualization treatment as DFT: a frequency-spectrum plot (time-averaged) and per-frequency topomaps, with proper permutation p-values.

---

## Design

### 1. Rename visualization functions (clean API)

`src/viz/language_plots.py`

| Old name | New name | Change |
|---|---|---|
| `plot_dft_spectrum` | `plot_itpc_spectrum` | Add `method_label: str` parameter; use it in title and filename |
| `plot_dft_topomap` | `plot_itpc_topomap` | Add `method_label: str` parameter; use it in title and filename |

Callers updated: `src/reports/language_tracking_report.py`, `tests/test_language_tracking_report.py`.

Filename pattern after rename:
- `{patient_id}_lang_{method_label.lower()}_spectrum.png`
- `{patient_id}_lang_topomap_{method_label.lower()}_{freq_label.lower()}.png`

### 2. Morlet permutation p-values in the pipeline

`src/pipelines/language_tracking.py`

**Problem**: `_morlet_itc` is an `AverageTFR` (trials already averaged). Per-trial complex data is discarded. We need per-trial phase information to run permutation tests.

**Solution**: After `tfr_morlet(..., average=True)` (for the TFR report), run a second call with `average=False, output='complex'` at only the three target frequency bins. Extract angle phases from complex output, store as `self._morlet_phases` (shape: `n_trials x n_channels x 3_freqs`).

**Generalize `compute_trial_shuffled_null_itpc`**:

```python
def compute_trial_shuffled_null_itpc(
    self,
    epochs,
    n_permutations: int = 500,
    metric: str = "comprehension",
    seed: int = 42,
    method: str = "dft",       # new: "dft" or "morlet"
) -> np.ndarray:
```

- `method="dft"`: existing behavior (rfft on epoch data).
- `method="morlet"`: use `self._morlet_phases` (already stored), shuffle trial axis, compute `|mean(exp(i*phase))|` per permutation.

Add to pipeline output DataFrame:
- `morlet_p_word`, `morlet_p_phrase`, `morlet_p_sentence`, `morlet_p_comprehension`

### 3. Morlet spectrum + topomaps in the report

`src/reports/language_tracking_report.py`

**Morlet spectrum**: time-average `pipeline._morlet_itc.data` over the time axis -> shape `(n_channels, n_freqs)`. Pass to `plot_itpc_spectrum(..., method_label="Morlet")` with `metrics` containing `morlet_p_*` values.

**Morlet topomaps**: call `plot_itpc_topomap` three times (word/phrase/sentence) using the time-averaged Morlet data and `method_label="Morlet"`.

**Report HTML**: add a "Morlet ITPC Frequency Analysis" section alongside the existing "DFT ITPC Frequency Analysis" section. Both use the same plot-grid layout.

---

## Data Flow

```
tfr_morlet(average=False, output='complex') at 3 target bins
  -> _morlet_phases (n_trials, n_channels, 3)
  -> compute_trial_shuffled_null_itpc(method="morlet")
  -> morlet_p_word/phrase/sentence/comprehension

tfr_morlet(average=True)
  -> _morlet_itc.data (n_channels, 60, n_times)
  -> time-average -> (n_channels, 60)
  -> plot_itpc_spectrum(method_label="Morlet")
  -> plot_itpc_topomap x3 (method_label="Morlet")
```

---

## Out of Scope

- DFT time-frequency representation (TFR): DFT is naturally a time-collapsed method; adding artificial temporal segmentation adds complexity with no clinical value.
- Changes to existing DFT permutation tests or DFT spectrum/topomap logic (beyond renaming).

---

## Files Changed

| File | Change |
|---|---|
| `src/viz/language_plots.py` | Rename two functions, add `method_label` param |
| `src/pipelines/language_tracking.py` | Store `_morlet_phases`, generalize permutation fn, add `morlet_p_*` columns |
| `src/reports/language_tracking_report.py` | Add Morlet spectrum + topomap plots and HTML section |
| `tests/test_language_tracking_report.py` | Update mock function names |
| `tasks/report.md` | Update documentation |
