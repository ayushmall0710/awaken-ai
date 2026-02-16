"""
ENG-03: Artifact Rejection (ICA)

This module applies session-level ICA-based artifact rejection and exports
fixed-window epochs per trial_type as MNE .fif files, plus QC metadata as Parquet.

Design constraints:
- Apply ICA once per EDF session (patient_id + date), then epoch all paradigms.
- Consume aligned events produced by ENG-02 (TimestampAligner) from:
  data/processed/aligned_events/{patient_id}_events.parquet
- Save:
  - epochs/{patient_id}/{date}/{trial_type}-epo.fif
  - qc/{patient_id}/{date}/eng03_qc.parquet
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.utils.signal_processing import exclude_non_eeg_channels
from src.utils.time_utils import detect_timezone_offset, unix_to_edf

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_SEC_BY_TRIAL_TYPE: Dict[str, float] = {
    "language": 16.0,
    "oddball": 35.0,
    "beep": 35.0,
    "control": 35.0,
    "left_command": 200.0,
    "right_command": 200.0,
}

DEFAULT_ICA_FILTER_HZ: Tuple[float, float] = (1.0, 40.0)
DEFAULT_REJECT_PTP_PERCENTILE: float = 95.0


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ICASummary:
    method: str
    n_components: int
    n_components_selected: Optional[int]
    excluded: List[int]
    eog_channels_used: List[str]
    eog_components: List[int]
    muscle_components: List[int]
    notes: List[str]


# ── Pure helpers (no class state) ────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for :func:`json.dumps` — handles NumPy scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _trial_type_window_sec(trial_type: str, fallback_duration: Optional[float]) -> Optional[float]:
    """Return fixed epoch-window length (seconds) for a given trial type."""
    tt = str(trial_type).lower().strip()
    if tt in WINDOW_SEC_BY_TRIAL_TYPE:
        return WINDOW_SEC_BY_TRIAL_TYPE[tt]
    if fallback_duration is None:
        return None
    try:
        dur = float(fallback_duration)
    except Exception:
        return None
    if 1.0 <= dur <= 600.0:
        return dur
    return None


def _find_eog_channels(raw: mne.io.BaseRaw) -> List[str]:
    """Detect EOG channels with progressive fallback.

    Priority order:
    1. Channels typed as EOG in the MNE info.
    2. Channels whose name contains ``EOG``.
    3. Infraorbital channels (IO1, IO2) — placed below the eyes for blink/movement.
    4. Frontal channels Fp1/Fp2 as surrogate EOG (standard heuristic).
    """
    # 1. Prefer explicitly-typed EOG channels.
    try:
        picks = mne.pick_types(raw.info, eog=True, exclude=[])
        if len(picks) > 0:
            return [raw.ch_names[i] for i in picks]
    except Exception:
        pass

    # 2. Fallback to common naming conventions.
    by_name = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if by_name:
        return by_name

    # 3. Infraorbital electrodes (IO1, IO2) — proper EOG reference if present.
    io_channels = [ch for ch in raw.ch_names if ch.upper() in {"IO1", "IO2"}]
    if io_channels:
        return io_channels

    # 4. Heuristic: use Fp1/Fp2 as surrogate EOG (documented fallback).
    return [ch for ch in raw.ch_names if ch.upper() in {"FP1", "FP2"}]


def _epoch_ptp_uv(epochs: mne.Epochs) -> np.ndarray:
    """
    Per-epoch max peak-to-peak amplitude across channels (microvolts).

    MNE stores EEG in Volts, so we multiply by 1e6.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    if data.size == 0:
        return np.array([], dtype=float)
    ptp_v = data.ptp(axis=2)  # (n_epochs, n_channels)
    max_ptp_v = ptp_v.max(axis=1)  # (n_epochs,)
    return max_ptp_v * 1e6


def _pick_eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Return channel indices for EEG-only picks, excluding non-EEG auxiliary channels.

    EDF files often label **all** channels as type ``eeg`` (the MNE default), so
    relying on :func:`mne.pick_types` alone lets EMG, ECG, respiratory, and other
    polysomnography channels leak into the EEG set.  We therefore **always** apply
    the keyword-based exclusion list from
    :data:`src.utils.signal_processing.NON_EEG_CHANNEL_KEYWORDS` on top of any
    type-based picks.
    """
    non_eeg_names = set(exclude_non_eeg_channels(raw))

    # Start with typed EEG picks, then subtract keyword-matched non-EEG channels.
    try:
        typed_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
        picks = np.array(
            [i for i in typed_picks if raw.ch_names[i] not in non_eeg_names],
            dtype=int,
        )
        if len(picks) > 0:
            return picks
    except Exception:
        pass

    # Fallback: all channels that are NOT in the non-EEG keyword list.
    return np.array(
        [i for i, ch in enumerate(raw.ch_names) if ch not in non_eeg_names],
        dtype=int,
    )


# ── Main class ───────────────────────────────────────────────────────────────


class ArtifactRejector:
    """
    ENG-03 driver: run ICA artifact rejection per session and export
    fixed-window EEG-only epochs + QC metadata.
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        use_clipped: bool = True,
        ica_filter_hz: Tuple[float, float] = DEFAULT_ICA_FILTER_HZ,
        reject_ptp_percentile: float = DEFAULT_REJECT_PTP_PERCENTILE,
        verbose: bool = False,
    ):
        self.loader = UnifiedDataLoader(data_root=data_root, verbose=verbose)
        self.use_clipped = use_clipped
        self.ica_filter_hz = ica_filter_hz
        self.reject_ptp_percentile = float(reject_ptp_percentile)
        self.verbose = verbose

        if not verbose:
            mne.set_log_level("WARNING")

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, patient_ids: List[str], save: bool = True) -> Dict[Tuple[str, str], Dict[str, Path]]:
        """
        Run ENG-03 for a list of patients.

        Returns:
            Mapping of ``(patient_id, date) → {trial_type: epochs_fif_path}``.
        """
        out: Dict[Tuple[str, str], Dict[str, Path]] = {}
        for patient_id in patient_ids:
            patient = self.loader.get_patient(patient_id)
            for date in patient.list_sessions():
                out[(patient_id, date)] = self.run_session(patient_id, date, save=save)
        return out

    def run_session(self, patient_id: str, date: str, save: bool = True) -> Dict[str, Path]:
        """
        Run ENG-03 for a single patient session.

        Steps:
        1. Load EDF + aligned events for this session.
        2. Apply session-level ICA (EEG channels only).
        3. Build fixed-window, EEG-only epochs per trial type.
        4. Auto-reject noisy epochs (percentile-based PTP).
        5. Save ``.fif`` epochs + QC ``.parquet``.
        """
        # ── 1. Load inputs ───────────────────────────────────────────────
        raw = self.loader.load_edf(patient_id, date=date, use_clipped=self.use_clipped)

        aligned_df = self._load_aligned_events(patient_id)
        session_df = aligned_df[aligned_df["date"] == date].copy()
        if session_df.empty:
            raise FileNotFoundError(
                f"No aligned events found for {patient_id} date={date}. "
                f"Expected {config.ALIGNED_EVENTS_DIR / f'{patient_id}_events.parquet'}"
            )

        # Reuse shared time-utils (same logic as ENG-02 TimestampAligner)
        timezone_offset = detect_timezone_offset(raw, session_df)
        edf_start_unix = raw.info["meas_date"].timestamp() if raw.info.get("meas_date") is not None else 0.0

        # ── 2. Session-level ICA ─────────────────────────────────────────
        raw_clean, ica_summary = self._apply_ica(raw)

        # ── 3 & 4. Build epochs per trial type ──────────────────────────
        saved_paths: Dict[str, Path] = {}
        qc_rows: List[Dict[str, Any]] = []

        for trial_type, tt_df in session_df.groupby(session_df["trial_type"].astype(str).str.lower()):
            epochs = self._build_fixed_window_epochs(
                raw_clean,
                tt_df,
                trial_type=trial_type,
                edf_start_unix=edf_start_unix,
                timezone_offset=timezone_offset,
            )

            if epochs is None or len(epochs) == 0:
                qc_rows.append(
                    self._qc_row(
                        patient_id=patient_id,
                        date=date,
                        trial_type=trial_type,
                        n_total=0,
                        n_dropped=0,
                        threshold_uv=None,
                        ica_summary=ica_summary,
                        notes=["no_epochs"],
                    )
                )
                continue

            n_total_before = len(epochs)
            ptp_uv_before = _epoch_ptp_uv(epochs)
            ptp_stats = self._ptp_stats(ptp_uv_before)
            epochs, threshold_uv, dropped = self._auto_reject_epochs(epochs)
            n_dropped = int(len(dropped))

            qc_rows.append(
                self._qc_row(
                    patient_id=patient_id,
                    date=date,
                    trial_type=trial_type,
                    n_total=n_total_before,
                    n_dropped=n_dropped,
                    threshold_uv=threshold_uv,
                    ica_summary=ica_summary,
                    notes=[],
                    ptp_stats=ptp_stats,
                    drop_reason=(
                        f"ENG03_PTP_GT_P{self.reject_ptp_percentile:g}"
                        if (dropped and threshold_uv is not None)
                        else None
                    ),
                )
            )

            # ── 5. Save ──────────────────────────────────────────────────
            if save:
                fif_path = self._epochs_output_path(patient_id, date, trial_type)
                fif_path.parent.mkdir(parents=True, exist_ok=True)
                epochs.save(fif_path, overwrite=True)
                saved_paths[trial_type] = fif_path

        if save:
            qc_path = self._qc_output_path(patient_id, date)
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(qc_rows).to_parquet(qc_path, index=False)

        return saved_paths

    # ── I/O helpers ──────────────────────────────────────────────────────

    def _load_aligned_events(self, patient_id: str) -> pd.DataFrame:
        path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Aligned events parquet not found: {path}")
        return pd.read_parquet(path)

    @staticmethod
    def _epochs_output_path(patient_id: str, date: str, trial_type: str) -> Path:
        safe_tt = str(trial_type).lower().strip()
        return config.EPOCHS_DIR / patient_id / date / f"{safe_tt}-epo.fif"

    @staticmethod
    def _qc_output_path(patient_id: str, date: str) -> Path:
        return config.QC_DIR / patient_id / date / "eng03_qc.parquet"

    # ── Core processing ──────────────────────────────────────────────────

    def _apply_ica(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, ICASummary]:
        """
        Fit ICA on **EEG-only** channels of a bandpass-filtered copy, then apply
        artifact-component subtraction to the original (unfiltered) Raw.

        Key design decisions
        --------------------
        * We keep *all* channels in ``raw_for_ica`` so that dedicated EOG channels
          (IO1/IO2, or Fp1/Fp2 as surrogates) remain available as reference for
          ``find_bads_eog``.  Only the ``picks_eeg`` subset is used for ICA fitting
          and filtering.
        * ``find_bads_eog`` uses ``threshold=2.5`` (lower than the MNE default of
          3.0) because surrogate-EOG channels produce weaker cross-correlations
          than dedicated EOG electrodes.
        """
        notes: List[str] = []

        # Identify non-EEG channels for diagnostic logging.
        non_eeg_names = exclude_non_eeg_channels(raw)
        if non_eeg_names:
            notes.append(f"non_eeg_channels={non_eeg_names}")

        # Full copy — we do NOT drop channels so that EOG reference channels
        # (IO1/IO2, Fp1/Fp2) remain available for find_bads_eog.
        raw_for_ica = raw.copy()

        # Pick EEG-only channels (keyword exclusion is always applied).
        picks_eeg = _pick_eeg_indices(raw_for_ica)
        notes.append(f"n_eeg_channels={len(picks_eeg)}")
        if len(picks_eeg) == 0:
            notes.append("no_eeg_channels_after_exclusion_fallback=all")
            picks_eeg = np.arange(len(raw_for_ica.ch_names))

        # Bandpass filter EEG channels only (for ICA stability).
        l_freq, h_freq = self.ica_filter_hz
        try:
            raw_for_ica.filter(l_freq=l_freq, h_freq=h_freq, picks=picks_eeg, verbose=self.verbose)
        except Exception as e:
            notes.append(f"filter_failed={type(e).__name__}:{e}")

        # Fit ICA on EEG-only picks.
        ica = mne.preprocessing.ICA(
            n_components=0.99,
            method="fastica",
            max_iter="auto",
            random_state=97,
        )
        ica.fit(raw_for_ica, picks=picks_eeg)

        # ── Auto-detect artifact components ─ EOG (eye blinks) ──────────
        eog_channels = _find_eog_channels(raw_for_ica)
        eog_components: List[int] = []
        eog_channels_used: List[str] = []

        if eog_channels:
            for ch in eog_channels:
                try:
                    inds, _scores = ica.find_bads_eog(
                        raw_for_ica,
                        ch_name=ch,
                        threshold=2.5,
                    )
                    eog_components.extend(list(inds))
                    eog_channels_used.append(ch)
                except Exception as e:
                    notes.append(f"find_bads_eog_ch={ch}_failed={type(e).__name__}:{e}")
            eog_components = sorted(set(eog_components))
        else:
            notes.append("no_eog_channels_detected")

        # Fallback: let MNE try its own EOG-channel synthesis if we found nothing.
        if not eog_components:
            try:
                inds, _scores = ica.find_bads_eog(raw_for_ica, threshold=2.5)
                if inds:
                    eog_components = sorted(set(map(int, inds)))
                    notes.append(f"eog_fallback_mne_auto=found_{len(inds)}")
            except Exception as e:
                notes.append(f"eog_fallback_failed={type(e).__name__}:{e}")

        # ── Auto-detect artifact components ─ muscle noise ───────────────
        # find_bads_muscle uses topography + slope when sensor positions are
        # available, but falls back to slope-only when they aren't (common with
        # clinical EDF files).  The slope-only criterion tends to be
        # over-aggressive, so we skip muscle detection when no digitization is
        # present to avoid discarding legitimate brain signal.
        muscle_components: List[int] = []
        has_dig = bool(raw_for_ica.info.get("dig"))
        try:
            if hasattr(ica, "find_bads_muscle") and has_dig:
                inds, _scores = ica.find_bads_muscle(raw_for_ica)
                muscle_components = sorted(set(map(int, inds)))
            elif not has_dig:
                notes.append("skip_muscle_detection_no_sensor_positions")
        except Exception as e:
            notes.append(f"find_bads_muscle_failed={type(e).__name__}:{e}")

        excluded_components = sorted(set(eog_components + muscle_components))
        ica.exclude = excluded_components

        if not excluded_components:
            notes.append("WARNING_no_components_excluded")

        # Apply ICA to original raw (MNE subtracts excluded components from
        # the channels the ICA was trained on; other channels are untouched).
        raw_clean = raw.copy()
        ica.apply(raw_clean)

        summary = ICASummary(
            method="fastica",
            n_components=int(getattr(ica, "n_components_", len(ica.get_components()))),
            n_components_selected=None,
            excluded=excluded_components,
            eog_channels_used=eog_channels_used,
            eog_components=eog_components,
            muscle_components=muscle_components,
            notes=notes,
        )
        return raw_clean, summary

    def _build_fixed_window_epochs(
        self,
        raw: mne.io.BaseRaw,
        trials_df: pd.DataFrame,
        *,
        trial_type: str,
        edf_start_unix: float,
        timezone_offset: float,
    ) -> Optional[mne.Epochs]:
        """
        Create fixed-length, **EEG-only** epochs aligned to trial start times.

        Uses the shared :func:`unix_to_edf` from ``src.utils.time_utils`` for
        timestamp conversion (same formula as ENG-02).
        """
        if trials_df.empty:
            return None

        window_sec = _trial_type_window_sec(trial_type, fallback_duration=float(trials_df["duration"].median()))
        if window_sec is None:
            return None

        sfreq = float(raw.info["sfreq"])

        # Pick EEG-only channels for the epoch (no DC/AUX in output).
        picks_eeg = _pick_eeg_indices(raw)

        events_list: List[List[int]] = []
        metadata_rows: List[Dict[str, Any]] = []

        for idx, row in trials_df.reset_index(drop=True).iterrows():
            start_unix = row.get("start_time")
            if start_unix is None or (isinstance(start_unix, float) and np.isnan(start_unix)):
                continue

            # Reuse shared time conversion (same as ENG-02 TimestampAligner._unix_to_edf)
            start_edf = unix_to_edf(float(start_unix), edf_start_unix=edf_start_unix, timezone_offset=timezone_offset)
            if start_edf < 0:
                continue

            start_samp = int(round(start_edf * sfreq))
            end_edf = start_edf + float(window_sec)
            if end_edf > raw.times[-1]:
                continue

            events_list.append([start_samp, 0, 1])
            metadata_rows.append(
                {
                    "trial_idx": int(idx),
                    "start_time_unix": float(start_unix),
                    "end_time_unix": float(row.get("end_time")) if row.get("end_time") is not None else None,
                    "duration_log_sec": float(row.get("duration")) if row.get("duration") is not None else None,
                    "source_file": row.get("source_file"),
                }
            )

        if not events_list:
            return None

        events_arr = np.array(events_list, dtype=int)
        event_id = {str(trial_type).lower().strip(): 1}

        epochs = mne.Epochs(
            raw,
            events_arr,
            event_id=event_id,
            tmin=0.0,
            tmax=float(window_sec),
            picks=picks_eeg,
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
            verbose=self.verbose,
        )

        try:
            epochs.metadata = pd.DataFrame(metadata_rows)
        except Exception:
            pass

        return epochs

    def _auto_reject_epochs(self, epochs: mne.Epochs) -> Tuple[mne.Epochs, Optional[float], List[int]]:
        """
        Drop epochs whose max peak-to-peak amplitude (µV) exceeds the
        configured percentile threshold across all epochs.
        """
        if len(epochs) == 0:
            return epochs, None, []

        ptp_uv = _epoch_ptp_uv(epochs)
        if ptp_uv.size == 0:
            return epochs, None, []

        threshold_uv = float(np.percentile(ptp_uv, self.reject_ptp_percentile))
        drop_idx = np.where(ptp_uv > threshold_uv)[0].astype(int).tolist()

        if drop_idx:
            epochs.drop(drop_idx, reason=f"ENG03_PTP_GT_P{self.reject_ptp_percentile:g}")

        return epochs, threshold_uv, drop_idx

    # ── QC helpers ───────────────────────────────────────────────────────

    def _qc_row(
        self,
        *,
        patient_id: str,
        date: str,
        trial_type: str,
        n_total: int,
        n_dropped: int,
        threshold_uv: Optional[float],
        ica_summary: ICASummary,
        notes: List[str],
        ptp_stats: Optional[Dict[str, float]] = None,
        drop_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "patient_id": patient_id,
            "date": date,
            "trial_type": str(trial_type).lower().strip(),
            "window_sec": float(_trial_type_window_sec(trial_type, fallback_duration=None) or np.nan),
            "reject_ptp_percentile": float(self.reject_ptp_percentile),
            "reject_ptp_threshold_uv": float(threshold_uv) if threshold_uv is not None else None,
            "n_epochs_total": int(n_total),
            "n_epochs_dropped": int(n_dropped),
            "n_epochs_kept": int(max(0, n_total - n_dropped)),
            "drop_reason": drop_reason,
            "ica": json.dumps(asdict(ica_summary), sort_keys=True, default=_json_default),
            "notes": json.dumps(list(notes), sort_keys=True),
        }
        if ptp_stats:
            row.update(ptp_stats)
        return row

    @staticmethod
    def _ptp_stats(ptp_uv: np.ndarray) -> Dict[str, float]:
        if ptp_uv.size == 0:
            return {}
        return {
            "ptp_uv_p50": float(np.percentile(ptp_uv, 50)),
            "ptp_uv_p95": float(np.percentile(ptp_uv, 95)),
            "ptp_uv_p99": float(np.percentile(ptp_uv, 99)),
            "ptp_uv_max": float(np.max(ptp_uv)),
            "ptp_uv_mean": float(np.mean(ptp_uv)),
        }
