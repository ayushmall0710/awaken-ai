"""
Artifact Rejection (ICA)

This module applies session-level ICA-based artifact rejection and exports
fixed-window epochs per trial_type as MNE .fif files, plus QC metadata as Parquet.

Design constraints:
- Apply ICA once per EDF session (patient_id + date), then epoch all paradigms.
- Consume aligned events produced by ENG-02 (TimestampAligner) from:
  data/processed/aligned_events/{patient_id}_events.parquet
- Save:
  - epochs/{patient_id}/{date}/{trial_type}-epo.fif
  - qc/{patient_id}/{date}/eng03_qc.parquet

Classification strategy (Option B):
- Primary: ICLabel neural-network classifier (7 artifact types).
- Fallback: correlation-based (find_bads_eog / find_bads_ecg) when ICLabel
  cannot run (e.g. montage setup fails).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.utils.signal_processing import exclude_non_eeg_channels, normalize_channel_names
from src.utils.time_utils import detect_timezone_offset

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

DEFAULT_ICA_FILTER_HZ: Tuple[float, float] = (0.5, 100.0)
DEFAULT_REJECT_PTP_PERCENTILE: float = 95.0
DEFAULT_ICLABEL_THRESHOLD: float = 0.5


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ICASummary:
    method: str
    classification_method: str
    n_components: int
    n_components_selected: Optional[int]
    excluded: List[int]
    eog_channels_used: List[str]
    eog_components: List[int]
    ecg_channels_used: List[str]
    ecg_components: List[int]
    muscle_components: List[int]
    line_noise_components: List[int] = field(default_factory=list)
    channel_noise_components: List[int] = field(default_factory=list)
    iclabel_labels: Optional[List[str]] = None
    iclabel_probs: Optional[List[List[float]]] = None
    notes: List[str] = field(default_factory=list)


# ── Pure helpers (no class state) ────────────────────────────────────────────


def _note(notes: List[str], msg: str) -> None:
    """Append *msg* to the notes list **and** emit it via ``logger.debug``."""
    notes.append(msg)
    logger.debug(msg)


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
    2. Union of name-based ``EOG`` channels **and** infraorbital (IO1, IO2)
       — both can coexist in the same recording so we return all of them.
    3. Frontal channels Fp1/Fp2 as surrogate EOG (standard heuristic).
    """
    # 1. Prefer explicitly-typed EOG channels.
    try:
        picks = mne.pick_types(raw.info, eog=True, exclude=[])
        if len(picks) > 0:
            return [raw.ch_names[i] for i in picks]
    except Exception:
        pass

    # 2 & 3. Collect both named EOG channels and infraorbital (IO1/IO2).
    by_name = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    io_channels = [ch for ch in raw.ch_names if ch.upper() in {"IO1", "IO2"}]
    combined = by_name + [ch for ch in io_channels if ch not in by_name]
    if combined:
        return combined

    # 4. Heuristic: use Fp1/Fp2 as surrogate EOG (documented fallback).
    return [ch for ch in raw.ch_names if ch.upper() in {"FP1", "FP2"}]


def _epoch_ptp_uv(epochs: mne.Epochs) -> np.ndarray:
    """Per-epoch max peak-to-peak amplitude across channels (microvolts)."""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    if data.size == 0:
        return np.array([], dtype=float)
    ptp_v = np.ptp(data, axis=2)  # (n_epochs, n_channels)
    max_ptp_v = ptp_v.max(axis=1)  # (n_epochs,)
    return max_ptp_v * 1e6


def _pick_eeg_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    """Return channel indices for EEG-only picks, excluding non-EEG auxiliary channels.

    EDF files often label **all** channels as type ``eeg``, so we **always**
    apply the keyword-based exclusion from
    :data:`src.utils.signal_processing.NON_EEG_CHANNEL_KEYWORDS`.
    """
    non_eeg_names = set(exclude_non_eeg_channels(raw))

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

    return np.array(
        [i for i, ch in enumerate(raw.ch_names) if ch not in non_eeg_names],
        dtype=int,
    )


# ── ICA sub-functions ────────────────────────────────────────────────────────


def _prepare_ica_data(
    raw: mne.io.BaseRaw,
    ica_filter_hz: Tuple[float, float],
    verbose: bool,
) -> Tuple[mne.io.BaseRaw, np.ndarray, List[str]]:
    """Copy raw, pick EEG indices, and bandpass-filter for ICA stability.

    Returns ``(raw_for_ica, picks_eeg, notes)``.
    """
    notes: List[str] = []

    non_eeg_names = exclude_non_eeg_channels(raw)
    if non_eeg_names:
        _note(notes, f"[CHANNELS] non_eeg_excluded={non_eeg_names}")

    raw_for_ica = raw.copy()

    # Set non-EEG channels to "misc" so that CAR / montage only touch EEG.
    if non_eeg_names:
        type_map = {ch: "misc" for ch in non_eeg_names if ch in raw_for_ica.ch_names}
        if type_map:
            raw_for_ica.set_channel_types(type_map)

    picks_eeg = _pick_eeg_indices(raw_for_ica)
    _note(notes, f"[CHANNELS] n_eeg={len(picks_eeg)}")

    if len(picks_eeg) == 0:
        _note(notes, "[CHANNELS] no_eeg_after_exclusion\tfallback=all")
        picks_eeg = np.arange(len(raw_for_ica.ch_names))

    l_freq, h_freq = ica_filter_hz
    try:
        raw_for_ica.filter(l_freq=l_freq, h_freq=h_freq, picks=picks_eeg, verbose=verbose)
    except Exception as e:
        _note(notes, f"[FILTER] failed={type(e).__name__}: {e}")

    return raw_for_ica, picks_eeg, notes


def _fit_ica(raw_for_ica: mne.io.BaseRaw, picks_eeg: np.ndarray) -> mne.preprocessing.ICA:
    """Create and fit an ICA object using extended infomax."""
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        method="infomax",
        fit_params=dict(extended=True),
        max_iter="auto",
        random_state=97,
    )
    ica.fit(raw_for_ica, picks=picks_eeg)
    return ica


# ── Standard 10-20 electrode names (upper-cased) for montage mapping ────────
_STANDARD_1020_UPPER = {
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AFZ",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "FZ",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "T3",
    "T4",
    "T5",
    "T6",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "PZ",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "IZ",
    "I1",
    "I2",
    "A1",
    "A2",
}


def _try_set_montage(raw: mne.io.BaseRaw, notes: List[str]) -> bool:
    """Attempt to set a standard 10-20 montage on EEG channels.

    Strips common prefixes (``EEG ``, ``EEG-``) and checks if channel names
    match the standard 10-20 set.  Returns ``True`` when the montage is
    successfully applied, ``False`` otherwise.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_names_upper = {n.upper() for n in montage.ch_names}

    rename_map: Dict[str, str] = {}
    normalized_names = normalize_channel_names(raw.ch_names)

    for ch, clean in zip(raw.ch_names, normalized_names):
        if clean.upper() in montage_names_upper and clean != ch:
            rename_map[ch] = clean

    if rename_map:
        raw.rename_channels(rename_map)
        _note(notes, f"[MONTAGE] renamed_channels={rename_map}")

    eeg_picks = _pick_eeg_indices(raw)
    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    matched = [n for n in eeg_names if n.upper() in montage_names_upper]

    if len(matched) < 5:
        _note(notes, f"[MONTAGE] skipped\tonly {len(matched)}/{len(eeg_names)} channels match 10-20")
        return False

    # Set channel types so only matched channels get the montage.
    try:
        raw.set_channel_types({ch: "eeg" for ch in matched})
        raw.set_montage(montage, on_missing="warn")
        _note(notes, f"[MONTAGE] set\tmatched={len(matched)}/{len(eeg_names)}")
        return True
    except Exception as e:
        _note(notes, f"[MONTAGE] failed={type(e).__name__}: {e}")
        return False


def _apply_car_reference(raw: mne.io.BaseRaw, notes: List[str]) -> None:
    """Apply Common Average Reference (CAR) to EEG channels.

    ICLabel was trained on CAR-referenced data, so applying CAR before
    fitting ICA and running ICLabel improves classification accuracy.
    """
    try:
        raw.set_eeg_reference("average", projection=False, verbose=False)
        _note(notes, "[REFERENCE] CAR applied")
    except Exception as e:
        _note(notes, f"[REFERENCE] CAR failed={type(e).__name__}: {e}")


def _classify_components_iclabel(
    raw_for_ica: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    notes: List[str],
    threshold: float = DEFAULT_ICLABEL_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Classify ICA components using ICLabel.

    Returns a dict with keys ``excluded``, ``eog_components``, ``ecg_components``,
    ``muscle_components``, ``line_noise_components``, ``channel_noise_components``,
    ``iclabel_labels``, ``iclabel_probs``, or ``None`` if ICLabel is unavailable
    or classification fails.
    """
    try:
        result = label_components(raw_for_ica, ica, method="iclabel")
    except Exception as e:
        _note(notes, f"[ICLABEL] classification failed={type(e).__name__}: {e}\tfallback=correlation")
        return None

    labels: List[str] = list(result["labels"])

    # y_pred_proba is normally (n_components, n_classes).
    # Some versions / edge cases return a 1D array — reshape so the first
    # axis always equals len(labels).
    raw_probs = np.asarray(result["y_pred_proba"])
    if raw_probs.ndim == 1:
        raw_probs = raw_probs[:, np.newaxis]  # (N,) → (N, 1)
    probs: List[List[float]] = raw_probs.tolist()

    # ICLabel categories: brain, muscle, eye, heart, line_noise, channel_noise, other.
    # Collect artifact component indices grouped by label.
    _KEEP_LABELS = {"brain", "other"}
    by_label: Dict[str, List[int]] = defaultdict(list)
    excluded: List[int] = []

    for idx, label in enumerate(labels):
        if label in _KEEP_LABELS:
            continue
        max_prob = float(max(probs[idx]))
        if max_prob < threshold:
            continue
        excluded.append(idx)
        by_label[label].append(idx)

    _note(
        notes,
        f"[ICLABEL] excluded={len(excluded)} "
        f"(eye={len(by_label['eye'])}, heart={len(by_label['heart'])}, "
        f"muscle={len(by_label['muscle'])}, line_noise={len(by_label['line_noise'])}, "
        f"ch_noise={len(by_label['channel_noise'])})",
    )

    return {
        "excluded": sorted(excluded),
        "eog_components": sorted(by_label["eye"]),
        "ecg_components": sorted(by_label["heart"]),
        "muscle_components": sorted(by_label["muscle"]),
        "line_noise_components": sorted(by_label["line_noise"]),
        "channel_noise_components": sorted(by_label["channel_noise"]),
        "iclabel_labels": labels,
        "iclabel_probs": probs,
        "eog_channels_used": [],
        "ecg_channels_used": [],
    }


def _corr_find_eog(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    notes: List[str],
) -> Tuple[List[int], List[str]]:
    """Detect EOG-related ICA components via correlation."""
    eog_channels = _find_eog_channels(raw)
    components: List[int] = []
    channels_used: List[str] = []

    if eog_channels:
        for ch in eog_channels:
            try:
                inds, _scores = ica.find_bads_eog(raw, ch_name=ch, threshold=2.5)
                components.extend(list(inds))
                channels_used.append(ch)
            except Exception as e:
                _note(notes, f"[EOG] find_bads ch={ch}\tfailed={type(e).__name__}: {e}")
        components = sorted(set(components))
    else:
        _note(notes, "[EOG] no_eog_channels_detected")

    # MNE auto-detect fallback when no EOG components were found.
    if not components:
        try:
            inds, _scores = ica.find_bads_eog(raw, threshold=2.5)
            if inds:
                components = sorted(set(map(int, inds)))
                _note(notes, f"[EOG] fallback_mne_auto\tfound={len(inds)}")
        except Exception as e:
            _note(notes, f"[EOG] fallback_failed={type(e).__name__}: {e}")

    return components, channels_used


def _corr_find_muscle(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    notes: List[str],
) -> List[int]:
    """Detect muscle-related ICA components via correlation."""
    has_dig = bool(raw.info.get("dig"))
    try:
        if hasattr(ica, "find_bads_muscle") and has_dig:
            inds, _scores = ica.find_bads_muscle(raw)
            return sorted(set(map(int, inds)))
        if not has_dig:
            _note(notes, "[MUSCLE] skipped\treason=no_sensor_positions")
    except Exception as e:
        _note(notes, f"[MUSCLE] failed={type(e).__name__}: {e}")
    return []


def _corr_find_ecg(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    notes: List[str],
) -> Tuple[List[int], List[str]]:
    """Detect ECG-related ICA components via correlation."""
    ecg_candidates = [ch for ch in raw.ch_names if any(k in ch.upper() for k in ("ECG", "EKG"))]
    try:
        if ecg_candidates:
            inds, _scores = ica.find_bads_ecg(raw, ch_name=ecg_candidates[0])
            return sorted(set(map(int, inds))), [ecg_candidates[0]]
        inds, _scores = ica.find_bads_ecg(raw)
        components = sorted(set(map(int, inds)))
        channels_used = ["MNE_synthetic"] if components else []
        return components, channels_used
    except Exception as e:
        _note(notes, f"[ECG] failed={type(e).__name__}: {e}")
    return [], []


def _classify_components_correlation(
    raw_for_ica: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    notes: List[str],
) -> Dict[str, Any]:
    """Classify ICA components via correlation (EOG, ECG, muscle).

    This is the fallback path when ICLabel cannot run.
    """
    eog_components, eog_channels_used = _corr_find_eog(raw_for_ica, ica, notes)
    muscle_components = _corr_find_muscle(raw_for_ica, ica, notes)
    ecg_components, ecg_channels_used = _corr_find_ecg(raw_for_ica, ica, notes)

    excluded = sorted(set(eog_components + muscle_components + ecg_components))
    if not excluded:
        _note(notes, "[CORRELATION] WARNING\tno_components_excluded")

    return {
        "excluded": excluded,
        "eog_components": eog_components,
        "ecg_components": ecg_components,
        "muscle_components": muscle_components,
        "line_noise_components": [],
        "channel_noise_components": [],
        "iclabel_labels": None,
        "iclabel_probs": None,
        "eog_channels_used": eog_channels_used,
        "ecg_channels_used": ecg_channels_used,
    }


def _apply_and_summarize(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    classification: Dict[str, Any],
    classification_method: str,
    notes: List[str],
) -> Tuple[mne.io.BaseRaw, ICASummary]:
    """Apply ICA exclusions and build the summary dataclass."""
    excluded = classification["excluded"]
    ica.exclude = excluded

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    summary = ICASummary(
        method="infomax",
        classification_method=classification_method,
        n_components=int(getattr(ica, "n_components_", len(ica.get_components()))),
        n_components_selected=None,
        excluded=excluded,
        eog_channels_used=classification["eog_channels_used"],
        eog_components=classification["eog_components"],
        ecg_channels_used=classification["ecg_channels_used"],
        ecg_components=classification["ecg_components"],
        muscle_components=classification["muscle_components"],
        line_noise_components=classification.get("line_noise_components", []),
        channel_noise_components=classification.get("channel_noise_components", []),
        iclabel_labels=classification.get("iclabel_labels"),
        iclabel_probs=classification.get("iclabel_probs"),
        notes=notes,
    )
    return raw_clean, summary


# ── Main class ───────────────────────────────────────────────────────────────


class ArtifactRejector:
    """Preprocessing driver: run ICA artifact rejection per session and export
    fixed-window EEG-only epochs + QC metadata.
    """

    def __init__(
        self,
        data_root: Optional[Path] = None,
        loader: Optional[UnifiedDataLoader] = None,
        use_clipped: bool = True,
        ica_filter_hz: Tuple[float, float] = DEFAULT_ICA_FILTER_HZ,
        reject_ptp_percentile: float = DEFAULT_REJECT_PTP_PERCENTILE,
        iclabel_threshold: float = DEFAULT_ICLABEL_THRESHOLD,
        verbose: bool = False,
    ):
        self.loader = loader or UnifiedDataLoader(data_root=data_root, verbose=verbose)
        self.use_clipped = use_clipped
        self.ica_filter_hz = ica_filter_hz
        self.reject_ptp_percentile = float(reject_ptp_percentile)
        self.iclabel_threshold = float(iclabel_threshold)
        self.verbose = verbose

        if not verbose:
            mne.set_log_level("WARNING")

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, patient_ids: List[str], save: bool = True) -> Dict[Tuple[str, str], Dict[str, Path]]:
        """Run artifact rejection for a list of patients.

        Returns:
            Mapping of ``(patient_id, session_id) -> {trial_type: epochs_fif_path}``.
        """
        out: Dict[Tuple[str, str], Dict[str, Path]] = {}
        for patient_id in patient_ids:
            patient = self.loader.get_patient(patient_id)
            for session_id in patient.list_session_ids():
                try:
                    out[(patient_id, session_id)] = self.run_session(patient_id, session_id, save=save)
                except Exception as e:
                    logger.error(f"Artifact rejection failed for {patient_id}/{session_id}: {e}")
                    out[(patient_id, session_id)] = {}
        return out

    def run_session(
        self,
        patient_id: str,
        session_id: str,
        save: bool = True,
        return_raw_clean: bool = False,
    ) -> Union[Dict[str, Path], Tuple[Dict[str, Path], Optional[mne.io.BaseRaw]]]:
        """Run artifact rejection for a single patient session.

        Steps: load -> ICA -> epoch per trial_type -> auto-reject -> save.

        Args:
            patient_id: Patient identifier.
            session_id: The unique session ID (e.g. s_CON008_202501100000).
            save: Whether to save epochs and QC parquet to disk.
            return_raw_clean: If True, return (saved_paths, raw_clean) where raw_clean
                is the ICA-cleaned raw EEG before epoch creation. If False, return only
                saved_paths.

        Returns:
            saved_paths: Mapping of trial_type -> epochs .fif path.
            raw_clean (optional): ICA-cleaned raw when return_raw_clean=True.
        """
        inputs = self._load_session_inputs(patient_id, session_id)
        if inputs is None:
            if return_raw_clean:
                return {}, None
            return {}

        raw, session_df, date, edf_start_unix, timezone_offset = inputs
        raw_clean, ica_summary = self._apply_ica(raw)

        saved_paths: Dict[str, Path] = {}
        qc_rows: List[Dict[str, Any]] = []

        for trial_type, tt_df in session_df.groupby(session_df["trial_type"].astype(str).str.lower()):
            epochs, qc_row = self._process_trial_type(
                raw_clean,
                tt_df,
                trial_type=trial_type,
                patient_id=patient_id,
                date=date,
                session_id=session_id,
                edf_start_unix=edf_start_unix,
                timezone_offset=timezone_offset,
                ica_summary=ica_summary,
            )
            qc_rows.append(qc_row)

            if save and epochs is not None and len(epochs) > 0:
                fif_path = self._epochs_output_path(patient_id, session_id, trial_type)
                fif_path.parent.mkdir(parents=True, exist_ok=True)
                epochs.save(fif_path, overwrite=True)
                saved_paths[trial_type] = fif_path

        if save:
            self._save_qc(qc_rows, patient_id, session_id)

        if return_raw_clean:
            return saved_paths, raw_clean
        return saved_paths

    # ── I/O helpers ──────────────────────────────────────────────────────

    def _load_session_inputs(
        self, patient_id: str, session_id: str
    ) -> Optional[Tuple[mne.io.BaseRaw, pd.DataFrame, str, float, float]]:
        """Load EDF, aligned events, and compute time-conversion constants. Returns the date as well for QC rows."""
        aligned_df = self.loader.load_aligned_events(patient_id)
        session_df = aligned_df[aligned_df["session_id"] == session_id].copy()

        if session_df.empty:
            logger.error(
                f"Artifact rejection failed for {patient_id}/{session_id}: "
                f"Aligned events parquet not found: {config.ALIGNED_EVENTS_DIR / f'{patient_id}_events.parquet'}"
            )
            return None

        date = session_df["date"].iloc[0]
        raw = self.loader.load_edf(patient_id, date=date, use_clipped=self.use_clipped)

        timezone_offset = detect_timezone_offset(raw, session_df)
        edf_start_unix = raw.info["meas_date"].timestamp() if raw.info.get("meas_date") is not None else 0.0
        return raw, session_df, date, edf_start_unix, timezone_offset

    @staticmethod
    def _epochs_output_path(patient_id: str, session_id: str, trial_type: str) -> Path:
        safe_tt = str(trial_type).lower().strip()
        return config.EPOCHS_DIR / patient_id / session_id / f"{safe_tt}-epo.fif"

    @staticmethod
    def _save_qc(qc_rows: List[Dict[str, Any]], patient_id: str, session_id: str) -> None:
        qc_path = config.QC_DIR / patient_id / session_id / "eng03_qc.parquet"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(qc_rows).to_parquet(qc_path, index=False)

    # ── Core processing ──────────────────────────────────────────────────

    def _apply_ica(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, ICASummary]:
        """Fit ICA, classify components (ICLabel primary, correlation fallback),
        and subtract artifact components from the raw signal.

        The recommended ICLabel workflow is:
        filter → montage → CAR → fit ICA → classify → apply.
        """
        raw_for_ica, picks_eeg, notes = _prepare_ica_data(raw, self.ica_filter_hz, self.verbose)

        # Set montage + CAR *before* fitting ICA (matches ICLabel training pipeline).
        montage_ok = _try_set_montage(raw_for_ica, notes)
        if montage_ok:
            _apply_car_reference(raw_for_ica, notes)

        ica = _fit_ica(raw_for_ica, picks_eeg)

        # Primary path: ICLabel (requires montage for topographic features).
        classification: Optional[Dict[str, Any]] = None
        method = "correlation"

        if montage_ok:
            classification = _classify_components_iclabel(
                raw_for_ica,
                ica,
                notes,
                threshold=self.iclabel_threshold,
            )
            if classification is not None:
                method = "iclabel"

        # Fallback: correlation-based detection.
        if classification is None:
            classification = _classify_components_correlation(raw_for_ica, ica, notes)

        return _apply_and_summarize(raw, ica, classification, method, notes)

    def _process_trial_type(
        self,
        raw_clean: mne.io.BaseRaw,
        tt_df: pd.DataFrame,
        *,
        trial_type: str,
        patient_id: str,
        date: str,
        session_id: str,
        edf_start_unix: float,
        timezone_offset: float,
        ica_summary: ICASummary,
    ) -> Tuple[Optional[mne.Epochs], Dict[str, Any]]:
        """Build epochs for one trial type, auto-reject, and return a QC row."""
        epochs = self._build_fixed_window_epochs(
            raw_clean,
            tt_df,
            trial_type=trial_type,
            edf_start_unix=edf_start_unix,
            timezone_offset=timezone_offset,
        )

        if epochs is None or len(epochs) == 0:
            return None, self._qc_row(
                patient_id=patient_id,
                date=date,
                session_id=session_id,
                trial_type=trial_type,
                n_total=0,
                n_dropped=0,
                threshold_uv=None,
                ica_summary=ica_summary,
                notes=["no_epochs"],
            )

        n_total_before = len(epochs)
        ptp_stats = self._ptp_stats(_epoch_ptp_uv(epochs))
        epochs, threshold_uv, dropped = self._auto_reject_epochs(epochs)

        qc = self._qc_row(
            patient_id=patient_id,
            date=date,
            session_id=session_id,
            trial_type=trial_type,
            n_total=n_total_before,
            n_dropped=len(dropped),
            threshold_uv=threshold_uv,
            ica_summary=ica_summary,
            notes=[],
            ptp_stats=ptp_stats,
            drop_reason=(
                f"ENG03_PTP_GT_P{self.reject_ptp_percentile:g}" if (dropped and threshold_uv is not None) else None
            ),
        )
        return epochs, qc

    def _build_fixed_window_epochs(
        self,
        raw: mne.io.BaseRaw,
        trials_df: pd.DataFrame,
        *,
        trial_type: str,
        edf_start_unix: float,
        timezone_offset: float,
    ) -> Optional[mne.Epochs]:
        """Create fixed-length, EEG-only epochs using vectorized time conversion."""
        if trials_df.empty:
            return None

        window_sec = _trial_type_window_sec(trial_type, fallback_duration=float(trials_df["duration"].median()))
        if window_sec is None:
            return None

        sfreq = float(raw.info["sfreq"])
        max_time = float(raw.times[-1])
        picks_eeg = _pick_eeg_indices(raw)

        # Vectorized time conversion (replaces iterrows).
        start_unix = trials_df["start_time"].values.astype(float)
        valid_mask = ~np.isnan(start_unix)
        start_edf = start_unix[valid_mask] - edf_start_unix + timezone_offset
        in_range = (start_edf >= 0) & (start_edf + window_sec <= max_time)
        start_edf = start_edf[in_range]

        if len(start_edf) == 0:
            return None

        start_samps = np.round(start_edf * sfreq).astype(int)
        events_arr = np.column_stack(
            [
                start_samps,
                np.zeros(len(start_samps), dtype=int),
                np.ones(len(start_samps), dtype=int),
            ]
        )

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
            event_repeated="drop",
            verbose=self.verbose,
        )

        # Build metadata for the valid rows.
        valid_df = trials_df.iloc[np.where(valid_mask)[0][in_range]].reset_index(drop=True)
        try:
            epochs.metadata = pd.DataFrame(
                {
                    "start_time_unix": valid_df["start_time"].values.astype(float),
                    "end_time_unix": valid_df["end_time"].values.astype(float) if "end_time" in valid_df else np.nan,
                    "duration_log_sec": valid_df["duration"].values.astype(float) if "duration" in valid_df else np.nan,
                    "source_file": valid_df["source_file"].values if "source_file" in valid_df else None,
                    "session_id": valid_df["session_id"].values,
                    "trial_id": valid_df["trial_id"].values,
                }
            )
        except Exception:
            pass

        return epochs

    def _auto_reject_epochs(self, epochs: mne.Epochs) -> Tuple[mne.Epochs, Optional[float], List[int]]:
        """Drop epochs whose max PTP (uV) exceeds the configured percentile."""
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
        session_id: str,
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
            "session_id": session_id,
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
