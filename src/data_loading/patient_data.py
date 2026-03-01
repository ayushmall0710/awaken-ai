"""PatientData — single-patient data view wrapping UnifiedDataLoader and patient_records.json."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

from src.data_loading import config

if TYPE_CHECKING:
    import mne

    from src.data_loading.unified_data_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


class PatientData:
    """Single-patient data view with trial filtering and lazy EDF loading.

    Wraps UnifiedDataLoader for EDF access and enriches with clinical metadata
    from patient_records.json. Intended as the primary interface for all
    single-patient workflows (CLI, pipelines, notebooks).

    Args:
        patient_id: Patient identifier (e.g. 'CON008')
        loader: UnifiedDataLoader instance for EDF and trial access.
        session: Optional session date filter (e.g. '2025-01-10')
        trial_type: Optional trial type filter (e.g. 'left_command', 'oddball')
    """

    def __init__(
        self,
        patient_id: str,
        loader: "UnifiedDataLoader",
        session: Optional[str] = None,
        trial_type: Optional[str] = None,
    ):
        self.patient_id = patient_id
        self._loader = loader
        self._session = session
        self._trial_type = trial_type

        trials = loader.trials_df[loader.trials_df["patient_id"] == patient_id]
        if session:
            trials = trials[trials["date"] == session]
        if trial_type:
            trials = trials[trials["trial_type"] == trial_type]
        self.trials_df = trials

        records: Dict[str, Any] = {}
        if config.PATIENT_RECORDS_PATH.exists():
            with open(config.PATIENT_RECORDS_PATH) as f:
                records = json.load(f)
        self._record = records.get(patient_id, {})

    # ─── Trial access ─────────────────────────────────────────────────────────

    def list_sessions(self) -> List[str]:
        """Sorted list of recording session dates."""
        return sorted(self.trials_df["date"].unique().tolist())

    def list_session_ids(self) -> List[str]:
        """Sorted list of unique session IDs."""
        if "session_id" not in self.trials_df.columns:
            return []
        return sorted(self.trials_df["session_id"].dropna().unique().tolist())

    def get_trial_types(self) -> List[str]:
        """Sorted list of unique trial types for this patient."""
        return sorted(self.trials_df["trial_type"].unique().tolist())

    def get_trials_by_type(self, trial_type: Optional[str] = None) -> pd.DataFrame:
        """Trials of a specific type. Uses init-time trial_type if not specified.

        Returns empty DataFrame (with warning) if no trials found.
        """
        filter_type = trial_type or self._trial_type
        if not filter_type:
            return self.trials_df.copy()
        filtered = self.trials_df[self.trials_df["trial_type"] == filter_type]
        if filtered.empty:
            logger.warning(
                "No trials found for type '%s' in patient %s. Available: %s",
                filter_type,
                self.patient_id,
                self.get_trial_types(),
            )
        return filtered.copy()

    def get_trial_by_id(self, trial_id: str) -> pd.Series:
        """Single trial by trial_id string (e.g. 'lt1', 'obt2')."""
        matches = self.trials_df[self.trials_df["trial_id"] == trial_id]
        if matches.empty:
            raise KeyError(f"No trial with trial_id='{trial_id}' for patient {self.patient_id}")
        return matches.iloc[0]

    # ─── EDF access ───────────────────────────────────────────────────────────

    def get_raw_edf(self, date: Optional[str] = None) -> Union["mne.io.Raw", Dict[str, "mne.io.Raw"]]:
        """Load EEG data. Returns Raw for single/specified session, Dict[date, Raw] for multi-session."""
        return self._loader.load_edf(self.patient_id, date=date or self._session)

    @property
    def raw_edf(self) -> Union["mne.io.Raw", Dict[str, "mne.io.Raw"]]:
        """Lazy-load EEG data. Returns Raw for single session, Dict[date, Raw] for multi."""
        return self.get_raw_edf()

    @property
    def edf_paths(self) -> Union["Path", Dict[str, "Path"]]:  # noqa: F821
        """EDF path(s): Path for single session, Dict[date, Path] for multi-session."""
        sessions = self.list_sessions()
        if len(sessions) == 1:
            return self._loader._find_edf(self.patient_id, sessions[0], use_clipped=True)
        return {s: self._loader._find_edf(self.patient_id, s, use_clipped=True) for s in sessions}

    # ─── Clinical metadata (from patient_records.json) ────────────────────────

    @property
    def first_visit(self) -> Optional[str]:
        """Date of first recorded visit, from patient_records.json."""
        return self._record.get("first_visit")

    @property
    def last_visit(self) -> Optional[str]:
        """Date of most recent visit, from patient_records.json."""
        return self._record.get("last_visit")

    @property
    def notes(self) -> List[Dict[str, str]]:
        """Clinical notes [{date, notes}], from patient_records.json."""
        return self._record.get("notes", [])

    @property
    def visit_history(self) -> List[Dict[str, str]]:
        """Visit history [{date}], from patient_records.json."""
        return self._record.get("visit_history", [])

    # ─── Summary ──────────────────────────────────────────────────────────────

    def info(self) -> Dict[str, Any]:
        """Structured summary for display (CLI, notebooks). No EDF loading."""
        trial_counts = {t: len(self.get_trials_by_type(t)) for t in self.get_trial_types()}
        return {
            "patient_id": self.patient_id,
            "sessions": self.list_sessions(),
            "total_trials": len(self.trials_df),
            "trial_counts": trial_counts,
            "first_visit": self.first_visit,
            "last_visit": self.last_visit,
            "notes": self.notes,
            "visit_history": self.visit_history,
        }

    # ─── EEG metadata (triggers EDF load) ─────────────────────────────────────

    def get_eeg_info(self) -> Dict[str, Any]:
        """EEG hardware metadata — triggers EDF loading. For multi-session uses first session."""
        raw_data = self.raw_edf
        raw = list(raw_data.values())[0] if isinstance(raw_data, dict) else raw_data
        return {
            "patient_id": self.patient_id,
            "n_channels": len(raw.ch_names),
            "channel_names": raw.ch_names,
            "sampling_rate": raw.info["sfreq"],
            "duration_seconds": raw.times[-1],
            "duration_minutes": raw.times[-1] / 60,
            "measurement_date": raw.info["meas_date"],
        }

    def __repr__(self) -> str:
        n_sessions = len(self.list_sessions())
        filters = []
        if self._session:
            filters.append(f"session={self._session}")
        if self._trial_type:
            filters.append(f"trial_type={self._trial_type}")
        filter_str = f", filters=[{', '.join(filters)}]" if filters else ""
        return (
            f"PatientData(patient={self.patient_id}, {len(self.trials_df)} trials, {n_sessions} session(s){filter_str})"
        )
