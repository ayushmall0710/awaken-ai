"""P300/Oddball analysis pipeline (ENG-02b) built on BasePipeline.

This pipeline:
- loads ENG-03 ICA-cleaned 35s oddball epochs
- maps rare/standard stimulus events into those trial windows
- extracts 900ms sub-epochs time-locked to rare stimuli
- computes ERPs and P300 features (including composite scores and QC)

It refactors the logic from ``src.data_processing.erp_pipeline.OddballERPPipeline``
into a BasePipeline-compatible class for CLI orchestration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.data_loading import UnifiedDataLoader, config
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


ERP_CONFIG = {
    "tmin": -0.2,  # Epoch start: 200ms before stimulus
    "tmax": 0.7,  # Epoch end: 700ms after stimulus
    "baseline": (None, 0),  # Baseline correction: -200ms to 0ms
    "p300_window": (0.3, 0.6),  # P300 search window: 300-600ms
    "min_epochs": 2,  # Minimum rare events needed per trial/session
    "midline_electrodes": ["Pz", "Cz", "Fz"],
    "p300_min_amplitude": 0.0,  # µV — must be positive
    "p300_expected_latency_range": (300, 500),  # typical range for controls (ms)
    "p300_max_latency_range": (250, 600),  # hard rejection cutoff (ms)
}

# Event labels used for standard (frequent) stimuli in the oddball paradigm.
STANDARD_EVENT_LABELS = {"standard", "frequent"}


@dataclass
class SessionData:
    """Container for per-session oddball data and results."""

    session_id: str
    date: str
    epochs35: mne.Epochs
    status: str = "pending"
    rare_events: Optional[List[Dict[str, Any]]] = None
    standard_events: Optional[List[Dict[str, Any]]] = None
    rare_mapped_df: Optional[pd.DataFrame] = None
    epochs: Optional[mne.Epochs] = None
    rare_erp: Optional[mne.Evoked] = None
    rare_sem: Optional[mne.Evoked] = None
    standard_erp: Optional[mne.Evoked] = None
    standard_sem: Optional[mne.Evoked] = None
    diff_erp: Optional[mne.Evoked] = None
    n_standard_epochs: int = 0
    n_standard_events_candidate: int = 0
    mapping_diag: Optional[Dict[str, Any]] = None


class P300OddballPipeline(BasePipeline):
    """P300/Oddball ERP pipeline implemented as a BasePipeline."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        loader: Optional[UnifiedDataLoader] = None,
        verbose: bool = False,
    ):
        super().__init__(loader=loader or UnifiedDataLoader(data_root=data_root or config.LOCAL_DATA_ROOT, verbose=verbose))
        self.data_root = data_root or config.LOCAL_DATA_ROOT
        self.output_dir = output_dir or config.PROCESSED_DATA_DIR
        self.verbose = verbose

        if not verbose:
            mne.set_log_level("WARNING")

        self._output_paths = self._get_output_paths()
        self._create_output_directories()

        self._session_data: Dict[str, SessionData] = {}
        self._oddball_trials: Optional[pd.DataFrame] = None
        self._last_epoch_diagnostics: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Output directory helpers (adapted from OddballERPPipeline)
    # ------------------------------------------------------------------

    def _get_output_paths(self):
        """Use config constants when output_dir is the default; otherwise nest under output_dir."""
        if self.output_dir == config.PROCESSED_DATA_DIR:
            return type(
                "Paths",
                (),
                {
                    "erps": config.ERPS_DIR,
                    "features": config.FEATURES_DIR,
                    "plots_erp": config.ERP_PLOTS_DIR,
                    "qc": config.QC_REPORTS_DIR,
                },
            )()
        return type(
            "Paths",
            (),
            {
                "erps": self.output_dir / "erps",
                "features": self.output_dir / "features",
                "plots_erp": self.output_dir / "plots" / "erp",
                "qc": self.output_dir / "qc",
            },
        )()

    def _create_output_directories(self) -> None:
        p = self._output_paths
        for dir_path in [p.erps, p.features, p.plots_erp, p.qc]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BasePipeline interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ENG-03 oddball epochs and aligned oddball trials for this patient."""
        if self.aligned_events is None or self.aligned_events.empty:
            raise ValueError(f"No aligned events found for patient {self.patient_id}")

        oddball_trials = self.aligned_events[
            self.aligned_events["trial_type"].str.lower() == "oddball"
        ].copy()

        if oddball_trials.empty:
            raise ValueError(f"No oddball trials found for patient {self.patient_id}")

        self._oddball_trials = oddball_trials

        # Pre-load ENG-03 epochs per session/date
        for date in oddball_trials["date"].unique():
            trials_for_date = oddball_trials[oddball_trials["date"] == date]
            session_id = trials_for_date["session_id"].iloc[0]
            try:
                epochs35 = self.loader.load_clean_epochs(
                    patient_id=self.patient_id,
                    session_id=session_id,
                    trial_type="oddball",
                )
            except FileNotFoundError as e:
                logger.error(
                    f"ENG-03 oddball epochs not found for {self.patient_id} - {date}. "
                    "Run ENG-03 (ArtifactRejector.run_session) first.",
                )
                continue
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Failed to load ENG-03 epochs for {self.patient_id} - {date}: {e}", exc_info=True)
                continue

            self._session_data[date] = SessionData(
                session_id=session_id,
                date=str(date),
                epochs35=epochs35,
            )

        if not self._session_data:
            raise ValueError(f"Could not load ENG-03 oddball epochs for any session of {self.patient_id}")

    def preprocess(self) -> None:
        """Extract sub-epochs and compute per-session ERPs."""
        if self._oddball_trials is None:
            raise RuntimeError("load() must be called before preprocess().")

        for date, sess in list(self._session_data.items()):
            try:
                aligned = self._oddball_trials[self._oddball_trials["date"] == date]
                epochs35 = sess.epochs35

                rare_events = self._extract_rare_events(aligned)
                standard_events = self._extract_standard_events(aligned)

                if len(rare_events) < ERP_CONFIG["min_epochs"]:
                    logger.warning(
                        f"Insufficient rare events for {self.patient_id} on {date}: "
                        f"{len(rare_events)} < {ERP_CONFIG['min_epochs']}",
                    )
                    sess.status = "insufficient_rare_events"
                    continue

                trial_windows = self._build_trial_windows(epochs35)
                rare_mapped_df, mapping_diag = self._map_events_to_trials(
                    rare_events,
                    trial_windows,
                    sfreq=float(epochs35.info["sfreq"]),
                )
                self._last_epoch_diagnostics = mapping_diag

                epochs = self._extract_subepochs(epochs35, rare_mapped_df)

                if len(epochs) < ERP_CONFIG["min_epochs"]:
                    logger.warning(
                        f"Insufficient rare epochs after mapping for {self.patient_id} on {date}: {len(epochs)}",
                    )
                    sess.status = "insufficient_epochs"
                    continue

                rare_erp, rare_sem = self._compute_erp(epochs)

                standard_erp = None
                standard_sem = None
                diff_erp = None
                n_standard_epochs = 0
                n_standard_events_candidate = len(standard_events)

                if len(standard_events) > 0:
                    std_mapped_df, _ = self._map_events_to_trials(
                        standard_events,
                        trial_windows,
                        sfreq=float(epochs35.info["sfreq"]),
                    )
                    standard_epochs = self._extract_subepochs(epochs35, std_mapped_df)
                    n_standard_epochs = len(standard_epochs)

                    if n_standard_epochs >= ERP_CONFIG["min_epochs"]:
                        standard_erp, standard_sem = self._compute_erp(standard_epochs)
                        diff_erp = self._compute_difference_erp(rare_erp, standard_erp)

                sess.status = "success"
                sess.rare_events = rare_events
                sess.standard_events = standard_events
                sess.rare_mapped_df = rare_mapped_df
                sess.epochs = epochs
                sess.rare_erp = rare_erp
                sess.rare_sem = rare_sem
                sess.standard_erp = standard_erp
                sess.standard_sem = standard_sem
                sess.diff_erp = diff_erp
                sess.n_standard_epochs = n_standard_epochs
                sess.n_standard_events_candidate = n_standard_events_candidate
                sess.mapping_diag = mapping_diag
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Preprocessing failed for {self.patient_id} - {date}: {e}", exc_info=True)
                sess.status = "preprocessing_error"

    def analyze(
        self,
        custom_electrodes: Optional[List[str]] = None,
        **_: Any,
    ) -> pd.DataFrame:
        """Quantify P300 features for each successful session."""
        rows: List[Dict[str, Any]] = []

        for date in sorted(self._session_data.keys()):
            sess = self._session_data[date]
            if sess.status != "success" or sess.epochs is None or sess.rare_erp is None:
                logger.warning(f"Skipping session {self.patient_id} {date}: status={sess.status}")
                continue

            try:
                # Restore mapping diagnostics from session for _quantify_p300 to use
                if sess.mapping_diag:
                    self._last_epoch_diagnostics = sess.mapping_diag

                features = self._quantify_p300(
                    erp=sess.rare_erp,
                    patient_id=self.patient_id or "",
                    date=date,
                    n_epochs=len(sess.epochs),
                    custom_electrodes=custom_electrodes,
                    diff_erp=sess.diff_erp,
                    n_standard_epochs=sess.n_standard_epochs,
                    n_standard_events_candidate=sess.n_standard_events_candidate,
                )

                rows.append(features)

                # Persist ERPs, features, and plots
                self._save_outputs(
                    patient_id=self.patient_id or "",
                    date=date,
                    epochs=sess.epochs,
                    erp=sess.rare_erp,
                    features=features,
                    standard_erp=sess.standard_erp,
                    diff_erp=sess.diff_erp,
                )

                self._plot_erp_figure(
                    rare_erp=sess.rare_erp,
                    rare_sem=sess.rare_sem,
                    standard_erp=sess.standard_erp,
                    standard_sem=sess.standard_sem,
                    diff_erp=sess.diff_erp,
                    features=features,
                    patient_id=self.patient_id or "",
                    date=date,
                    custom_electrodes=custom_electrodes,
                )

                self._plot_erp_image(
                    epochs=sess.epochs,
                    patient_id=self.patient_id or "",
                    date=date,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Analysis failed for {self.patient_id} - {date}: {e}", exc_info=True)
                continue

        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self.results = df
        return df

    def generate_summary(self) -> Dict[str, Any]:
        """Summarize P300 detection status across sessions."""
        if self.results is None or self.results.empty:
            return {
                "patient_id": self.patient_id,
                "status": "NO_DATA",
                "n_sessions": 0,
            }

        has_p300 = (self.results["p300_amplitude_uV"] > 2.0).any()

        return {
            "patient_id": self.patient_id,
            "status": "P300+" if has_p300 else "P300-",
            "n_sessions": int(len(self.results)),
            "mean_amplitude_uV": float(self.results["p300_amplitude_uV"].mean()),
            "mean_latency_ms": float(self.results["p300_latency_ms"].mean()),
        }

    # ------------------------------------------------------------------
    # Event extraction utilities (adapted from OddballERPPipeline)
    # ------------------------------------------------------------------

    def _extract_rare_events(self, trials_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract rare beep events with timestamps from oddball trials."""
        rare_events: List[Dict[str, Any]] = []

        for idx, trial in trials_df.iterrows():
            try:
                events = trial["sentences"]
            except (KeyError, TypeError):
                logger.warning(f"Trial {idx}: No 'sentences' field found")
                continue

            if isinstance(events, np.ndarray):
                events = events.tolist()

            if not isinstance(events, list):
                logger.warning(f"Trial {idx}: sentences is not a list (type: {type(events)})")
                continue

            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "rare" and "event_start" in event:
                    rare_events.append(
                        {
                            "timestamp_unix": event["event_start"],
                            "date": trial["date"],
                            "trial_idx": idx,
                            "correlation_score": event.get("correlation_score"),
                            "peak_amplitude": event.get("peak_amplitude"),
                        },
                    )

        logger.info(f"Extracted {len(rare_events)} rare events")
        return rare_events

    def _extract_standard_events(self, trials_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract standard (frequent) beep events with timestamps from oddball trials."""
        standard_events: List[Dict[str, Any]] = []

        for idx, trial in trials_df.iterrows():
            try:
                events = trial["sentences"]
            except (KeyError, TypeError):
                logger.warning(f"Trial {idx}: No 'sentences' field found")
                continue

            if isinstance(events, np.ndarray):
                events = events.tolist()

            if not isinstance(events, list):
                logger.warning(f"Trial {idx}: sentences is not a list (type: {type(events)})")
                continue

            for event in events:
                if not isinstance(event, dict):
                    continue
                label = event.get("event")
                if label in STANDARD_EVENT_LABELS and "event_start" in event:
                    standard_events.append(
                        {
                            "timestamp_unix": event["event_start"],
                            "date": trial["date"],
                            "trial_idx": idx,
                        },
                    )

        logger.info(f"Extracted {len(standard_events)} standard events")
        return standard_events

    # ------------------------------------------------------------------
    # Trial window mapping & epoch extraction
    # ------------------------------------------------------------------

    def _build_trial_windows(self, epochs35: mne.Epochs, window_sec: float = 35.0) -> pd.DataFrame:
        """Build a table of ENG-03 trial start/end times from epoch metadata."""
        if epochs35.metadata is None:
            raise ValueError("ENG-03 oddball epochs have no metadata; expected start_time_unix field.")

        md = epochs35.metadata
        if "start_time_unix" not in md.columns:
            raise ValueError("ENG-03 metadata missing start_time_unix; cannot map rare events.")

        starts = md["start_time_unix"].astype(float).values
        ends = md["end_time_unix"].astype(float).values if "end_time_unix" in md.columns else starts + float(window_sec)

        return pd.DataFrame(
            {
                "eng03_epoch_idx": np.arange(len(starts), dtype=int),
                "start_time_unix": starts,
                "end_time_unix": ends,
                "window_sec": float(window_sec),
            },
        )

    def _map_events_to_trials(
        self,
        events: List[Dict[str, Any]],
        trial_windows_df: pd.DataFrame,
        sfreq: float,
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Map event timestamps into ENG-03 35s trial windows."""
        tmin = ERP_CONFIG["tmin"]
        tmax = ERP_CONFIG["tmax"]
        starts = trial_windows_df["start_time_unix"].values
        window_sec = float(trial_windows_df["window_sec"].iloc[0])
        epoch_ids = trial_windows_df["eng03_epoch_idx"].values

        rows: List[Dict[str, Any]] = []
        n_unmapped = 0
        n_duplicate = 0
        n_boundary = 0

        for event in events:
            ts = float(event["timestamp_unix"])
            mask = (starts <= ts) & (ts < starts + window_sec)
            matched = epoch_ids[mask]

            if len(matched) != 1:
                if len(matched) > 1:
                    n_duplicate += 1
                else:
                    n_unmapped += 1
                continue

            epoch_idx = int(matched[0])
            offset_sec = ts - float(starts[epoch_idx])
            start_sample = int(np.round((offset_sec + tmin) * sfreq))
            n_target = int(np.round((tmax - tmin) * sfreq)) + 1
            end_sample = start_sample + n_target

            if start_sample < 0 or end_sample > int(np.round(window_sec * sfreq)) + 1:
                n_boundary += 1
                continue

            rows.append(
                {
                    "timestamp_unix": ts,
                    "eng03_epoch_idx": epoch_idx,
                    "offset_sec": offset_sec,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                },
            )

        mapped_df = pd.DataFrame(rows)
        diagnostics = {
            "n_rare_events": len(events),
            "n_mapped": len(mapped_df),
            "n_unmapped": n_unmapped,
            "n_duplicate": n_duplicate,
            "n_boundary_clipped": n_boundary,
            "mapping_rate": len(mapped_df) / max(len(events), 1),
        }
        logger.info(
            "Event mapping: %d/%d mapped (unmapped=%d, duplicate=%d, boundary=%d)",
            diagnostics["n_mapped"],
            diagnostics["n_rare_events"],
            n_unmapped,
            n_duplicate,
            n_boundary,
        )
        return mapped_df, diagnostics

    def _extract_subepochs(
        self,
        epochs35: mne.Epochs,
        mapped_df: pd.DataFrame,
    ) -> mne.Epochs:
        """Slice 900ms sub-epochs from 35s ENG-03 epochs."""
        tmin = ERP_CONFIG["tmin"]
        tmax = ERP_CONFIG["tmax"]
        sfreq = float(epochs35.info["sfreq"])
        n_target = int(np.round((tmax - tmin) * sfreq)) + 1

        src = epochs35.get_data()
        slices: List[np.ndarray] = []

        for _, row in mapped_df.iterrows():
            eidx = int(row["eng03_epoch_idx"])
            s0 = int(row["start_sample"])
            s1 = s0 + n_target
            if s0 < 0 or s1 > src.shape[-1]:
                continue
            slices.append(src[eidx, :, s0:s1])

        if len(slices) == 0:
            logger.warning("No sub-epochs could be extracted from ENG-03 trials")
            data_1 = np.zeros((1, len(epochs35.ch_names), n_target))
            placeholder = mne.EpochsArray(
                data_1,
                info=epochs35.info.copy(),
                tmin=tmin,
                baseline=ERP_CONFIG["baseline"],
                verbose=False,
            )
            placeholder.drop([0], reason="PLACEHOLDER")
            return placeholder

        data = np.stack(slices, axis=0)
        sub_epochs = mne.EpochsArray(
            data,
            info=epochs35.info.copy(),
            tmin=tmin,
            baseline=ERP_CONFIG["baseline"],
            verbose=False,
        )
        logger.info(f"Extracted {len(sub_epochs)} sub-epochs from ENG-03 35s trials")
        return sub_epochs

    # ------------------------------------------------------------------
    # ERP computation and P300 quantification
    # ------------------------------------------------------------------

    def _compute_erp(self, epochs: mne.Epochs) -> tuple[mne.Evoked, mne.Evoked]:
        """Average epochs to produce the ERP and standard error."""
        erp = epochs.average()
        erp_sem = epochs.standard_error()
        logger.info(f"Computed ERP from {len(epochs)} epochs")
        return erp, erp_sem

    def _compute_difference_erp(self, rare_erp: mne.Evoked, standard_erp: mne.Evoked) -> mne.Evoked:
        """Compute difference ERP (rare - standard)."""
        if rare_erp.data.shape != standard_erp.data.shape:
            logger.error(
                "Rare and standard ERPs have mismatched shapes: "
                f"{rare_erp.data.shape} vs {standard_erp.data.shape}",
            )
            raise ValueError("Rare and standard ERPs must have identical channel/time dimensions")

        diff_data = rare_erp.data - standard_erp.data
        info = rare_erp.info.copy()

        try:
            if not info.get_montage():
                montage = mne.channels.make_standard_montage("standard_1020")
                info.set_montage(montage, on_missing="ignore")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not set montage for topomap: {e}")

        diff_erp = mne.EvokedArray(
            diff_data,
            info=info,
            tmin=rare_erp.tmin,
            verbose=False,
        )
        logger.info("Computed difference ERP (rare - standard)")
        return diff_erp

    def _generate_qc_notes(self, composite: Dict[str, Any]) -> str:
        """Build a concise QC summary from electrode checks."""
        n_valid = composite["n_valid_electrodes"]
        n_flagged = composite["n_flagged_electrodes"]

        if n_flagged == 0:
            qc_part = f"All electrodes valid, averaged {n_valid}/3"
        else:
            issues: List[str] = []
            for electrode in ["Pz", "Cz", "Fz"]:
                if not composite.get(f"{electrode}_is_valid", False):
                    electrode_issues = composite.get(f"{electrode}_issues", "")
                    if "negative_or_zero_amplitude" in electrode_issues:
                        issues.append(f"{electrode} inverted")
                    elif "latency_out_of_range" in electrode_issues:
                        issues.append(f"{electrode} latency OOR")
                    elif "latency_atypical" in electrode_issues:
                        issues.append(f"{electrode} atypical latency")

            best = composite.get("best_electrode")
            if n_valid == 1 and best:
                qc_part = f"{', '.join(issues)} (used {best} only)"
            else:
                qc_part = ", ".join(issues)

        subtype = composite.get("p300_subtype", "unknown")
        if subtype == "P3a":
            subtype_note = "P3a pattern (Fz-max) — P3b may be absent"
        elif subtype == "P3b":
            subtype_note = "P3b pattern (Pz-max)"
        elif subtype == "mixed":
            subtype_note = "Mixed pattern (Cz-max)"
        elif subtype == "absent":
            subtype_note = "No P300 detected"
        else:
            subtype_note = str(subtype)

        return f"{qc_part}; {subtype_note}"

    def _quantify_p300(
        self,
        erp: mne.Evoked,
        patient_id: str,
        date: str,
        n_epochs: int,
        custom_electrodes: Optional[List[str]] = None,
        diff_erp: Optional[mne.Evoked] = None,
        n_standard_epochs: Optional[int] = None,
        n_standard_events_candidate: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Quantify P300 features from ERP."""
        features: Dict[str, Any] = {
            "patient_id": patient_id,
            "date": date,
            "n_epochs": n_epochs,
            "processing_timestamp": datetime.now().isoformat(),
        }

        baseline_mask = (erp.times >= ERP_CONFIG["tmin"]) & (erp.times <= 0)
        baseline_data = erp.data[:, baseline_mask]
        features["baseline_std_uV"] = float(np.std(baseline_data) * 1e6)

        if custom_electrodes:
            logger.info(f"{patient_id}: Using custom electrodes: {custom_electrodes}")
            for electrode in custom_electrodes:
                p300_features = self._detect_p300_peak(erp, electrode)
                features[f"p300_amplitude_{electrode}_uV"] = p300_features["amplitude"]
                features[f"p300_latency_{electrode}_ms"] = p300_features["latency"]
            features["qc_notes"] = f"Custom electrode analysis: {','.join(custom_electrodes)}"
        else:
            for electrode in ERP_CONFIG["midline_electrodes"]:
                p300_features = self._detect_p300_peak(erp, electrode)
                features[f"p300_amplitude_{electrode}_uV"] = p300_features["amplitude"]
                features[f"p300_latency_{electrode}_ms"] = p300_features["latency"]

            if diff_erp is not None:
                for electrode in ERP_CONFIG["midline_electrodes"]:
                    diff_features = self._detect_p300_peak(diff_erp, electrode)
                    features[f"diff_amplitude_{electrode}_uV"] = diff_features["amplitude"]
                    features[f"diff_latency_{electrode}_ms"] = diff_features["latency"]

            composite = self._compute_composite_p300(erp, patient_id)

            features.update(
                {
                    "p300_composite_amplitude_uV": composite["composite_amplitude"],
                    "p300_composite_latency_ms": composite["composite_latency"],
                    "p300_best_electrode": composite["best_electrode"],
                    "p300_n_valid_electrodes": composite["n_valid_electrodes"],
                    "p300_n_flagged_electrodes": composite["n_flagged_electrodes"],
                    "p300_subtype": composite.get("p300_subtype", "unknown"),
                },
            )

            features["qc_notes"] = self._generate_qc_notes(composite)

            features["p300_amplitude_uV"] = composite["composite_amplitude"]
            features["p300_latency_ms"] = composite["composite_latency"]

            if composite["n_flagged_electrodes"] > 0:
                logger.warning(
                    f"{patient_id} QC warning: {composite['n_flagged_electrodes']} electrode(s) flagged - "
                    f"{features['qc_notes']}",
                )

        # Merge mapping diagnostics (from rare event mapping)
        if self._last_epoch_diagnostics is not None:
            for key, value in self._last_epoch_diagnostics.items():
                if key not in features:
                    features[key] = value

        if n_standard_epochs is not None:
            features["n_standard_epochs"] = n_standard_epochs
        if n_standard_events_candidate is not None:
            features["n_standard_events"] = n_standard_events_candidate

        return features

    def _detect_p300_peak(self, erp: mne.Evoked, electrode: str) -> Dict[str, float]:
        """Detect P300 peak in a specific electrode."""
        electrode_names = [ch.upper() for ch in erp.ch_names]
        electrode_upper = electrode.upper()

        if electrode_upper not in electrode_names:
            available = ", ".join(erp.ch_names[:10])
            if len(erp.ch_names) > 10:
                available += f" ... ({len(erp.ch_names)} total)"
            logger.warning(
                f"Electrode {electrode} not found. Available: {available}. "
                "Use --electrodes or custom mode to see all electrodes.",
            )
            return {"amplitude": float("nan"), "latency": float("nan")}

        ch_idx = electrode_names.index(electrode_upper)
        data = erp.data[ch_idx, :]
        times = erp.times

        window_start, window_end = ERP_CONFIG["p300_window"]
        window_mask = (times >= window_start) & (times <= window_end)

        if not window_mask.any():
            logger.warning(f"P300 window outside epoch range for {electrode}")
            return {"amplitude": float("nan"), "latency": float("nan")}

        window_data = data[window_mask]
        window_times = times[window_mask]

        peak_idx = np.argmax(window_data)
        amplitude = float(window_data[peak_idx] * 1e6)
        latency = float(window_times[peak_idx] * 1000)

        return {"amplitude": amplitude, "latency": latency}

    def _validate_p300_electrode(
        self,
        electrode: str,
        amplitude: float,
        latency: float,
        patient_id: str,
    ) -> Dict[str, Any]:
        """Validate P300 quality for one electrode."""
        validation: Dict[str, Any] = {
            "is_valid": True,
            "is_positive": True,
            "is_on_time": True,
            "is_expected_latency": True,
            "issues": [],
        }

        if np.isnan(amplitude) or np.isnan(latency):
            validation["is_valid"] = False
            validation["issues"].append("missing_data")
            return validation

        if amplitude <= ERP_CONFIG["p300_min_amplitude"]:
            validation["is_valid"] = False
            validation["is_positive"] = False
            validation["issues"].append("negative_or_zero_amplitude")
            logger.warning(
                f"{patient_id} - {electrode}: Negative/zero amplitude ({amplitude:.2f}µV) - "
                "likely inverted reference or absent P300",
            )

        min_lat, max_lat = ERP_CONFIG["p300_max_latency_range"]
        if not (min_lat <= latency <= max_lat):
            validation["is_valid"] = False
            validation["is_on_time"] = False
            validation["issues"].append("latency_out_of_range")
            logger.warning(
                f"{patient_id} - {electrode}: Latency {latency:.1f}ms outside acceptable range "
                f"[{min_lat}-{max_lat}ms]",
            )

        exp_min, exp_max = ERP_CONFIG["p300_expected_latency_range"]
        if not (exp_min <= latency <= exp_max):
            validation["is_expected_latency"] = False
            validation["issues"].append("latency_atypical")
            logger.info(
                f"{patient_id} - {electrode}: Latency {latency:.1f}ms outside "
                f"typical range [{exp_min}-{exp_max}ms] but within acceptable limits",
            )

        return validation

    def _compute_composite_p300(self, erp: mne.Evoked, patient_id: str) -> Dict[str, Any]:
        """Compute composite P300 from valid midline electrodes."""
        electrode_data: Dict[str, Dict[str, Any]] = {}
        valid_amplitudes: List[float] = []
        valid_latencies: List[float] = []
        valid_electrodes: List[str] = []
        flagged_electrodes: List[str] = []

        for electrode in ERP_CONFIG["midline_electrodes"]:
            p300 = self._detect_p300_peak(erp, electrode)
            amplitude = p300["amplitude"]
            latency = p300["latency"]

            validation = self._validate_p300_electrode(electrode, amplitude, latency, patient_id)

            electrode_data[electrode] = {
                "amplitude": amplitude,
                "latency": latency,
                "is_valid": validation["is_valid"],
                "is_positive": validation["is_positive"],
                "is_on_time": validation["is_on_time"],
                "is_expected_latency": validation["is_expected_latency"],
                "issues": validation["issues"],
            }

            if validation["is_valid"]:
                valid_amplitudes.append(amplitude)
                valid_latencies.append(latency)
                valid_electrodes.append(electrode)
            else:
                flagged_electrodes.append(electrode)

        if valid_amplitudes:
            composite: Dict[str, Any] = {
                "composite_amplitude": float(np.mean(valid_amplitudes)),
                "composite_amplitude_std": float(np.std(valid_amplitudes)) if len(valid_amplitudes) > 1 else 0.0,
                "composite_latency": float(np.mean(valid_latencies)),
                "composite_latency_std": float(np.std(valid_latencies)) if len(valid_latencies) > 1 else 0.0,
                "n_valid_electrodes": len(valid_electrodes),
                "valid_electrodes": ",".join(valid_electrodes),
                "best_electrode": max(zip(valid_electrodes, valid_amplitudes), key=lambda x: x[1])[0],
                "best_electrode_amplitude": max(valid_amplitudes),
            }

            logger.info(
                "%s: Composite P300 = %.2fµV (±%.2f) from %d electrodes: %s",
                patient_id,
                composite["composite_amplitude"],
                composite["composite_amplitude_std"],
                len(valid_electrodes),
                ", ".join(valid_electrodes),
            )
        else:
            composite = {
                "composite_amplitude": float("nan"),
                "composite_amplitude_std": float("nan"),
                "composite_latency": float("nan"),
                "composite_latency_std": float("nan"),
                "n_valid_electrodes": 0,
                "valid_electrodes": "",
                "best_electrode": None,
                "best_electrode_amplitude": float("nan"),
            }
            logger.warning(f"{patient_id}: No valid P300 detected in any electrode!")

        composite["n_flagged_electrodes"] = len(flagged_electrodes)
        composite["flagged_electrodes"] = ",".join(flagged_electrodes) if flagged_electrodes else ""

        for electrode in ERP_CONFIG["midline_electrodes"]:
            data = electrode_data[electrode]
            composite[f"{electrode}_is_valid"] = data["is_valid"]
            composite[f"{electrode}_is_positive"] = data["is_positive"]
            composite[f"{electrode}_issues"] = ",".join(data["issues"]) if data["issues"] else ""

        if flagged_electrodes:
            for elec in flagged_electrodes:
                data = electrode_data[elec]
                logger.warning(
                    "%s - %s FLAGGED: amplitude=%.2fµV, latency=%.1fms, issues=%s",
                    patient_id,
                    elec,
                    data["amplitude"],
                    data["latency"],
                    data["issues"],
                )

        n_valid = composite["n_valid_electrodes"]
        if n_valid == 0:
            subtype = "absent"
        else:
            best_elec = composite.get("best_electrode")
            if best_elec == "Pz":
                subtype = "P3b"
            elif best_elec == "Fz":
                subtype = "P3a"
            elif best_elec == "Cz":
                subtype = "mixed"
            else:
                subtype = "absent"

        composite["p300_subtype"] = subtype
        return composite

    # ------------------------------------------------------------------
    # Persistence and plotting
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        patient_id: str,
        date: str,
        epochs: mne.Epochs,
        erp: mne.Evoked,
        features: Dict[str, Any],
        standard_erp: Optional[mne.Evoked] = None,
        diff_erp: Optional[mne.Evoked] = None,
    ) -> None:
        """Save ERP and features to disk."""
        erp_file = self._output_paths.erps / f"{patient_id}_{date}_oddball-ave.fif"
        erp.save(erp_file, overwrite=True)
        logger.info("Saved ERP: %s", erp_file)

        if standard_erp is not None:
            std_file = self._output_paths.erps / f"{patient_id}_{date}_oddball_standard-ave.fif"
            standard_erp.save(std_file, overwrite=True)
            logger.info("Saved standard ERP: %s", std_file)

        if diff_erp is not None:
            diff_file = self._output_paths.erps / f"{patient_id}_{date}_oddball_diff-ave.fif"
            diff_erp.save(diff_file, overwrite=True)
            logger.info("Saved difference ERP: %s", diff_file)

        # Build and save three structured feature tables
        clinical_df = self._build_clinical_table(patient_id, date, features)
        detail_df = self._build_electrode_detail_table(patient_id, date, features)
        qc_df = self._build_mapping_qc_table(patient_id, date, features)

        self._update_master_feature_tables(clinical_df, detail_df, qc_df)

    def _build_clinical_table(
        self, patient_id: str, date: str, features: Dict[str, Any]
    ) -> pd.DataFrame:
        """Build Table 1: Main analysis table (one row per patient-session)."""
        qc_pass = (
            features.get("p300_n_valid_electrodes", 0) >= 2
            and features.get("p300_subtype") != "absent"
        )

        return pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "session_date": date,
                    "n_rare_epochs": features.get("n_epochs"),
                    "n_standard_epochs": features.get("n_standard_epochs"),
                    "baseline_std_uV": features.get("baseline_std_uV"),
                    "p300_diff_amplitude_Pz_uV": features.get("diff_amplitude_Pz_uV"),
                    "p300_diff_latency_Pz_ms": features.get("diff_latency_Pz_ms"),
                    "p300_diff_amplitude_Cz_uV": features.get("diff_amplitude_Cz_uV"),
                    "p300_diff_latency_Cz_ms": features.get("diff_latency_Cz_ms"),
                    "p300_diff_amplitude_Fz_uV": features.get("diff_amplitude_Fz_uV"),
                    "p300_diff_latency_Fz_ms": features.get("diff_latency_Fz_ms"),
                    "p300_best_electrode": features.get("p300_best_electrode"),
                    "p300_subtype": features.get("p300_subtype"),
                    "p300_amplitude_uV": features.get("p300_amplitude_uV"),
                    "p300_latency_ms": features.get("p300_latency_ms"),
                    "p300_n_valid_electrodes": features.get("p300_n_valid_electrodes"),
                    "qc_notes": features.get("qc_notes", ""),
                    "qc_pass": qc_pass,
                }
            ]
        )

    def _build_electrode_detail_table(
        self, patient_id: str, date: str, features: Dict[str, Any]
    ) -> pd.DataFrame:
        """Build Table 2: Per-electrode breakdown (one row per electrode per session)."""
        rows = []
        for electrode in ["Fz", "Cz", "Pz"]:
            amp_key = f"p300_amplitude_{electrode}_uV"
            lat_key = f"p300_latency_{electrode}_ms"
            diff_amp_key = f"diff_amplitude_{electrode}_uV"
            diff_lat_key = f"diff_latency_{electrode}_ms"

            amp = features.get(amp_key)
            lat = features.get(lat_key)

            is_valid = amp is not None and amp > 0 and lat is not None and 250 <= lat <= 600
            flagged_reason = None
            if amp is None or (isinstance(amp, float) and np.isnan(amp)):
                flagged_reason = "missing"
            elif amp <= 0:
                flagged_reason = "inverted"
            elif lat is None or (isinstance(lat, float) and np.isnan(lat)):
                flagged_reason = "missing_latency"
            elif not (250 <= lat <= 600):
                flagged_reason = "out_of_range"

            rows.append(
                {
                    "patient_id": patient_id,
                    "session_date": date,
                    "electrode": electrode,
                    "p300_amplitude_uV": amp,
                    "p300_latency_ms": lat,
                    "is_valid": is_valid,
                    "flagged_reason": flagged_reason,
                    "diff_amplitude_uV": features.get(diff_amp_key),
                    "diff_latency_ms": features.get(diff_lat_key),
                }
            )

        return pd.DataFrame(rows)

    def _build_mapping_qc_table(
        self, patient_id: str, date: str, features: Dict[str, Any]
    ) -> pd.DataFrame:
        """Build Table 3: Mapping & QC diagnostics (one row per patient-session)."""
        return pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "session_date": date,
                    "n_rare_events_candidate": features.get("n_rare_events"),
                    "n_rare_mapped": features.get("n_mapped"),
                    "n_rare_unmapped": features.get("n_unmapped"),
                    "n_rare_boundary_clipped": features.get("n_boundary_clipped"),
                    "rare_mapping_rate": features.get("mapping_rate"),
                    "n_standard_events_candidate": features.get("n_standard_events"),
                    "n_standard_mapped": features.get("n_standard_epochs"),
                    "processing_timestamp": features.get("processing_timestamp"),
                    "pipeline_version": "ENG-02b_v1.0",
                }
            ]
        )

    def _update_master_feature_tables(
        self,
        clinical_df: pd.DataFrame,
        detail_df: pd.DataFrame,
        qc_df: pd.DataFrame,
    ) -> None:
        """Upsert three feature tables into master parquet files."""
        # Table 1: Clinical
        clinical_path = self._output_paths.features / "p300_oddball_clinical.parquet"
        if clinical_path.exists():
            master_clinical = pd.read_parquet(clinical_path)
            clinical_combined = pd.concat(
                [master_clinical, clinical_df], ignore_index=True
            )
        else:
            clinical_combined = clinical_df.copy()

        clinical_combined = clinical_combined.drop_duplicates(
            subset=["patient_id", "session_date"], keep="last"
        )
        clinical_combined.to_parquet(clinical_path, index=False)
        logger.info(
            "Updated clinical table: %s (%d rows)", clinical_path, len(clinical_combined)
        )

        # Table 2: Electrode detail
        detail_path = self._output_paths.features / "p300_oddball_electrode_detail.parquet"
        if detail_path.exists():
            master_detail = pd.read_parquet(detail_path)
            detail_combined = pd.concat([master_detail, detail_df], ignore_index=True)
        else:
            detail_combined = detail_df.copy()

        detail_combined = detail_combined.drop_duplicates(
            subset=["patient_id", "session_date", "electrode"], keep="last"
        )
        detail_combined.to_parquet(detail_path, index=False)
        logger.info(
            "Updated electrode detail table: %s (%d rows)",
            detail_path,
            len(detail_combined),
        )

        # Table 3: Mapping QC
        qc_path = self._output_paths.features / "p300_oddball_mapping_qc.parquet"
        if qc_path.exists():
            master_qc = pd.read_parquet(qc_path)
            qc_combined = pd.concat([master_qc, qc_df], ignore_index=True)
        else:
            qc_combined = qc_df.copy()

        qc_combined = qc_combined.drop_duplicates(
            subset=["patient_id", "session_date"], keep="last"
        )
        qc_combined.to_parquet(qc_path, index=False)
        logger.info("Updated mapping QC table: %s (%d rows)", qc_path, len(qc_combined))

    def _plot_erp_panels(
        self,
        erp: mne.Evoked,
        axes: Any,
        title_top: str,
        electrodes_to_plot: List[str],
        panel_title_bottom: str,
        color_map_or_list: Any,
    ) -> None:
        """Draw two-panel ERP plot (butterfly + selected electrodes)."""
        times = erp.times * 1000
        data = erp.data * 1e6
        ch_names_upper = [ch.upper() for ch in erp.ch_names]

        for ch_idx in range(data.shape[0]):
            axes[0].plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)
        axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        axes[0].axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title(title_top)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        is_dict = isinstance(color_map_or_list, dict)
        for idx, electrode in enumerate(electrodes_to_plot):
            color = color_map_or_list[electrode] if is_dict else color_map_or_list[idx % len(color_map_or_list)]
            try:
                elec_idx = ch_names_upper.index(electrode.upper())
                axes[1].plot(times, data[elec_idx, :], linewidth=2, color=color, label=electrode)
            except ValueError:
                logger.warning("Electrode %s not found in data", electrode)
        axes[1].axvline(x=0, color="k", linestyle="--", linewidth=1)
        axes[1].axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        axes[1].axhline(y=0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Amplitude (µV)")
        axes[1].set_title(panel_title_bottom)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    def _plot_erp_figure(
        self,
        rare_erp: mne.Evoked,
        rare_sem: mne.Evoked,
        standard_erp: Optional[mne.Evoked],
        standard_sem: Optional[mne.Evoked],
        diff_erp: Optional[mne.Evoked],
        features: Dict[str, Any],
        patient_id: str,
        date: str,
        custom_electrodes: Optional[List[str]] = None,
    ) -> None:
        """Generate and save ERP summary figure (3-panel when diff_erp is available)."""
        if diff_erp is None or standard_erp is None:
            self._plot_individual_erp_legacy(rare_erp, patient_id, date, custom_electrodes)
            return

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

        ax1 = fig.add_subplot(gs[0])
        times = rare_erp.times * 1000
        data = rare_erp.data * 1e6
        for ch_idx in range(data.shape[0]):
            ax1.plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)
        ax1.axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        ax1.axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.set_title(f"{patient_id} - {date} - All Channels (Butterfly)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1])
        electrodes = ["Fz", "Cz", "Pz"]
        colors = {"Fz": "red", "Cz": "green", "Pz": "blue"}

        ch_names_upper = [ch.upper() for ch in rare_erp.ch_names]
        for electrode in electrodes:
            if electrode.upper() not in ch_names_upper:
                continue
            ch_idx = ch_names_upper.index(electrode.upper())

            rare_trace = rare_erp.data[ch_idx, :] * 1e6
            rare_sem_trace = rare_sem.data[ch_idx, :] * 1e6
            color = colors[electrode]

            ax2.plot(times, rare_trace, linewidth=2, color=color, label=f"{electrode} (rare)")
            ax2.fill_between(times, rare_trace - rare_sem_trace, rare_trace + rare_sem_trace, alpha=0.2, color=color)

            if standard_erp is not None and standard_sem is not None:
                std_trace = standard_erp.data[ch_idx, :] * 1e6
                std_sem_trace = standard_sem.data[ch_idx, :] * 1e6
                ax2.plot(times, std_trace, linewidth=1.5, color=color, linestyle="--", label=f"{electrode} (std)")
                ax2.fill_between(
                    times,
                    std_trace - std_sem_trace,
                    std_trace + std_sem_trace,
                    alpha=0.1,
                    color=color,
                )

        ax2.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax2.axvspan(300, 600, alpha=0.1, color="gray")
        ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Amplitude (µV)")
        subtype = features.get("p300_subtype", "unknown")
        ax2.set_title(f"Rare vs Standard — {subtype}")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[2])
        diff_data = diff_erp.data * 1e6
        for electrode in electrodes:
            if electrode.upper() not in ch_names_upper:
                continue
            ch_idx = ch_names_upper.index(electrode.upper())
            diff_trace = diff_data[ch_idx, :]
            color = colors[electrode]
            ax3.plot(times, diff_trace, linewidth=2, color=color, label=electrode)

        ax3.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax3.axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        ax3.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Amplitude (µV)")
        ax3.set_title("Difference Wave (Rare - Standard)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self._output_paths.plots_erp / f"{patient_id}_{date}_oddball_erp.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 3-panel ERP plot: %s", save_path)

        self._plot_topomap(diff_erp, patient_id, date)

    def _plot_erp_image(self, epochs: mne.Epochs, patient_id: str, date: str) -> None:
        """Generate ERP image (single-trial heatmap) for rare 900ms epochs at Pz."""
        if len(epochs) < 3:
            logger.debug("%s %s: skipping ERP image (only %d epochs < 3)", patient_id, date, len(epochs))
            return

        try:
            ret = mne.viz.plot_epochs_image(
                epochs,
                picks=["Pz"],
                show=False,
            )
            fig = ret[0] if isinstance(ret, (list, tuple)) else ret

            fig.set_size_inches(12, 10)
            fig.subplots_adjust(top=0.85, bottom=0.15, hspace=0.5)

            title_text = (
                "ERP Image: Single-Trial Responses to Rare (Target) Stimuli at Pz — "
                f"{patient_id} | {date}"
            )
            fig.text(
                0.5,
                0.98,
                title_text,
                ha="center",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            )

            caption = (
                "Top: Each row = one trial. Bottom: Average. Time 0 = stimulus. "
                "Color = voltage (µV)."
            )
            fig.text(
                0.5,
                0.03,
                caption,
                ha="center",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            )

            save_path = self._output_paths.plots_erp / f"{patient_id}_{date}_oddball_erp_image.png"
            fig.savefig(save_path, dpi=150, pad_inches=0.4)
            plt.close(fig)
            logger.info("Saved ERP image: %s", save_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not generate ERP image for %s %s: %s", patient_id, date, e)

    def _plot_topomap(self, diff_erp: mne.Evoked, patient_id: str, date: str) -> None:
        """Save topomap series for the difference ERP."""
        try:
            logger.info("Generating topomap series for %s %s", patient_id, date)
            times_to_plot = np.arange(-0.2, 0.75, 0.1)

            # MNE version in this environment does not support selective naming,
            # so we enable show_names=True (all labels) rather than passing a
            # custom `names` argument.
            fig = diff_erp.plot_topomap(
                times=times_to_plot,
                show=False,
                colorbar=True,
                size=5,
                show_names=True,
            )

            if isinstance(fig, (list, tuple)):
                fig_obj = fig[0]
            else:
                fig_obj = fig

            save_path = self._output_paths.plots_erp / f"{patient_id}_{date}_oddball_topomap.png"
            fig_obj.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig_obj)
            logger.info("Saved topomap series: %s", save_path)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Failed to generate topomap for %s %s: %s", patient_id, date, e, exc_info=True)

    def _plot_individual_erp_legacy(
        self,
        erp: mne.Evoked,
        patient_id: str,
        date: str,
        custom_electrodes: Optional[List[str]] = None,
    ) -> None:
        """Generate and save individual ERP plot (2-panel legacy format)."""
        save_path = self._output_paths.plots_erp / f"{patient_id}_{date}_oddball_erp.png"
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        if custom_electrodes:
            electrodes_to_plot = custom_electrodes
            panel_title = f"{patient_id} - {date} - Custom Electrodes: {', '.join(custom_electrodes)}"
            colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
        else:
            electrodes_to_plot = ["Fz", "Cz", "Pz"]
            panel_title = f"{patient_id} - {date} - Midline Electrodes (Composite Scoring)"
            colors = ["red", "green", "blue"]

        self._plot_erp_panels(
            erp,
            axes,
            title_top=f"{patient_id} - {date} - All Channels",
            electrodes_to_plot=electrodes_to_plot,
            panel_title_bottom=panel_title,
            color_map_or_list=colors,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ERP plot: %s", save_path)

