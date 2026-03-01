"""ERP/Oddball Pipeline (ENG-02b).

Loads ICA-cleaned 35s epochs from ENG-03, maps rare-event timestamps into trial windows,
extracts 900ms sub-epochs, computes ERPs, and quantifies P300 features.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader

logger = logging.getLogger(__name__)


ERP_CONFIG = {
    "tmin": -0.2,  # Epoch start: 200ms before stimulus
    "tmax": 0.7,  # Epoch end: 700ms after stimulus
    "baseline": (None, 0),  # Baseline correction: -200ms to 0ms
    "p300_window": (0.3, 0.6),  # P300 search window: 300-600ms
    "min_epochs": 2,  # Minimum rare events needed per trial
    "midline_electrodes": ["Pz", "Cz", "Fz"],
    "p300_min_amplitude": 0.0,  # µV — must be positive
    "p300_expected_latency_range": (300, 500),  # typical range for controls (ms)
    "p300_max_latency_range": (250, 600),  # hard rejection cutoff (ms)
}


class OddballERPPipeline:
    """Pipeline for extracting and analyzing ERPs from oddball trials."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        loader: Optional[UnifiedDataLoader] = None,
        verbose: bool = False,
    ):
        """
        Initialize ERP pipeline.

        Args:
            data_root: Root directory for data files (defaults to config.LOCAL_DATA_ROOT)
            output_dir: Output directory for processed files (defaults to config.PROCESSED_DATA_DIR)
            loader: Optional shared UnifiedDataLoader (reuses LRU cache when running with ENG-03 etc.)
            verbose: If True, enable detailed logging
        """
        self.data_root = data_root or config.LOCAL_DATA_ROOT
        self.output_dir = output_dir or config.PROCESSED_DATA_DIR
        self.verbose = verbose

        self.loader = loader if loader is not None else UnifiedDataLoader(data_root=self.data_root, verbose=verbose)

        if not verbose:
            mne.set_log_level("WARNING")

        self._output_paths = self._get_output_paths()
        self._create_output_directories()
        self._last_epoch_diagnostics: Dict[str, Any] = {}

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

    def _create_output_directories(self):
        p = self._output_paths
        for dir_path in [p.erps, p.features, p.plots_erp, p.qc]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def process_patient(
        self,
        patient_id: str,
        date: Optional[str] = None,
        custom_electrodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single patient (or single session if date is specified).

        Args:
            patient_id: Patient identifier
            date: Specific session date (YYYY-MM-DD). If None, processes all sessions.
            custom_electrodes: Optional list of custom electrodes to analyze instead of default Pz/Cz/Fz

        Returns:
            Dictionary containing epochs, erp, features, and metadata
        """
        logger.info(f"Processing patient: {patient_id}")

        try:
            aligned_trials = self._load_aligned_trials(patient_id)

            if aligned_trials.empty:
                logger.warning(f"No aligned trials found for {patient_id}")
                return {"patient_id": patient_id, "status": "no_data"}

            if date:
                aligned_trials = aligned_trials[aligned_trials["date"] == date]
                if aligned_trials.empty:
                    logger.warning(f"No trials found for {patient_id} on {date}")
                    return {"patient_id": patient_id, "date": date, "status": "no_data"}

            all_session_results = []
            sessions = aligned_trials["date"].unique()

            for session_date in sessions:
                session_trials = aligned_trials[aligned_trials["date"] == session_date]
                session_result = self._process_session(
                    patient_id,
                    session_date,
                    session_trials,
                    custom_electrodes=custom_electrodes,
                )
                if session_result["status"] != "success":
                    logger.warning(
                        "Session %s %s: %s",
                        patient_id,
                        session_date,
                        session_result["status"],
                    )
                all_session_results.append(session_result)

            successful = [r for r in all_session_results if r.get("status") == "success"]
            if not successful:
                return {
                    "patient_id": patient_id,
                    "status": "no_successful_sessions",
                    "session_results": all_session_results,
                }

            if len(successful) == 1:
                return successful[0]
            else:
                all_features = pd.concat([r["features"] for r in successful], ignore_index=True)
                return {
                    "patient_id": patient_id,
                    "status": "success",
                    "sessions": len(successful),
                    "features": all_features,
                }

        except Exception as e:
            logger.error(f"Failed to process {patient_id}: {e}", exc_info=True)
            return {"patient_id": patient_id, "status": "error", "error": str(e)}

    def _process_session(
        self,
        patient_id: str,
        date: str,
        aligned_trials: pd.DataFrame,
        custom_electrodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single session.

        Args:
            patient_id: Patient identifier
            date: Session date
            aligned_trials: DataFrame of aligned trials for this session
            custom_electrodes: Optional list of custom electrodes to analyze instead of default Pz/Cz/Fz

        Returns:
            Dictionary with epochs, erp, features, and metadata
        """
        logger.info(f"Processing session: {patient_id} - {date}")

        try:
            logger.info(f"Loading ENG-03 oddball epochs for {patient_id} - {date}")
            try:
                epochs35 = self.loader.load_clean_epochs(
                    patient_id=patient_id,
                    date=date,
                    trial_type="oddball",
                )
            except FileNotFoundError as e:
                error_msg = (
                    f"ENG-03 oddball epochs not found for {patient_id} - {date}. "
                    f"Run ENG-03 (ArtifactRejector.run_session) first. {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            rare_events = self._extract_rare_events(aligned_trials)

            if len(rare_events) < ERP_CONFIG["min_epochs"]:
                logger.warning(
                    f"Insufficient rare events for {patient_id} on {date}: "
                    f"{len(rare_events)} < {ERP_CONFIG['min_epochs']}"
                )
                return {
                    "patient_id": patient_id,
                    "date": date,
                    "status": "insufficient_data",
                }

            trial_windows = self._build_trial_windows(epochs35)
            mapped_df, mapping_diag = self._map_rare_to_trials(
                rare_events,
                trial_windows,
                sfreq=float(epochs35.info["sfreq"]),
            )
            self._last_epoch_diagnostics = mapping_diag

            epochs = self._extract_subepochs(epochs35, mapped_df)

            if len(epochs) < ERP_CONFIG["min_epochs"]:
                logger.warning(f"Insufficient epochs after mapping for {patient_id} on {date}: {len(epochs)}")
                return {
                    "patient_id": patient_id,
                    "date": date,
                    "status": "insufficient_epochs",
                }

            erp = self._compute_erp(epochs)
            features = self._quantify_p300(
                erp,
                patient_id,
                date,
                len(epochs),
                custom_electrodes=custom_electrodes,
            )

            self._save_outputs(patient_id, date, epochs, erp, features)
            self._plot_individual_erp(erp, patient_id, date, custom_electrodes=custom_electrodes)

            if custom_electrodes:
                logger.info(
                    f"Successfully processed {patient_id} - {date}: {len(epochs)} epochs, custom electrode mode"
                )
            else:
                logger.info(
                    f"Successfully processed {patient_id} - {date}: "
                    f"{len(epochs)} epochs, P300 amplitude = {features.get('composite_p300_amplitude', 0):.2f}µV"
                )

            return {
                "patient_id": patient_id,
                "date": date,
                "status": "success",
                "epochs": epochs,
                "erp": erp,
                "features": pd.DataFrame([features]),
            }

        except Exception as e:
            logger.error(f"Failed to process session {patient_id} - {date}: {e}", exc_info=True)
            return {
                "patient_id": patient_id,
                "date": date,
                "status": "error",
                "error": str(e),
            }

    def _load_aligned_trials(self, patient_id: str) -> pd.DataFrame:
        """
        Load aligned events for a patient from ENG-02 output.

        Args:
            patient_id: Patient identifier

        Returns:
            DataFrame of aligned trials
        """
        aligned_path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"

        if not aligned_path.exists():
            logger.warning(f"No aligned events file found: {aligned_path}")
            return pd.DataFrame()

        df = pd.read_parquet(aligned_path, engine="pyarrow")
        oddball_trials = df[df["trial_type"].str.lower() == "oddball"].copy()

        logger.info(f"Loaded {len(oddball_trials)} oddball trials for {patient_id}")
        return oddball_trials

    def _extract_rare_events(self, trials_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract rare beep events with timestamps from oddball trials.

        Args:
            trials_df: DataFrame of oddball trials

        Returns:
            List of rare event dictionaries with timestamps
        """
        rare_events = []

        for idx, trial in trials_df.iterrows():
            try:
                events = trial["sentences"]
            except (KeyError, TypeError):
                logger.warning(f"Trial {idx}: No 'sentences' field found")
                continue

            # pandas deserialises nested parquet columns as numpy arrays, not lists
            if isinstance(events, np.ndarray):
                events = events.tolist()

            event_len = len(events) if isinstance(events, (list, np.ndarray)) else "N/A"
            logger.debug(f"Trial {idx}: sentences type = {type(events)}, length = {event_len}")

            if not isinstance(events, list):
                logger.warning(f"Trial {idx}: sentences is not a list (type: {type(events)})")
                continue

            rare_count_in_trial = 0
            for event in events:
                if not isinstance(event, dict):
                    logger.debug(f"Trial {idx}: event is not a dict (type: {type(event)})")
                    continue

                if event.get("event") == "rare" and "event_start" in event:
                    rare_count_in_trial += 1
                    rare_events.append(
                        {
                            "timestamp_unix": event["event_start"],
                            "date": trial["date"],
                            "trial_idx": idx,
                            "correlation_score": event.get("correlation_score"),
                            "peak_amplitude": event.get("peak_amplitude"),
                        }
                    )

            logger.debug(f"Trial {idx}: found {rare_count_in_trial} rare events")

        logger.info(f"Extracted {len(rare_events)} rare events")
        return rare_events

    def _build_trial_windows(self, epochs35: mne.Epochs, window_sec: float = 35.0) -> pd.DataFrame:
        """Build a table of ENG-03 trial start/end times from epoch metadata.

        Args:
            epochs35: ENG-03 oddball epochs (35s fixed-window).
            window_sec: Trial window length in seconds.

        Returns:
            DataFrame with columns eng03_epoch_idx, start_time_unix, end_time_unix.
        """
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
            }
        )

    def _map_rare_to_trials(
        self,
        rare_events: List[Dict[str, Any]],
        trial_windows_df: pd.DataFrame,
        sfreq: float,
    ) -> tuple:
        """Map rare-event timestamps into ENG-03 35s trial windows.

        Args:
            rare_events: Rare events with ``timestamp_unix`` keys.
            trial_windows_df: Output of ``_build_trial_windows``.
            sfreq: Sampling frequency of the ENG-03 epochs.

        Returns:
            (mapped_df, diagnostics) where *mapped_df* contains only events that
            successfully map to exactly one trial without boundary clipping, and
            *diagnostics* is a summary dict.
        """
        tmin = ERP_CONFIG["tmin"]
        tmax = ERP_CONFIG["tmax"]
        starts = trial_windows_df["start_time_unix"].values
        window_sec = float(trial_windows_df["window_sec"].iloc[0])
        epoch_ids = trial_windows_df["eng03_epoch_idx"].values

        rows: List[Dict[str, Any]] = []
        n_unmapped = 0
        n_duplicate = 0
        n_boundary = 0

        for event in rare_events:
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
                }
            )

        mapped_df = pd.DataFrame(rows)
        diagnostics = {
            "n_rare_events": len(rare_events),
            "n_mapped": len(mapped_df),
            "n_unmapped": n_unmapped,
            "n_duplicate": n_duplicate,
            "n_boundary_clipped": n_boundary,
            "mapping_rate": len(mapped_df) / max(len(rare_events), 1),
        }
        logger.info(
            "Rare-event mapping: %d/%d mapped (unmapped=%d, duplicate=%d, boundary=%d)",
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
    ) -> mne.EpochsArray:
        """Slice 900ms sub-epochs from 35s ENG-03 epochs.

        Args:
            epochs35: Full 35s epoch objects.
            mapped_df: Output of ``_map_rare_to_trials`` with start_sample / end_sample.

        Returns:
            ``mne.EpochsArray`` of shape (n_events, n_channels, n_times).
        """
        tmin = ERP_CONFIG["tmin"]
        tmax = ERP_CONFIG["tmax"]
        sfreq = float(epochs35.info["sfreq"])
        n_target = int(np.round((tmax - tmin) * sfreq)) + 1

        src = epochs35.get_data()  # (n_epochs, n_channels, n_times)
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

    def _compute_erp(self, epochs: mne.Epochs) -> mne.Evoked:
        """Average epochs to produce the ERP."""
        erp = epochs.average()
        logger.info(f"Computed ERP from {len(epochs)} epochs")
        return erp

    def _generate_qc_notes(self, composite: Dict[str, Any]) -> str:
        """
        Build a concise QC summary from electrode checks.

        Args:
            composite: Composite P300 dictionary with validation results

        Returns:
            QC summary string
        """
        n_valid = composite["n_valid_electrodes"]
        n_flagged = composite["n_flagged_electrodes"]

        if n_flagged == 0:
            return f"All electrodes valid, averaged {n_valid}/3"

        issues = []
        for electrode in ["Pz", "Cz", "Fz"]:
            if not composite[f"{electrode}_is_valid"]:
                electrode_issues = composite[f"{electrode}_issues"]
                if "negative_or_zero_amplitude" in electrode_issues:
                    issues.append(f"{electrode} inverted")
                elif "latency_out_of_range" in electrode_issues:
                    issues.append(f"{electrode} latency OOR")
                elif "latency_atypical" in electrode_issues:
                    issues.append(f"{electrode} atypical latency")

        best = composite["best_electrode"]
        if n_valid == 1:
            return f"{', '.join(issues)} (used {best} only)"
        else:
            return f"{', '.join(issues)}"

    def _quantify_p300(
        self,
        erp: mne.Evoked,
        patient_id: str,
        date: str,
        n_epochs: int,
        custom_electrodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Quantify P300 features from ERP.

        Args:
            erp: MNE Evoked object (averaged ERP)
            patient_id: Patient identifier
            date: Session date
            n_epochs: Number of epochs averaged
            custom_electrodes: Optional list of custom electrodes to analyze instead of defaults

        Returns:
            Dictionary of P300 features
        """
        features = {
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
            # No composite scoring in custom mode
            features["qc_notes"] = f"Custom electrode analysis: {','.join(custom_electrodes)}"

        else:
            for electrode in ERP_CONFIG["midline_electrodes"]:
                p300_features = self._detect_p300_peak(erp, electrode)
                features[f"p300_amplitude_{electrode}_uV"] = p300_features["amplitude"]
                features[f"p300_latency_{electrode}_ms"] = p300_features["latency"]

            composite = self._compute_composite_p300(erp, patient_id)

            features.update(
                {
                    "p300_composite_amplitude_uV": composite["composite_amplitude"],
                    "p300_composite_latency_ms": composite["composite_latency"],
                    "p300_best_electrode": composite["best_electrode"],
                    "p300_n_valid_electrodes": composite["n_valid_electrodes"],
                    "p300_n_flagged_electrodes": composite["n_flagged_electrodes"],
                }
            )

            features["qc_notes"] = self._generate_qc_notes(composite)

            # Aliases expected by downstream code
            features["p300_amplitude_uV"] = composite["composite_amplitude"]
            features["p300_latency_ms"] = composite["composite_latency"]

            if composite["n_flagged_electrodes"] > 0:
                logger.warning(
                    f"{patient_id} QC warning: {composite['n_flagged_electrodes']} electrode(s) flagged - "
                    f"{features['qc_notes']}"
                )

        # Mapping diagnostics flow through to the feature table for traceability
        if self._last_epoch_diagnostics:
            for key, value in self._last_epoch_diagnostics.items():
                if key not in features:
                    features[key] = value

        return features

    def _detect_p300_peak(self, erp: mne.Evoked, electrode: str) -> Dict[str, float]:
        """
        Detect P300 peak in a specific electrode.

        Args:
            erp: MNE Evoked object
            electrode: Electrode name (e.g., 'Pz')

        Returns:
            Dictionary with amplitude and latency
        """
        electrode_names = [ch.upper() for ch in erp.ch_names]
        electrode_upper = electrode.upper()

        if electrode_upper not in electrode_names:
            available = ", ".join(erp.ch_names[:10])
            if len(erp.ch_names) > 10:
                available += f" ... ({len(erp.ch_names)} total)"
            logger.warning(
                f"Electrode {electrode} not found. Available: {available}. Use --list-electrodes to see all electrodes."
            )
            return {"amplitude": np.nan, "latency": np.nan}

        ch_idx = electrode_names.index(electrode_upper)
        data = erp.data[ch_idx, :]
        times = erp.times

        window_start, window_end = ERP_CONFIG["p300_window"]
        window_mask = (times >= window_start) & (times <= window_end)

        if not window_mask.any():
            logger.warning(f"P300 window outside epoch range for {electrode}")
            return {"amplitude": np.nan, "latency": np.nan}

        window_data = data[window_mask]
        window_times = times[window_mask]

        peak_idx = np.argmax(window_data)
        amplitude = float(window_data[peak_idx] * 1e6)  # V → µV
        latency = float(window_times[peak_idx] * 1000)  # s → ms

        return {"amplitude": amplitude, "latency": latency}

    def _validate_p300_electrode(
        self, electrode: str, amplitude: float, latency: float, patient_id: str
    ) -> Dict[str, Any]:
        """
        Validate P300 quality for one electrode.

        Args:
            electrode: Electrode name (e.g., 'Pz')
            amplitude: P300 amplitude in µV
            latency: P300 latency in ms
            patient_id: Patient ID for logging

        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - is_positive: bool
            - is_on_time: bool
            - is_expected_latency: bool
            - issues: list of issues found
        """
        validation = {
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
                "likely inverted reference or absent P300"
            )

        min_lat, max_lat = ERP_CONFIG["p300_max_latency_range"]
        if not (min_lat <= latency <= max_lat):
            validation["is_valid"] = False
            validation["is_on_time"] = False
            validation["issues"].append("latency_out_of_range")
            logger.warning(
                f"{patient_id} - {electrode}: Latency {latency:.1f}ms outside acceptable range [{min_lat}-{max_lat}ms]"
            )

        # Atypical latency flags for QC but doesn't invalidate the electrode
        exp_min, exp_max = ERP_CONFIG["p300_expected_latency_range"]
        if not (exp_min <= latency <= exp_max):
            validation["is_expected_latency"] = False
            validation["issues"].append("latency_atypical")
            logger.info(
                f"{patient_id} - {electrode}: Latency {latency:.1f}ms outside "
                f"typical range [{exp_min}-{exp_max}ms] but within acceptable limits"
            )

        return validation

    def _compute_composite_p300(self, erp: mne.Evoked, patient_id: str) -> Dict[str, Any]:
        """
        Compute composite P300 from valid midline electrodes.

        Args:
            erp: MNE Evoked object
            patient_id: Patient ID for logging

        Returns:
            Composite metrics and QC fields
        """
        electrode_data = {}
        valid_amplitudes = []
        valid_latencies = []
        valid_electrodes = []
        flagged_electrodes = []

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
            composite = {
                "composite_amplitude": float(np.mean(valid_amplitudes)),
                "composite_amplitude_std": (float(np.std(valid_amplitudes)) if len(valid_amplitudes) > 1 else 0.0),
                "composite_latency": float(np.mean(valid_latencies)),
                "composite_latency_std": (float(np.std(valid_latencies)) if len(valid_latencies) > 1 else 0.0),
                "n_valid_electrodes": len(valid_electrodes),
                "valid_electrodes": ",".join(valid_electrodes),
                "best_electrode": max(zip(valid_electrodes, valid_amplitudes), key=lambda x: x[1])[0],
                "best_electrode_amplitude": max(valid_amplitudes),
            }

            logger.info(
                f"{patient_id}: Composite P300 = {composite['composite_amplitude']:.2f}µV "
                f"(±{composite['composite_amplitude_std']:.2f}) from {len(valid_electrodes)} electrodes: "
                f"{', '.join(valid_electrodes)}"
            )
        else:
            composite = {
                "composite_amplitude": np.nan,
                "composite_amplitude_std": np.nan,
                "composite_latency": np.nan,
                "composite_latency_std": np.nan,
                "n_valid_electrodes": 0,
                "valid_electrodes": "",
                "best_electrode": None,
                "best_electrode_amplitude": np.nan,
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
                    f"{patient_id} - {elec} FLAGGED: "
                    f"amplitude={data['amplitude']:.2f}µV, latency={data['latency']:.1f}ms, "
                    f"issues={data['issues']}"
                )

        return composite

    def _save_outputs(
        self,
        patient_id: str,
        date: str,
        epochs: mne.Epochs,
        erp: mne.Evoked,
        features: Dict[str, Any],
    ):
        """
        Save ERP and features to disk.

        Args:
            patient_id: Patient identifier
            date: Session date
            epochs: MNE Epochs object (not saved — regenerated on demand from ENG-03)
            erp: MNE Evoked object
            features: Dictionary of P300 features
        """
        erp_file = self._output_paths.erps / f"{patient_id}_{date}_oddball-ave.fif"
        erp.save(erp_file, overwrite=True)
        logger.info(f"Saved ERP: {erp_file}")

        features_df = pd.DataFrame([features])
        self._update_master_feature_table(features_df)

    def _update_master_feature_table(self, incoming_features: pd.DataFrame) -> pd.DataFrame:
        """
        Upsert session features into the master feature table.

        Args:
            incoming_features: DataFrame with one or more session feature rows

        Returns:
            Updated master feature DataFrame
        """
        master_path = self._output_paths.features / "p300_features.parquet"
        if master_path.exists():
            master_df = pd.read_parquet(master_path)
            combined = pd.concat([master_df, incoming_features], ignore_index=True)
        else:
            combined = incoming_features.copy()

        combined = combined.drop_duplicates(subset=["patient_id", "date"], keep="last")
        combined.to_parquet(master_path)
        logger.info(f"Updated master feature table: {master_path} ({len(combined)} rows)")
        return combined

    def _plot_erp_panels(
        self,
        erp: mne.Evoked,
        axes: Any,
        title_top: str,
        electrodes_to_plot: List[str],
        panel_title_bottom: str,
        color_map_or_list: Any,
    ) -> None:
        """
        Draw two-panel ERP plot: Panel 0 butterfly + P300 window, Panel 1 selected electrodes.
        Caller creates fig/axes and handles save/close.
        """
        times = erp.times * 1000  # ms
        data = erp.data * 1e6  # µV
        ch_names_upper = [ch.upper() for ch in erp.ch_names]

        # Panel 0: butterfly
        for ch_idx in range(data.shape[0]):
            axes[0].plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)
        axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        axes[0].axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title(title_top)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 1: selected electrodes
        is_dict = isinstance(color_map_or_list, dict)
        for idx, electrode in enumerate(electrodes_to_plot):
            color = color_map_or_list[electrode] if is_dict else color_map_or_list[idx % len(color_map_or_list)]
            try:
                elec_idx = ch_names_upper.index(electrode.upper())
                axes[1].plot(times, data[elec_idx, :], linewidth=2, color=color, label=electrode)
            except ValueError:
                logger.warning(f"Electrode {electrode} not found in data")
        axes[1].axvline(x=0, color="k", linestyle="--", linewidth=1)
        axes[1].axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        axes[1].axhline(y=0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Amplitude (µV)")
        axes[1].set_title(panel_title_bottom)
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    def _plot_individual_erp(
        self,
        erp: mne.Evoked,
        patient_id: str,
        date: str,
        custom_electrodes: Optional[List[str]] = None,
    ):
        """
        Generate and save individual ERP plot.

        Args:
            erp: MNE Evoked object
            patient_id: Patient identifier
            date: Session date
            custom_electrodes: Optional list of custom electrodes to plot
        """
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
        plt.close()
        logger.info(f"Saved ERP plot: {save_path}")

    def process_all_patients(
        self,
        patient_ids: Optional[List[str]] = None,
        custom_electrodes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Process all patients with oddball data.

        Args:
            patient_ids: List of patient IDs to process. If None, processes all patients
                        with aligned oddball data.
            custom_electrodes: Optional list of custom electrodes to analyze instead of default Pz/Cz/Fz

        Returns:
            DataFrame of P300 features for all patients
        """
        if patient_ids is None:
            patient_ids = self._get_patients_with_oddball()

        logger.info(f"Processing {len(patient_ids)} patients")

        all_features = []

        for patient_id in tqdm(patient_ids, desc="Processing patients"):
            result = self.process_patient(patient_id, custom_electrodes=custom_electrodes)

            if result.get("status") == "success":
                features = result.get("features")
                if isinstance(features, pd.DataFrame):
                    all_features.append(features)
                elif isinstance(features, dict):
                    all_features.append(pd.DataFrame([features]))

        if not all_features:
            logger.warning("No features extracted from any patient")
            return pd.DataFrame()

        features_df = pd.concat(all_features, ignore_index=True)
        return self._update_master_feature_table(features_df)

    def _get_patients_with_oddball(self) -> List[str]:
        """
        Get list of patient IDs with aligned oddball data.

        Returns:
            List of patient IDs
        """
        aligned_files = list(config.ALIGNED_EVENTS_DIR.glob("*_events.parquet"))
        patient_ids = []

        for file_path in aligned_files:
            patient_id = file_path.stem.replace("_events", "")
            try:
                df = pd.read_parquet(file_path)
                has_oddball = (df["trial_type"].str.lower() == "oddball").any()
                if has_oddball:
                    patient_ids.append(patient_id)
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue

        logger.info(f"Found {len(patient_ids)} patients with oddball data")
        return sorted(patient_ids)

    def compute_grand_average(self, patient_ids: Optional[List[str]] = None) -> Optional[mne.Evoked]:
        """
        Compute grand average ERP across multiple patients.

        Args:
            patient_ids: List of patient IDs to include. If None, uses all patients
                        with saved ERPs.

        Returns:
            MNE Evoked object (grand average ERP)
        """
        aggregate_filename = "grand_average_oddball-ave.fif"
        aggregate_path = self._output_paths.erps / aggregate_filename

        if patient_ids is None:
            candidate_files = list(self._output_paths.erps.glob("*_oddball-ave.fif"))
        else:
            candidate_files = []
            for patient_id in patient_ids:
                patient_erps = list(self._output_paths.erps.glob(f"{patient_id}_*_oddball-ave.fif"))
                candidate_files.extend(patient_erps)

        erp_files = []
        excluded_files = []
        for erp_file in candidate_files:
            # Skip the grand-average file itself and any oddly named files
            if erp_file.name == aggregate_filename or erp_file == aggregate_path:
                excluded_files.append(erp_file.name)
                continue
            if erp_file.stem.count("_") < 2:
                excluded_files.append(erp_file.name)
                continue
            erp_files.append(erp_file)

        if excluded_files:
            logger.info(
                "Excluded %d non-session ERP file(s) from grand average: %s",
                len(excluded_files),
                ", ".join(sorted(set(excluded_files))),
            )

        if not erp_files:
            logger.warning("No ERP files found for grand average")
            return None

        logger.info("Computing grand average from %d session ERP file(s)", len(erp_files))

        all_erps = []
        for erp_file in erp_files:
            try:
                evokeds = mne.read_evokeds(erp_file, verbose=False)
                erp = evokeds[0] if isinstance(evokeds, list) else evokeds
                all_erps.append(erp)
            except Exception as e:
                logger.warning(f"Failed to load {erp_file}: {e}")
                continue

        if not all_erps:
            logger.warning("No ERPs successfully loaded")
            return None

        grand_avg = mne.grand_average(all_erps)

        grand_avg_file = self._output_paths.erps / "grand_average_oddball-ave.fif"
        grand_avg.save(grand_avg_file, overwrite=True)
        logger.info(f"Saved grand average ERP: {grand_avg_file}")

        self._plot_grand_average(grand_avg, len(all_erps))

        return grand_avg

    def _plot_grand_average(self, grand_avg: mne.Evoked, n_subjects: int):
        """
        Generate and save grand average ERP plot.

        Args:
            grand_avg: Grand average MNE Evoked object
            n_subjects: Number of subjects included
        """
        save_path = self._output_paths.plots_erp / "grand_average_oddball_erp.png"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        electrode_colors = {"Fz": "red", "Cz": "green", "Pz": "blue"}
        self._plot_erp_panels(
            grand_avg,
            axes,
            title_top=f"Grand Average ERP (N={n_subjects}) - All Channels",
            electrodes_to_plot=["Fz", "Cz", "Pz"],
            panel_title_bottom=f"Grand Average ERP (N={n_subjects}) - Midline Electrodes (Composite Scoring)",
            color_map_or_list=electrode_colors,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved grand average plot: {save_path}")

    def generate_qc_report(self) -> Dict[str, Any]:
        """
        Generate quality control report from processed data.

        Returns:
            Dictionary containing QC metrics
        """
        features_path = self._output_paths.features / "p300_features.parquet"

        if not features_path.exists():
            logger.warning(f"No feature table found: {features_path}")
            return {"status": "no_data"}

        features = pd.read_parquet(features_path)

        if features.empty:
            return {"status": "empty"}

        report = {
            "total_patients": int(features["patient_id"].nunique()),
            "total_sessions": len(features),
            "total_epochs": int(features["n_epochs"].sum()),
            "p300_detection_rate": float(
                (features["p300_amplitude_uV"].notna() & (features["p300_amplitude_uV"] > 2.0)).mean()
            ),
            "mean_amplitude_uV": float(features["p300_amplitude_uV"].mean()),
            "std_amplitude_uV": float(features["p300_amplitude_uV"].std()),
            "mean_latency_ms": float(features["p300_latency_ms"].mean()),
            "std_latency_ms": float(features["p300_latency_ms"].std()),
            "mean_baseline_noise_uV": float(features["baseline_std_uV"].mean()),
            "by_patient": features.groupby("patient_id")
            .agg(
                {
                    "n_epochs": "sum",
                    "p300_amplitude_uV": "mean",
                    "p300_latency_ms": "mean",
                }
            )
            .to_dict("index"),
        }

        if "timezone_warning_flag" in features.columns:
            report["sessions_with_timezone_warning"] = int(features["timezone_warning_flag"].fillna(False).sum())
        if "timezone_confidence" in features.columns:
            report["timezone_confidence_breakdown"] = (
                features["timezone_confidence"].fillna("unknown").value_counts().to_dict()
            )

        report_path = self._output_paths.qc / "erp_qc_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved QC report: {report_path}")
        self._print_qc_summary(report)

        return report

    def _print_qc_summary(self, report: Dict[str, Any]):
        """Print QC report summary to console."""
        print(f"\n{'=' * 60}")
        print("  ERP Pipeline QC Report")
        print(f"{'=' * 60}")
        print(f"  Total Patients:     {report['total_patients']}")
        print(f"  Total Sessions:     {report['total_sessions']}")
        print(f"  Total Epochs:       {report['total_epochs']}")
        print(f"  P300 Detection:     {report['p300_detection_rate']:.1%}")
        print()
        print(f"  Mean P300 Amplitude:  {report['mean_amplitude_uV']:.2f} ± {report['std_amplitude_uV']:.2f} µV")
        print(f"  Mean P300 Latency:    {report['mean_latency_ms']:.1f} ± {report['std_latency_ms']:.1f} ms")
        print(f"  Mean Baseline Noise:  {report['mean_baseline_noise_uV']:.2f} µV")
        print(f"{'=' * 60}\n")
