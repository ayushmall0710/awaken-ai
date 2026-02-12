"""
ERP/Oddball Pipeline Module (ENG-02b)

Extracts epochs from aligned oddball trials, computes ERPs, and quantifies P300 features.
Supports batch processing across patients and generates validation plots and QC reports.
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


# ERP Analysis Configuration
ERP_CONFIG = {
    "tmin": -0.2,  # Epoch start: 200ms before stimulus
    "tmax": 0.7,  # Epoch end: 700ms after stimulus
    "baseline": (None, 0),  # Baseline correction: -200ms to 0ms
    "p300_window": (0.3, 0.6),  # P300 search window: 300-600ms
    "min_epochs": 2,  # Minimum rare events needed per trial
    "midline_electrodes": ["Pz", "Cz", "Fz"],  # Primary electrodes for P300
    # P300 validation thresholds for composite scoring
    "p300_min_amplitude": 0.0,  # Minimum amplitude (µV) - must be positive
    "p300_expected_latency_range": (
        300,
        500,
    ),  # Expected latency range (ms) for controls
    "p300_max_latency_range": (250, 600),  # Maximum acceptable range (ms)
}


class OddballERPPipeline:
    """Pipeline for extracting and analyzing ERPs from oddball trials."""

    def __init__(
        self,
        data_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """
        Initialize ERP pipeline.

        Args:
            data_root: Root directory for data files (defaults to config.LOCAL_DATA_ROOT)
            output_dir: Output directory for processed files (defaults to config.PROCESSED_DATA_DIR)
            verbose: If True, enable detailed logging
        """
        self.data_root = data_root or config.LOCAL_DATA_ROOT
        self.output_dir = output_dir or config.PROCESSED_DATA_DIR
        self.verbose = verbose

        # Initialize data loader
        self.loader = UnifiedDataLoader(data_root=self.data_root, verbose=verbose)

        # Set MNE logging level
        if not verbose:
            mne.set_log_level("WARNING")

        # Ensure output directories exist
        self._create_output_directories()
        self._last_epoch_diagnostics: Dict[str, Any] = {}

    def _create_output_directories(self):
        """Create all necessary output directories."""
        directories = [
            self.output_dir / "epochs",
            self.output_dir / "erps",
            self.output_dir / "features",
            self.output_dir / "plots" / "erp",
            self.output_dir / "qc",
        ]
        for dir_path in directories:
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
            # Load aligned events
            aligned_trials = self._load_aligned_trials(patient_id)

            if aligned_trials.empty:
                logger.warning(f"No aligned trials found for {patient_id}")
                return {"patient_id": patient_id, "status": "no_data"}

            # Filter for specific date if provided
            if date:
                aligned_trials = aligned_trials[aligned_trials["date"] == date]
                if aligned_trials.empty:
                    logger.warning(f"No trials found for {patient_id} on {date}")
                    return {"patient_id": patient_id, "date": date, "status": "no_data"}

            # Process each session separately
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

                if session_result["status"] == "success":
                    all_session_results.append(session_result)

            if not all_session_results:
                return {"patient_id": patient_id, "status": "failed"}

            # Return results (for single session, return that session; for multi-session, aggregate)
            if len(all_session_results) == 1:
                return all_session_results[0]
            else:
                # Aggregate features from all sessions
                all_features = pd.concat([r["features"] for r in all_session_results], ignore_index=True)
                return {
                    "patient_id": patient_id,
                    "status": "success",
                    "sessions": len(all_session_results),
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
            # Load EDF for this session
            raw = self.loader.load_edf(patient_id, date=date, use_clipped=True)

            # Extract rare events from oddball trials
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

            # Create epochs
            epochs = self._create_epochs(raw, rare_events)

            if len(epochs) < ERP_CONFIG["min_epochs"]:
                logger.warning(f"Insufficient epochs after creation for {patient_id} on {date}: {len(epochs)}")
                return {
                    "patient_id": patient_id,
                    "date": date,
                    "status": "insufficient_epochs",
                }

            # Compute ERP (average across epochs)
            erp = self._compute_erp(epochs)

            # Quantify P300 features
            features = self._quantify_p300(
                erp,
                patient_id,
                date,
                len(epochs),
                custom_electrodes=custom_electrodes,
            )

            # Save outputs
            self._save_outputs(patient_id, date, epochs, erp, features)

            # Generate plots
            self._plot_individual_erp(erp, patient_id, date, custom_electrodes=custom_electrodes)

            # Log summary
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

        # Filter for oddball trials
        oddball_trials = df[df["trial_type"].str.lower() == "oddball"].copy()

        logger.info(f"Loaded {len(oddball_trials)} oddball trials for {patient_id}")
        return oddball_trials

    def _detect_timezone_offset(self, raw: mne.io.Raw, rare_events: List[Dict[str, Any]]) -> float:
        """
        Detect timezone offset between EDF and Unix timestamps.

        Matches the ENG-02 timestamp aligner logic.

        Args:
            raw: MNE Raw object
            rare_events: List of rare events with Unix timestamps

        Returns:
            Timezone offset in seconds
        """
        meas_date = raw.info.get("meas_date")
        if meas_date is None or not rare_events:
            return 0.0

        edf_start_unix = meas_date.timestamp()
        # Use earliest event to mirror ENG-02 behavior.
        first_event_unix = min(event["timestamp_unix"] for event in rare_events)

        diff = abs(first_event_unix - edf_start_unix)

        # If difference > 30 minutes, apply hour-based correction
        if diff > 1800:
            correction = (diff // 1800) * 1800
            if first_event_unix > edf_start_unix:
                correction = -correction
            logger.info(f"Timezone offset detected: {correction / 3600:.1f} hours")
            return correction

        return 0.0

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
            # Use bracket notation like timestamp_aligner does
            try:
                events = trial["sentences"]
            except (KeyError, TypeError):
                logger.warning(f"Trial {idx}: No 'sentences' field found")
                continue

            # Convert numpy array to list if necessary (pandas reads parquet nested data as numpy arrays)
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

                # Check if this is a rare event with aligned timestamp
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

    def _create_epochs(self, raw: mne.io.Raw, rare_events: List[Dict[str, Any]]) -> mne.Epochs:
        """
        Create MNE Epochs around rare beep events.

        Args:
            raw: MNE Raw object (continuous EEG data)
            rare_events: List of rare event dictionaries with Unix timestamps

        Returns:
            MNE Epochs object
        """
        # Get EDF metadata
        edf_start_unix = raw.info["meas_date"].timestamp()
        sfreq = raw.info["sfreq"]
        recording_duration = raw.times[-1]  # Duration of recording in seconds
        max_sample = int(recording_duration * sfreq)

        # Detect timezone offset (same logic as timestamp_aligner)
        timezone_offset = self._detect_timezone_offset(raw, rare_events)
        logger.debug(f"Timezone offset: {timezone_offset / 3600:.1f} hours")

        diagnostics = {
            "n_rare_events": len(rare_events),
            "timezone_offset_seconds": float(timezone_offset),
            "timezone_offset_hours": float(timezone_offset / 3600),
            "n_out_of_recording": 0,
            "n_too_close_to_start": 0,
            "n_too_close_to_end": 0,
            "n_valid_events_pre_mne": 0,
            "n_epochs_post_mne": 0,
            "n_dropped_by_mne": 0,
            "timezone_confidence": "high",
            "timezone_warning_flag": False,
            "diagnostic_note": "",
        }

        # Convert Unix timestamps to EDF-relative time (with timezone correction)
        valid_events = []
        for event in rare_events:
            # Apply timezone offset correction (inverse of _edf_to_unix)
            event["edf_time"] = (event["timestamp_unix"] - edf_start_unix) + timezone_offset
            event["sample_idx"] = int(event["edf_time"] * sfreq)

            # Validate within recording bounds first.
            if event["sample_idx"] < 0 or event["sample_idx"] >= max_sample:
                diagnostics["n_out_of_recording"] += 1
                logger.warning(
                    f"Event at Unix {event['timestamp_unix']} → EDF time {event['edf_time']:.2f}s → "
                    f"sample {event['sample_idx']} is outside recording range [0, {max_sample}]. Skipping."
                )
                continue

            # Validate full epoch window bounds to avoid silent post-hoc drops.
            if event["edf_time"] + ERP_CONFIG["tmin"] < 0:
                diagnostics["n_too_close_to_start"] += 1
                logger.warning(
                    f"Event at Unix {event['timestamp_unix']} is too close to recording start for "
                    f"epoch window [{ERP_CONFIG['tmin']}, {ERP_CONFIG['tmax']}]. Skipping."
                )
                continue
            if event["edf_time"] + ERP_CONFIG["tmax"] > recording_duration:
                diagnostics["n_too_close_to_end"] += 1
                logger.warning(
                    f"Event at Unix {event['timestamp_unix']} is too close to recording end for "
                    f"epoch window [{ERP_CONFIG['tmin']}, {ERP_CONFIG['tmax']}]. Skipping."
                )
                continue

            valid_events.append(event)
            logger.debug(
                f"Event: Unix {event['timestamp_unix']} → EDF {event['edf_time']:.2f}s → sample {event['sample_idx']}"
            )

        diagnostics["n_valid_events_pre_mne"] = len(valid_events)
        valid_ratio = (len(valid_events) / len(rare_events)) if rare_events else 0.0
        if abs(timezone_offset) >= 12 * 3600 or valid_ratio < 0.8:
            diagnostics["timezone_confidence"] = "low"
            diagnostics["timezone_warning_flag"] = True
            diagnostics["diagnostic_note"] = (
                f"Alignment warning: offset={timezone_offset / 3600:.1f}h, "
                f"valid_events={len(valid_events)}/{len(rare_events)}."
            )
            logger.warning(diagnostics["diagnostic_note"])
        else:
            diagnostics["diagnostic_note"] = (
                f"Offset {timezone_offset / 3600:.1f}h; valid_events={len(valid_events)}/{len(rare_events)}."
            )

        if len(valid_events) == 0:
            logger.error("No valid events within recording range after timestamp conversion")
            self._last_epoch_diagnostics = diagnostics
            return mne.Epochs(
                raw,
                np.array([]).reshape(0, 3),
                event_id={"rare": 1},
                tmin=ERP_CONFIG["tmin"],
                tmax=ERP_CONFIG["tmax"],
                baseline=ERP_CONFIG["baseline"],
                preload=True,
                reject=None,
                verbose=False,
            )

        logger.info(
            "Valid events after filtering: %d/%d (out_of_recording=%d, too_close_to_start=%d, too_close_to_end=%d)",
            len(valid_events),
            len(rare_events),
            diagnostics["n_out_of_recording"],
            diagnostics["n_too_close_to_start"],
            diagnostics["n_too_close_to_end"],
        )

        # Create MNE events array: [sample_index, 0, event_id]
        mne_events = np.array([[e["sample_idx"], 0, 1] for e in valid_events])

        # Select EEG channels (exclude DC channels)
        picks = mne.pick_types(raw.info, eeg=True, exclude=["DC1", "DC2", "DC", "DC1", "AUX"])

        # Create epochs
        epochs = mne.Epochs(
            raw,
            mne_events,
            event_id={"rare": 1},
            tmin=ERP_CONFIG["tmin"],
            tmax=ERP_CONFIG["tmax"],
            baseline=ERP_CONFIG["baseline"],
            picks=picks,
            preload=True,
            proj=False,  # Don't apply projections yet (will be done in ENG-03)
            reject=None,  # No artifact rejection (will be done in ENG-03)
            verbose=self.verbose,
        )

        diagnostics["n_epochs_post_mne"] = len(epochs)
        diagnostics["n_dropped_by_mne"] = len(valid_events) - len(epochs)
        if diagnostics["n_dropped_by_mne"] > 0:
            logger.warning(
                "MNE dropped %d event(s) after pre-filtering",
                diagnostics["n_dropped_by_mne"],
            )
        self._last_epoch_diagnostics = diagnostics

        logger.info(f"Created {len(epochs)} epochs (shape: {epochs.get_data().shape})")
        return epochs

    def _compute_erp(self, epochs: mne.Epochs) -> mne.Evoked:
        """
        Compute ERP by averaging epochs.

        Args:
            epochs: MNE Epochs object

        Returns:
            MNE Evoked object (averaged ERP)
        """
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

        # Summarize issues
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
        # Initialize base features
        features = {
            "patient_id": patient_id,
            "date": date,
            "n_epochs": n_epochs,
            "processing_timestamp": datetime.now().isoformat(),
        }

        # Compute baseline noise level
        baseline_mask = (erp.times >= ERP_CONFIG["tmin"]) & (erp.times <= 0)
        baseline_data = erp.data[:, baseline_mask]
        features["baseline_std_uV"] = float(np.std(baseline_data) * 1e6)

        # Determine which electrodes to analyze
        if custom_electrodes:
            # Analyze only requested electrodes.
            logger.info(f"{patient_id}: Using custom electrodes: {custom_electrodes}")

            for electrode in custom_electrodes:
                p300_features = self._detect_p300_peak(erp, electrode)
                features[f"p300_amplitude_{electrode}_uV"] = p300_features["amplitude"]
                features[f"p300_latency_{electrode}_ms"] = p300_features["latency"]

            # Skip composite score for custom electrode mode.
            features["qc_notes"] = f"Custom electrode analysis: {','.join(custom_electrodes)}"

        else:
            # Default mode: analyze Pz/Cz/Fz and compute a composite score.
            for electrode in ERP_CONFIG["midline_electrodes"]:
                p300_features = self._detect_p300_peak(erp, electrode)
                features[f"p300_amplitude_{electrode}_uV"] = p300_features["amplitude"]
                features[f"p300_latency_{electrode}_ms"] = p300_features["latency"]

            # Compute composite P300 and QC fields.
            composite = self._compute_composite_p300(erp, patient_id)

            # Add composite fields.
            features.update(
                {
                    # Composite metrics
                    "p300_composite_amplitude_uV": composite["composite_amplitude"],
                    "p300_composite_latency_ms": composite["composite_latency"],
                    # Selected electrode
                    "p300_best_electrode": composite["best_electrode"],
                    # Reliability counts
                    "p300_n_valid_electrodes": composite["n_valid_electrodes"],
                    "p300_n_flagged_electrodes": composite["n_flagged_electrodes"],
                }
            )

            # Generate QC notes.
            features["qc_notes"] = self._generate_qc_notes(composite)

            # Compatibility fields expected by downstream code.
            features["p300_amplitude_uV"] = composite["composite_amplitude"]
            features["p300_latency_ms"] = composite["composite_latency"]

            # Log flagged electrodes.
            if composite["n_flagged_electrodes"] > 0:
                logger.warning(
                    f"{patient_id} QC warning: {composite['n_flagged_electrodes']} electrode(s) flagged - "
                    f"{features['qc_notes']}"
                )

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
        # Check if electrode exists (case-insensitive)
        electrode_names = [ch.upper() for ch in erp.ch_names]
        electrode_upper = electrode.upper()

        if electrode_upper not in electrode_names:
            # Show first 10 electrodes to help selection.
            available = ", ".join(erp.ch_names[:10])
            if len(erp.ch_names) > 10:
                available += f" ... ({len(erp.ch_names)} total)"
            logger.warning(
                f"Electrode {electrode} not found. Available: {available}. Use --list-electrodes to see all electrodes."
            )
            return {"amplitude": np.nan, "latency": np.nan}

        # Get electrode index (case-insensitive match)
        ch_idx = electrode_names.index(electrode_upper)
        data = erp.data[ch_idx, :]
        times = erp.times

        # Define P300 search window
        window_start, window_end = ERP_CONFIG["p300_window"]
        window_mask = (times >= window_start) & (times <= window_end)

        if not window_mask.any():
            logger.warning(f"P300 window outside epoch range for {electrode}")
            return {"amplitude": np.nan, "latency": np.nan}

        # Find peak (maximum positive deflection in window)
        window_data = data[window_mask]
        window_times = times[window_mask]

        peak_idx = np.argmax(window_data)
        amplitude = float(window_data[peak_idx] * 1e6)  # Convert V to µV
        latency = float(window_times[peak_idx] * 1000)  # Convert s to ms

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

        # Check for NaN
        if np.isnan(amplitude) or np.isnan(latency):
            validation["is_valid"] = False
            validation["issues"].append("missing_data")
            return validation

        # Check polarity (must be positive)
        if amplitude <= ERP_CONFIG["p300_min_amplitude"]:
            validation["is_valid"] = False
            validation["is_positive"] = False
            validation["issues"].append("negative_or_zero_amplitude")
            logger.warning(
                f"{patient_id} - {electrode}: Negative/zero amplitude ({amplitude:.2f}µV) - "
                "likely inverted reference or absent P300"
            )

        # Check latency range (max acceptable)
        min_lat, max_lat = ERP_CONFIG["p300_max_latency_range"]
        if not (min_lat <= latency <= max_lat):
            validation["is_valid"] = False
            validation["is_on_time"] = False
            validation["issues"].append("latency_out_of_range")
            logger.warning(
                f"{patient_id} - {electrode}: Latency {latency:.1f}ms outside acceptable range [{min_lat}-{max_lat}ms]"
            )

        # Check expected latency (for QC, doesn't invalidate)
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

        # Extract and validate each electrode.
        for electrode in ERP_CONFIG["midline_electrodes"]:
            p300 = self._detect_p300_peak(erp, electrode)
            amplitude = p300["amplitude"]
            latency = p300["latency"]

            # Validate quality.
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

            # Collect valid measurements.
            if validation["is_valid"]:
                valid_amplitudes.append(amplitude)
                valid_latencies.append(latency)
                valid_electrodes.append(electrode)
            else:
                flagged_electrodes.append(electrode)

        # Compute composite scores.
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

        # Add QC flags.
        composite["n_flagged_electrodes"] = len(flagged_electrodes)
        composite["flagged_electrodes"] = ",".join(flagged_electrodes) if flagged_electrodes else ""

        # Add per-electrode validation details.
        for electrode in ERP_CONFIG["midline_electrodes"]:
            data = electrode_data[electrode]
            composite[f"{electrode}_is_valid"] = data["is_valid"]
            composite[f"{electrode}_is_positive"] = data["is_positive"]
            composite[f"{electrode}_issues"] = ",".join(data["issues"]) if data["issues"] else ""

        # Log flagged electrodes.
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
        Save epochs, ERP, and features to disk.

        Args:
            patient_id: Patient identifier
            date: Session date
            epochs: MNE Epochs object
            erp: MNE Evoked object
            features: Dictionary of P300 features
        """
        # Save epochs
        epochs_file = self.output_dir / "epochs" / f"{patient_id}_{date}_oddball-epo.fif"
        epochs.save(epochs_file, overwrite=True)
        logger.info(f"Saved epochs: {epochs_file}")

        # Save ERP
        erp_file = self.output_dir / "erps" / f"{patient_id}_{date}_oddball-ave.fif"
        erp.save(erp_file, overwrite=True)
        logger.info(f"Saved ERP: {erp_file}")

        # Save features
        features_file = self.output_dir / "features" / f"{patient_id}_{date}_p300_features.parquet"
        features_df = pd.DataFrame([features])
        features_df.to_parquet(features_file)
        logger.info(f"Saved features: {features_file}")
        self._update_master_feature_table(features_df)

    def _update_master_feature_table(self, incoming_features: pd.DataFrame) -> pd.DataFrame:
        """
        Upsert session features into the master feature table.

        Args:
            incoming_features: DataFrame with one or more session feature rows

        Returns:
            Updated master feature DataFrame
        """
        master_path = self.output_dir / "features" / "p300_features.parquet"
        if master_path.exists():
            master_df = pd.read_parquet(master_path)
            combined = pd.concat([master_df, incoming_features], ignore_index=True)
        else:
            combined = incoming_features.copy()

        # Keep latest row per patient/date.
        combined = combined.drop_duplicates(subset=["patient_id", "date"], keep="last")
        combined.to_parquet(master_path)
        logger.info(f"Updated master feature table: {master_path} ({len(combined)} rows)")
        return combined

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
        save_path = self.output_dir / "plots" / "erp" / f"{patient_id}_{date}_oddball_erp.png"

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Panel 1: Butterfly plot (all channels)
        times = erp.times * 1000  # Convert to ms
        data = erp.data * 1e6  # Convert to µV

        for ch_idx in range(data.shape[0]):
            axes[0].plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)

        axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        axes[0].axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title(f"{patient_id} - {date} - All Channels")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: custom electrodes or default midline electrodes
        electrode_names_upper = [ch.upper() for ch in erp.ch_names]

        # Determine which electrodes to plot
        if custom_electrodes:
            # Custom electrode mode.
            electrodes_to_plot = custom_electrodes
            panel_title = f"{patient_id} - {date} - Custom Electrodes: {', '.join(custom_electrodes)}"
            # Assign colors dynamically for custom electrodes.
            color_palette = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
            ]
        else:
            # Default midline electrodes.
            electrodes_to_plot = ["Fz", "Cz", "Pz"]
            panel_title = f"{patient_id} - {date} - Midline Electrodes (Composite Scoring)"
            color_palette = ["red", "green", "blue"]

        found_any = False
        for idx, electrode in enumerate(electrodes_to_plot):
            color = color_palette[idx % len(color_palette)]
            try:
                elec_idx = electrode_names_upper.index(electrode.upper())
                elec_data = data[elec_idx, :]
                axes[1].plot(times, elec_data, linewidth=2, color=color, label=electrode)
                found_any = True
            except ValueError:
                # Electrode not found; skip it.
                logger.warning(f"Electrode {electrode} not found in data")
                pass

        if found_any:
            axes[1].axvline(x=0, color="k", linestyle="--", linewidth=1)
            axes[1].axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
            axes[1].axhline(y=0, color="gray", linestyle=":", linewidth=1)
            axes[1].set_xlabel("Time (ms)")
            axes[1].set_ylabel("Amplitude (µV)")
            axes[1].set_title(panel_title)
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(
                0.5,
                0.5,
                f"Electrodes {', '.join(electrodes_to_plot)} not available",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
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

        # Combine all features
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

            # Check if this patient has oddball trials
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

    def compute_grand_average(self, patient_ids: Optional[List[str]] = None) -> mne.Evoked:
        """
        Compute grand average ERP across multiple patients.

        Args:
            patient_ids: List of patient IDs to include. If None, uses all patients
                        with saved ERPs.

        Returns:
            MNE Evoked object (grand average ERP)
        """
        aggregate_filename = "grand_average_oddball-ave.fif"
        aggregate_path = self.output_dir / "erps" / aggregate_filename

        if patient_ids is None:
            # Include only session ERP files.
            candidate_files = list((self.output_dir / "erps").glob("*_oddball-ave.fif"))
        else:
            candidate_files = []
            for patient_id in patient_ids:
                patient_erps = list((self.output_dir / "erps").glob(f"{patient_id}_*_oddball-ave.fif"))
                candidate_files.extend(patient_erps)

        erp_files = []
        excluded_files = []
        for erp_file in candidate_files:
            # Require patient_date_oddball-ave.fif naming shape.
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

        # Load all ERPs
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

        # Compute grand average
        grand_avg = mne.grand_average(all_erps)

        # Save grand average
        grand_avg_file = self.output_dir / "erps" / "grand_average_oddball-ave.fif"
        grand_avg.save(grand_avg_file, overwrite=True)
        logger.info(f"Saved grand average ERP: {grand_avg_file}")

        # Plot grand average
        self._plot_grand_average(grand_avg, len(all_erps))

        return grand_avg

    def _plot_grand_average(self, grand_avg: mne.Evoked, n_subjects: int):
        """
        Generate and save grand average ERP plot.

        Args:
            grand_avg: Grand average MNE Evoked object
            n_subjects: Number of subjects included
        """
        save_path = self.output_dir / "plots" / "erp" / "grand_average_oddball_erp.png"

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        times = grand_avg.times * 1000  # Convert to ms
        data = grand_avg.data * 1e6  # Convert to µV

        # Panel 1: Butterfly plot (all channels)
        for ch_idx in range(data.shape[0]):
            axes[0].plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)

        axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1.5, label="Stimulus")
        axes[0].axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        axes[0].axhline(y=0, color="gray", linestyle=":", linewidth=1)
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title(f"Grand Average ERP (N={n_subjects}) - All Channels")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: midline electrodes
        electrode_colors = {"Fz": "red", "Cz": "green", "Pz": "blue"}
        electrode_labels = {
            "Fz": "Fz (frontal)",
            "Cz": "Cz (central)",
            "Pz": "Pz (parietal)",
        }

        for electrode, color in electrode_colors.items():
            try:
                ch_idx = [ch.upper() for ch in grand_avg.ch_names].index(electrode.upper())
                electrode_data = data[ch_idx, :]
                axes[1].plot(
                    times,
                    electrode_data,
                    linewidth=2,
                    color=color,
                    label=electrode_labels[electrode],
                )
            except ValueError:
                continue

        axes[1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)
        axes[1].axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        axes[1].axhline(y=0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Amplitude (µV)")
        axes[1].set_title(f"Grand Average ERP (N={n_subjects}) - Midline Electrodes (Composite Scoring)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

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
        features_path = self.output_dir / "features" / "p300_features.parquet"

        if not features_path.exists():
            logger.warning(f"No feature table found: {features_path}")
            return {"status": "no_data"}

        features = pd.read_parquet(features_path)

        if features.empty:
            return {"status": "empty"}

        # Aggregate statistics
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

        # Save report
        report_path = self.output_dir / "qc" / "erp_qc_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved QC report: {report_path}")

        # Print summary
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
