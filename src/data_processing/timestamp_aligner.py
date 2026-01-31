"""
Timestamp Alignment Module

Aligns stimulus events with EDF recordings using DC channel analysis.
Uses cross-correlation for language/command trials, peak detection for oddball/beep.
"""

from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import logging

import pandas as pd
import numpy as np
import mne

from src.data_loading import config
from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.utils import signal_processing as utils

logger = logging.getLogger(__name__)


ALIGNMENT_CONFIG = {
    "search_buffer_sec": 5.0,
}

TRIAL_TYPE_TO_METHOD = {
    "language": "correlation",
    "left_command": "correlation",
    "right_command": "correlation",
    "oddball": "peak_detection",
    "beep": "peak_detection",
    "control": "peak_detection",
    "loved_one_voice": "peak_detection",  # TODO: Confirm method (placeholder)
}


class TimestampAligner:
    """Aligns stimulus events with EDF recordings using DC channel."""

    def __init__(
        self,
        patient_id: Union[str, List[str]],
        data_root: Optional[Union[str, Path]] = None,
        dc_channel: Optional[str] = None,
        use_clipped: bool = True,
        verbose: bool = False,
    ):
        self.patient_ids = [patient_id] if isinstance(patient_id, str) else patient_id
        self.dc_channel_override = dc_channel
        self.use_clipped = use_clipped
        self.verbose = verbose

        self.loader = UnifiedDataLoader(data_root=data_root, verbose=verbose)

        if not verbose:
            mne.set_log_level("WARNING")

    def align(
        self,
        save: bool = True,
        output_dir: Path = config.ALIGNED_EVENTS_DIR,
    ) -> Dict[str, pd.DataFrame]:
        """Align all patients and return dict of patient_id -> events_df."""
        results = {}
        for patient_id in self.patient_ids:
            try:
                self.patient_id = patient_id  # Set current patient ID
                logger.info(f"Aligning patient: {self.patient_id}")
                events_df = self._align_patient()

                if save and len(events_df) > 0:
                    self._save(events_df, output_dir)

                results[self.patient_id] = events_df
                logger.info(f"{self.patient_id}: {len(events_df)} events aligned")
            except Exception as e:
                logger.error(f"Failed to align patient {patient_id}: {e}")
                results[patient_id] = pd.DataFrame()

        return results

    def _align_patient(self) -> pd.DataFrame:
        """Align all trials for single patient (handles multiple sessions)."""
        patient = self.loader.get_patient(self.patient_id)
        sessions = patient.list_sessions()

        all_events = []

        for date in sessions:
            logger.info(f"Processing {date} session for patient {self.patient_id}")

            # Get trials for this session
            session_trials = patient.trials_df[patient.trials_df["date"] == date].copy()
            if session_trials.empty:
                continue

            # Load EDF for this session
            try:
                # use use_clipped preference from init, using UnifiedDataLoader's cache
                raw = self.loader.load_edf(
                    self.patient_id, date=date, use_clipped=self.use_clipped
                )
            except Exception as e:
                logger.warning(f"Skipping session {date}: Could not load EDF ({e})")
                continue

            # Align this session
            session_events = self._align_session(raw, session_trials)
            all_events.append(session_events)

        if not all_events:
            return pd.DataFrame()

        return pd.concat(all_events, ignore_index=True)

    def _align_session(self, raw: mne.io.Raw, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Align trials for a single recording session."""
        # Set session-scoped instance attributes (used across alignment methods)
        self.raw = raw
        self.dc_channel = self.dc_channel_override or self._detect_dc_channel(raw)
        self.dc_signal = raw.get_data(picks=[self.dc_channel])[0]
        self.sr = raw.info["sfreq"]

        # Store EDF start time for unix-to-edf conversion
        self.edf_start_unix = raw.info["meas_date"].timestamp()

        # Detect timezone offset
        self.timezone_offset = self._detect_timezone_offset(raw, trials_df)

        events = []
        for _, trial in trials_df.iterrows():
            trial_type = trial["trial_type"].lower()
            method = TRIAL_TYPE_TO_METHOD.get(trial_type, "peak_detection")

            match method:
                case "correlation":
                    df = self._align_correlation(trial)
                case "peak_detection":
                    df = self._align_peaks(trial)
                case _:
                    raise ValueError(f"Unknown method: {method}")

            events.append(df)

        non_empty = [e for e in events if len(e) > 0]
        return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

    def _detect_dc_channel(self, raw: mne.io.Raw) -> str:
        """Auto-detect DC channel from EDF using signal_processing utility."""
        return utils.select_best_dc_channel(raw)

    def _detect_timezone_offset(
        self, raw: mne.io.Raw, stimulus_df: pd.DataFrame
    ) -> float:
        """Detect timezone offset by comparing EDF start time with first trial."""
        meas_date = raw.info.get("meas_date")
        if meas_date is None or stimulus_df.empty:
            return 0.0

        edf_start_unix = meas_date.timestamp()
        first_trial_start = stimulus_df["start_time"].min()

        diff = abs(first_trial_start - edf_start_unix)

        if diff > 1800:  # 30 mins (to handle fractional timezones like IST)
            correction = (diff // 1800) * 1800
            if first_trial_start > edf_start_unix:
                correction = -correction
            logger.info(f"Timezone offset detected: {correction / 3600:.1f} hours")
            return correction

        return 0.0

    def _edf_to_unix(self, edf_time: float) -> float:
        """Convert EDF-relative time to Unix timestamp."""
        return edf_time - self.timezone_offset + self.edf_start_unix

    def _unix_to_edf(self, unix_time: float) -> float:
        """Convert Unix timestamp to EDF-relative time."""
        return (unix_time - self.edf_start_unix) + self.timezone_offset

    def _align_correlation(self, trial: pd.Series) -> pd.DataFrame:
        """Align using cross-correlation with source audio (aligns each event individually)."""
        events = trial.get("sentences", [])
        if len(events) == 0:
            return pd.DataFrame()

        # Convert trial unix times to EDF-relative times
        trial_start_edf = self._unix_to_edf(trial["start_time"])
        trial_end_edf = self._unix_to_edf(trial["end_time"])
        buffer = ALIGNMENT_CONFIG["search_buffer_sec"]

        # Search window for this trial
        search_start = max(0, trial_start_edf - buffer)
        search_end = min(self.raw.times[-1], trial_end_edf + buffer)

        start_idx, end_idx = int(search_start * self.sr), int(search_end * self.sr)

        if start_idx >= len(self.dc_signal) or end_idx <= start_idx:
            return pd.DataFrame()

        # Align each event individually and collect metadata
        enriched_events = []
        current_search_start = search_start

        for event in events:
            event_id = str(event["event"])
            audio_path = self.loader.get_stimulus_audio_path(trial, event_id=event_id)

            if audio_path is None or not audio_path.exists():
                # Keep original event without enrichment
                enriched_events.append(event.copy())
                continue

            src_fs, src_data = self.loader.load_stimulus_audio(audio_path)
            audio_duration = len(src_data) / src_fs

            # Search from current position forward
            event_search_end = min(search_end, current_search_start + buffer * 2)
            event_start_idx = int(current_search_start * self.sr)
            event_end_idx = int(event_search_end * self.sr)

            if (
                event_start_idx >= len(self.dc_signal)
                or event_end_idx <= event_start_idx
            ):
                enriched_events.append(event.copy())
                continue

            chunk = utils.resample_signal(
                self.dc_signal[event_start_idx:event_end_idx], int(self.sr), src_fs
            )

            # Compute audio envelope for correlation
            src_envelope = utils.audio_envelope(src_data, sample_rate=src_fs)
            lag, score = utils.cross_correlate(chunk, src_envelope)

            # Convert EDF time back to unix time
            event_start_edf = current_search_start + (lag / src_fs)
            event_start_unix = self._edf_to_unix(event_start_edf)

            # Enrich event with alignment metadata
            enriched_event = event.copy()
            enriched_event.update(
                {
                    "event_start": event_start_unix,
                    "event_end": event_start_unix + audio_duration,
                    "event_duration": audio_duration,
                    "correlation_score": float(score),
                }
            )
            enriched_events.append(enriched_event)

            # Update search start for next event
            current_search_start += (lag / src_fs) + audio_duration

        return self._build_trial_result(trial, enriched_events, "correlation")

    def _align_peaks(self, trial: pd.Series) -> pd.DataFrame:
        """
        Align using peak detection on trial's time window.

        Approach:
        1. First detect instruction audio (if prompt file exists) and mask it out
        2. Use top-N peaks by prominence (N = event count) to filter ghost peaks
        3. Sort by time and match to events
        """
        trial_type = trial["trial_type"].lower()

        # Convert trial unix times to EDF-relative times
        t_start = self._unix_to_edf(trial["start_time"])
        t_end = self._unix_to_edf(trial["end_time"])

        # Extract DC signal chunk for this trial's time window
        start_idx = int(t_start * self.sr)
        end_idx = int(t_end * self.sr)

        if start_idx >= len(self.dc_signal) or end_idx <= start_idx:
            return pd.DataFrame()

        dc_chunk = self.dc_signal[start_idx:end_idx]

        # Step 1: Find instruction end using correlation
        search_start_idx = self._detect_instruction_end(dc_chunk, trial_type)

        # Step 2: Get valid chunk after instruction
        valid_chunk = dc_chunk[search_start_idx:]

        # Step 3: Apply highpass filter to remove low-frequency noise
        filtered_chunk = utils.highpass_filter(valid_chunk, sfreq=self.sr, cutoff_hz=50)

        # Step 4: Detect peaks in envelope
        events = trial.get("sentences", [])
        if len(events) == 0:
            return pd.DataFrame()

        peaks, widths = self._detect_envelope_peaks(
            filtered_chunk, num_events=len(events)
        )

        if len(peaks) == 0:
            return pd.DataFrame()

        # Adjust peak indices to full chunk coordinates
        peaks = peaks + search_start_idx

        # Step 5: Convert to timestamps and enrich events
        peak_times_unix = self._edf_to_unix(t_start + (peaks / self.sr))
        peak_amplitudes = dc_chunk[peaks]
        peak_durations = widths / self.sr

        enriched_events = []
        for idx, event in enumerate(events):
            enriched_event = event.copy()
            if idx < len(peak_times_unix):
                event_start = float(peak_times_unix[idx])
                beep_duration = (
                    float(peak_durations[idx])
                    if not np.isnan(peak_durations[idx])
                    else None
                )

                enriched_event.update(
                    {
                        "event_start": event_start,
                        "event_end": (
                            event_start + beep_duration
                            if beep_duration
                            else event_start
                        ),
                        "event_duration": beep_duration,
                        "peak_amplitude": float(peak_amplitudes[idx]),
                    }
                )
            enriched_events.append(enriched_event)

        return self._build_trial_result(trial, enriched_events, "peak_detection")

    def _detect_instruction_end(self, dc_chunk: np.ndarray, trial_type: str) -> int:
        """Find where instruction audio ends using cross-correlation."""
        prompt_map = {"oddball": "oddballprompt.wav"}
        prompt_file = prompt_map.get(trial_type)

        if not prompt_file:
            return 0

        prompt_path = config.PROMPTS_DIR / prompt_file
        try:
            src_fs, src_data = self.loader.load_stimulus_audio(prompt_path)
            src_envelope = utils.audio_envelope(src_data, sample_rate=src_fs)

            dc_resampled = utils.resample_signal(dc_chunk, int(self.sr), src_fs)
            lag, score = utils.cross_correlate(dc_resampled, src_envelope)

            if score >= 0.7:
                instruction_duration_samples = int(len(src_data) / src_fs * self.sr)
                instruction_start_samples = int(lag / src_fs * self.sr)
                search_start_idx = (
                    instruction_start_samples + instruction_duration_samples
                )
                logger.info(
                    f"Instruction detected (score={score:.2f}), masking first {search_start_idx/self.sr:.1f}s"
                )
                return search_start_idx
            else:
                logger.warning(
                    f"Instruction not found (score={score:.2f}). Using full window."
                )
        except Exception as e:
            logger.warning(f"Failed to detect instruction: {e}")

        return 0

    def _detect_envelope_peaks(
        self, signal_data: np.ndarray, num_events: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks in envelope and return top-N by prominence."""
        envelope = utils.audio_envelope(signal_data, sample_rate=self.sr, smooth_ms=30)

        peaks, properties = utils.detect_peaks(
            envelope,
            sfreq=self.sr,
            prominence=np.std(envelope) * 0.5,
            min_distance_sec=0.3,
            normalize=False,
        )

        if len(peaks) == 0:
            return np.array([]), np.array([])

        prominences = properties.get("prominences", np.ones(len(peaks)))
        widths = properties.get("widths", np.full(len(peaks), np.nan))

        if len(peaks) > num_events:
            top_indices = np.argsort(prominences)[::-1][:num_events]
            peaks = peaks[top_indices]
            widths = widths[top_indices]
            logger.info(
                f"Envelope: Filtered {len(prominences)} peaks to top {num_events}"
            )
        elif len(peaks) < num_events:
            raise ValueError(
                f"Insufficient peaks: Found {len(peaks)} for {num_events} events."
            )

        # Sort by time
        sort_order = np.argsort(peaks)
        return peaks[sort_order], widths[sort_order]

    def _build_trial_result(
        self,
        trial: pd.Series,
        enriched_events: list,
        method: str,
    ) -> pd.DataFrame:
        """Build single trial row with enriched sentences list."""
        return pd.DataFrame(
            [
                {
                    "patient_id": self.patient_id,
                    "date": trial["date"],
                    "trial_type": trial["trial_type"],
                    "start_time": trial["start_time"],
                    "end_time": trial["end_time"],
                    "duration": trial["duration"],
                    "sentences": enriched_events,
                    "dc_channel": self.dc_channel,
                    "alignment_method": method,
                    "source_file": trial.get("source_file", None),
                }
            ]
        )

    def _save(self, df: pd.DataFrame, output_dir: Path):
        """Save aligned events to parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.patient_id}_events.parquet"
        df.to_parquet(path)
        logger.info(f"Saved: {path}")

    @classmethod
    def validate(
        cls,
        patient_id: str,
        output_dir: Path = config.ALIGNED_EVENTS_DIR,
    ) -> Dict[str, any]:
        """Validate saved alignment results for a patient."""
        path = output_dir / f"{patient_id}_events.parquet"

        if not path.exists():
            return {"status": "error", "message": f"File not found: {path}"}

        df = pd.read_parquet(path)

        if df.empty:
            return {"status": "error", "message": "Empty DataFrame"}

        events_df = df.explode("sentences")
        s_list = events_df["sentences"].tolist()
        event_meta = pd.DataFrame(s_list)
        events_expanded = events_df[["trial_type"]].reset_index()
        events_expanded = pd.concat([events_expanded, event_meta], axis=1)

        # Calculate stats
        total_events = len(events_expanded)
        events_with_start = events_expanded["event_start"].notna().sum()

        # Type stats
        type_stats_df = (
            events_expanded.assign(is_aligned=events_expanded["event_start"].notna())
            .groupby("trial_type")
            .agg(total=("event", "count"), aligned=("is_aligned", "sum"))
        )
        type_stats = type_stats_df.to_dict("index")

        # Correlation stats
        if "correlation_score" in events_expanded.columns:
            scores = events_expanded["correlation_score"].dropna()
        else:
            scores = pd.Series(dtype=float)

        # Worst trials
        # Group by original trial index
        events_df["is_aligned"] = events_df["sentences"].apply(
            lambda x: x.get("event_start") is not None
        )
        # Re-join with trial_type from original df since index matches
        trial_stats = events_df.groupby(level=0).agg(
            total=("sentences", "count"), aligned=("is_aligned", "sum")
        )
        trial_stats["type"] = df["trial_type"]
        trial_stats["pct"] = (
            trial_stats["aligned"] / trial_stats["total"] * 100
        ).fillna(0)

        worst_trials_df = trial_stats.sort_values("pct").reset_index()
        worst_trials = worst_trials_df.to_dict("records")

        # Build report
        report = {
            "patient_id": patient_id,
            "trials": len(df),
            "stats": {
                "total": {
                    "total": int(total_events),
                    "aligned": int(events_with_start),
                },
                "by_type": type_stats,
            },
        }

        # Print clean report
        print(f"\n{'─' * 50}")
        print(f"  {patient_id} Alignment Report")
        print(f"{'─' * 50}")
        print(f"  Trials: {report['trials']}")

        # Overall Stats
        pct = (events_with_start / total_events * 100) if total_events > 0 else 0
        print(
            f"  Overall: {events_with_start}/{total_events} events aligned ({pct:.1f}%)"
        )
        print()

        print(f"  By Trial Type:")
        for t_type, stats in type_stats.items():
            t_pct = (
                (stats["aligned"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            print(
                f"    • {t_type:<10}: {stats['aligned']}/{stats['total']} ({t_pct:.1f}%)"
            )

        if not scores.empty:
            count = len(scores)
            above_50 = (scores >= 0.5).sum()
            above_80 = (scores >= 0.8).sum()

            print()
            print(f"  Correlation Scores ({count} events):")
            print(f"    • Mean:   {scores.mean():.3f}")
            print(f"    • Range:  {scores.min():.3f} - {scores.max():.3f}")
            print(f"    • ≥50%:   {above_50} ({above_50/count*100:.1f}%)")
            print(f"    • ≥80%:   {above_80} ({above_80/count*100:.1f}%)")

        print()
        print(f"  Trials Performance:")
        for t in worst_trials:
            if t["pct"] < 100:
                print(
                    f"    • Trial {t['index']} ({t['type']}): {t['aligned']}/{t['total']} aligned ({t['pct']:.1f}%)"
                )
            else:
                print("    • None (all trials 100% aligned)")
                break

        print(f"{'─' * 50}\n")

        return report
