"""
Command Following Analysis for Motor Imagery Paradigm

Detects Event-Related Desynchronization (ERD) in Alpha/Beta bands during
motor command/imagery tasks to identify Covert Command Following (CMD).

Approach:
- Each trial contains alternating 'keep' (motor imagery) and 'stop' (rest) commands
- Events are deduplicated and paired: each keep is paired with its adjacent stop
- ERD computed per-pair in dB: ERD_dB = 10 * log10(keep_power / stop_power)
- Negative ERD_dB = power decrease during imagery (desynchronization)
- Statistical testing: paired one-sided t-test + mixed effects model
- Classification: contralateral-first with effect size requirement
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.data_loading import UnifiedDataLoader
from src.utils.signal_processing import (
    calculate_band_power,
    compute_band_envelope,
    compute_welch_psd,
)
from src.utils.time_utils import detect_timezone_offset, unix_to_edf

logger = logging.getLogger(__name__)

# Set plotting defaults
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {"Alpha": (8, 13), "Beta": (13, 30)}
ROI_CHANNELS: List[str] = ["C3", "C4", "Cz"]
CONTRALATERAL_MAP = {"left": "C4", "right": "C3"}

EPOCH_TRIM_START = 0.5  # seconds to skip at epoch start (audio onset + transition)
EPOCH_TRIM_END = 0.1  # seconds to skip at epoch end (tail transition)
MIN_EPOCH_DURATION = 1.5  # minimum usable epoch length after trimming

# ---------------------------------------------------------------------------
# Visualization Constants
# ---------------------------------------------------------------------------

KEEP_COLOR = "#22c55e"
STOP_COLOR = "#ef4444"
RESPONSE_KEEP_COLOR = "#bbf7d0"  # light green
RESPONSE_STOP_COLOR = "#fecaca"  # light red
INSTRUCTION_COLOR = "#fbbf24"  # yellow


@dataclass
class CommandPair:
    """Paired keep-stop segments cropped from clean ENG-03 epochs."""

    keep: mne.Epochs
    stop: mne.Epochs
    side: str  # "left" or "right"
    trial_idx: int
    keep_start: float  # keep onset (Unix timestamp)
    stop_start: float  # stop onset (Unix timestamp)


# ---------------------------------------------------------------------------
# Event Deduplication (command-paradigm specific)
# ---------------------------------------------------------------------------


def deduplicate_and_label(events: list, start_with: str = "keep") -> list:
    """
    Merge overlapping detections into single events and assign alternating labels.

    Each ~13s position has TWO detections (keep + stop matched the same audio).
    We merge them into one event per position, then alternate labels.

    Filters out entries missing event_start/event_end,
    so raw sentence dicts can be passed directly.

    TODO: The alternating keep/stop assumption is a placeholder.
    Actual command sequence needs to be confirmed with Prof/Alex.
    Using this assumption to unblock downstream analysis.
    """
    valid = [e for e in events if e.get("event_start") and e.get("event_end")]
    if not valid:
        return []

    sorted_events = sorted(valid, key=lambda e: e["event_start"])

    positions = []
    i = 0
    while i < len(sorted_events):
        current = sorted_events[i]
        merged = {
            "start": current["event_start"],
            "end": current["event_end"],
            "corr": current.get("correlation_score", 0),
        }

        while i + 1 < len(sorted_events) and abs(sorted_events[i + 1]["event_start"] - current["event_start"]) < 1.0:
            i += 1
            nxt = sorted_events[i]
            merged["end"] = max(merged["end"], nxt["event_end"])
            merged["corr"] = max(merged["corr"], nxt.get("correlation_score", 0))

        positions.append(merged)
        i += 1

    labels = ["stop", "keep"] if start_with == "stop" else ["keep", "stop"]
    for idx, pos in enumerate(positions):
        pos["type"] = labels[idx % 2]

    return positions


# ---------------------------------------------------------------------------
# Statistics Utilities
# ---------------------------------------------------------------------------


def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction to control false discovery rate across channels/bands."""
    if len(p_values) == 0:
        return np.array([]), np.array([])

    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return rejected, p_corrected


def compute_cohens_d(differences: np.ndarray) -> float:
    """Compute paired Cohen's d to quantify effect SIZE (not just significance) of desynchronization."""
    if len(differences) < 2 or np.std(differences) == 0:
        return 0.0
    return np.mean(differences) / np.std(differences, ddof=1)


# ---------------------------------------------------------------------------
# Power Computation
# ---------------------------------------------------------------------------


def compute_log_band_power(segment: mne.Epochs, bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Compute log10 band power (dB-scale) for a single-epoch segment."""
    data = segment.get_data()[0]  # (n_channels, n_times)
    sfreq = segment.info["sfreq"]
    fmin = min(low for low, _ in bands.values())
    fmax = max(high for _, high in bands.values())

    freqs, psd = compute_welch_psd(data, sfreq=sfreq, fmin=fmin, fmax=fmax)
    band_power = calculate_band_power(psd, freqs, bands)

    # Convert to dB scale: 10 * log10(power)
    return {band: 10 * np.log10(np.maximum(power, 1e-30)) for band, power in band_power.items()}


# ---------------------------------------------------------------------------
# Main Analysis Class
# ---------------------------------------------------------------------------


class CommandFollowingAnalysis:
    """
    Analyzes Command Following trials to detect ERD in motor cortex.

    Uses paired keep-stop epochs with dB-scale ERD, one-sided paired t-tests,
    contralateral-first classification, and optional mixed effects modeling.
    """

    def __init__(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        command_types: Optional[List[str]] = None,
        roi_channels: Optional[List[str]] = None,
        loader: Optional[UnifiedDataLoader] = None,
    ):
        self.bands = bands or DEFAULT_BANDS.copy()
        self.command_types = command_types or ["left_command", "right_command"]
        self.roi_channels = roi_channels or ROI_CHANNELS.copy()
        self.loader = loader or UnifiedDataLoader()

        self.patient_id: Optional[str] = None
        self.aligned_events: Optional[pd.DataFrame] = None
        self.pairs: List[CommandPair] = []
        self.erd_results: Optional[pd.DataFrame] = None

    # ==================== Public API ====================

    def run(
        self,
        patient_id: str,
        alpha: float = 0.05,
        summary: bool = False,
        erd_threshold_dB: float = -1.0,
        d_threshold: float = 0.5,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Run the full analysis pipeline for a patient.

        Returns erd_df by default. If summary=True, returns (erd_df, summary_dict).
        """
        self.patient_id = patient_id
        self.aligned_events = self.loader.load_aligned_events(patient_id)
        self.pairs = []
        self.erd_results = None

        self.load_epochs()
        self.preprocess()
        erd_df = self.calculate_erd(bands=self.bands, alpha=alpha)

        if summary:
            return erd_df, self.generate_summary(erd_threshold_dB=erd_threshold_dB, d_threshold=d_threshold)
        return erd_df

    # ==================== Data Loading ====================

    def load_epochs(self) -> None:
        """
        Load clean epochs (from ENG-03 artifact rejection) and extract
        paired keep-stop command segments.

        ENG-03 saves artifact-rejected epochs per trial type as .fif files.
        This method loads those clean epochs, picks ROI channels, then uses
        Epochs.crop() to extract individual keep/stop segments directly.
        """
        cmd_trials = self.aligned_events[self.aligned_events["trial_type"].isin(self.command_types)]

        if len(cmd_trials) == 0:
            raise ValueError(f"No command trials found for patient {self.patient_id}")

        trial_idx = 0
        for (date, trial_type), group in cmd_trials.groupby(["date", "trial_type"]):
            all_epochs = self.loader.load_clean_epochs(self.patient_id, date, trial_type)
            if len(all_epochs) == 0:
                continue

            all_epochs.pick(self.roi_channels)

            date_trials = self.aligned_events[self.aligned_events["date"] == date]
            tz_offset = detect_timezone_offset(all_epochs, date_trials)
            side = trial_type.split("_")[0]

            # Pre-calculate mapping from epoch sample index to epoch index
            # We need to match trial start_time (Unix) -> sample index
            edf_start_unix = all_epochs.info["meas_date"].timestamp()
            sfreq = all_epochs.info["sfreq"]

            # Tolerance for matching (e.g. 0.5s in samples)
            sample_tolerance = int(0.5 * sfreq)

            for _, trial in group.iterrows():
                # Calculate expected sample for this trial
                # Logic mirrors ArtifactRejector._build_fixed_window_epochs
                start_edf = float(trial["start_time"]) - edf_start_unix + tz_offset
                if np.isnan(start_edf) or start_edf < 0:
                    continue

                # Find matching epoch
                match_epoch = self._find_matching_epoch(
                    all_epochs,
                    expected_start_edf=start_edf,
                    sample_tolerance=sample_tolerance,
                )

                if match_epoch is not None:
                    try:
                        pairs = self._extract_trial_pairs(match_epoch, tz_offset, trial, side, trial_idx)
                        self.pairs.extend(pairs)
                    except Exception as e:
                        logger.warning(f"Failed to extract pairs for trial {trial_idx}: {e}")
                else:
                    pass  # Epoch was likely dropped during cleaning

                trial_idx += 1

        logger.info(f"Loaded {len(self.pairs)} keep-stop pairs")

    # ==================== Epoch Extraction ====================

    def _extract_trial_pairs(
        self,
        epochs: mne.Epochs,
        tz_offset: float,
        trial: pd.Series,
        side: str,
        trial_idx: int,
    ) -> List[CommandPair]:
        """Extract paired keep-stop segments from a single trial's clean epochs.

        Deduplicates events via deduplicate_and_label (which guarantees alternating
        keep/stop), pairs them, then crops both. Skips pair if either crop fails.
        """
        positions = deduplicate_and_label(trial["sentences"], start_with="keep")

        # Pair by alternating position: even=keep, odd=stop (guaranteed by deduplicate_and_label)
        keep_positions = positions[0::2]
        stop_positions = positions[1::2]

        pairs = []
        for keep_pos, stop_pos in zip(keep_positions, stop_positions):
            keep_seg = self._crop_segment(epochs, keep_pos, tz_offset)
            stop_seg = self._crop_segment(epochs, stop_pos, tz_offset)

            if keep_seg is None or stop_seg is None:
                continue

            pairs.append(
                CommandPair(
                    keep=keep_seg,
                    stop=stop_seg,
                    side=side,
                    trial_idx=trial_idx,
                    keep_start=keep_pos["start"],
                    stop_start=stop_pos["start"],
                )
            )

        return pairs

    def _crop_segment(
        self,
        epochs: mne.Epochs,
        position: dict,
        tz_offset: float,
    ) -> Optional[mne.Epochs]:
        """Crop a single keep/stop segment from the trial epochs with transition trimming.

        Converts Unix timestamps to epoch-relative times, then uses Epochs.crop().
        """
        edf_start_unix = epochs.info["meas_date"].timestamp()
        event_edf_time = epochs.events[0, 0] / epochs.info["sfreq"]

        seg_start_edf = unix_to_edf(position["start"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset)
        seg_end_edf = unix_to_edf(position["end"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset)

        # Convert EDF-relative seconds to epoch-relative seconds
        tmin = (seg_start_edf - event_edf_time) + EPOCH_TRIM_START
        tmax = (seg_end_edf - event_edf_time) - EPOCH_TRIM_END

        if tmax - tmin < MIN_EPOCH_DURATION:
            logger.debug(f"Segment too short after trimming: {tmax - tmin:.2f}s")
            return None

        if tmin < epochs.times[0] or tmax > epochs.times[-1]:
            return None

        return epochs.copy().crop(tmin=tmin, tmax=tmax)

    def _find_matching_epoch(
        self,
        epochs: mne.Epochs,
        expected_start_edf: float,
        sample_tolerance: int,
    ) -> Optional[mne.Epochs]:
        """Find the single epoch in `epochs` that matches the expected start time.

        Args:
            epochs: MNE Epochs object containing all clean epochs for the session.
            expected_start_edf: Expected start time in EDF-relative seconds.
            sample_tolerance: Allowed difference in samples for a match.

        Returns:
            The matching single-epoch MNE object, or None if no match found.
        """
        sfreq = epochs.info["sfreq"]
        expected_samp = int(np.round(expected_start_edf * sfreq))

        # epochs.events is (n_epochs, 3) array where col 0 is sample index
        diffs = np.abs(epochs.events[:, 0] - expected_samp)
        match_idx = np.argmin(diffs)

        if diffs[match_idx] <= sample_tolerance:
            return epochs[match_idx]

        return None

    # ==================== Preprocessing ====================

    def preprocess(self) -> None:
        """Apply bandpass filter (8-30Hz) to isolate alpha/beta bands.

        Note: Artifact rejection (ICA) is already done by ENG-03.
        Bandpass is still needed to isolate motor-relevant frequencies.
        """
        if len(self.pairs) == 0:
            raise ValueError("No pairs loaded. Call load_epochs() first.")

        l_freq, h_freq = 8.0, 30.0
        logger.info(f"Bandpass filtering {len(self.pairs)} pairs ({l_freq}-{h_freq}Hz)...")

        for pair in self.pairs:
            pair.keep.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
            pair.stop.filter(l_freq, h_freq, fir_design="firwin", verbose=False)

    # ==================== ERD Computation ====================

    def calculate_erd(
        self,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Calculate per-pair ERD in dB with paired one-sided t-test and mixed effects.

        ERD_dB = 10 * log10(keep_power) - 10 * log10(stop_power)
        Negative ERD_dB indicates desynchronization during motor imagery.
        """
        if len(self.pairs) == 0:
            raise ValueError("No pairs loaded. Call load_epochs() first.")

        bands = bands or DEFAULT_BANDS.copy()
        results = []

        for side in ["left", "right"]:
            side_pairs = [p for p in self.pairs if p.side == side]
            if len(side_pairs) < 2:
                continue
            side_results = self._compute_side_erd(side, side_pairs, bands)
            results.extend(side_results)

        df = pd.DataFrame(results)

        if len(df) > 0:
            # FDR correction per side
            for side in df["side"].unique():
                mask = df["side"] == side
                rejected, p_corrected = apply_fdr_correction(df.loc[mask, "p_value_raw"].tolist(), alpha)
                df.loc[mask, "p_value"] = p_corrected
                df.loc[mask, "significant"] = rejected

        self.erd_results = df
        return df

    def _compute_side_erd(
        self, side: str, pairs: List[CommandPair], bands: Dict[str, Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """Compute per-pair ERD for one side with paired tests and mixed effects."""
        keep_powers = [compute_log_band_power(p.keep, bands) for p in pairs]
        stop_powers = [compute_log_band_power(p.stop, bands) for p in pairs]
        trial_indices = [p.trial_idx for p in pairs]

        channels = pairs[0].keep.ch_names
        results = []

        for band_name in bands:
            for ch_idx, ch in enumerate(channels):
                keep_arr = np.array([kp[band_name][ch_idx] for kp in keep_powers])
                stop_arr = np.array([sp[band_name][ch_idx] for sp in stop_powers])

                erd_per_pair = keep_arr - stop_arr  # negative = desynchronization
                mean_erd = np.mean(erd_per_pair)

                # One-sided paired t-test: H1: keep < stop (erd_dB < 0)
                _, p_val = stats.ttest_rel(keep_arr, stop_arr, alternative="less")
                d = compute_cohens_d(erd_per_pair)

                # Mixed effects: accounts for within-trial correlation (epochs from
                # same trial share brain state, so are not independent)
                p_mixed = self._run_mixed_model(erd_per_pair, trial_indices)
                is_contralateral = ch == CONTRALATERAL_MAP.get(side)

                results.append(
                    {
                        "side": side,
                        "channel": ch,
                        "band": band_name,
                        "erd_dB": mean_erd,
                        "erd_std": np.std(erd_per_pair, ddof=1),
                        "n_pairs": len(pairs),
                        "p_value_raw": p_val,
                        "cohens_d": d,
                        "p_mixed": p_mixed,
                        "is_contralateral": is_contralateral,
                    }
                )

        return results

    def _run_mixed_model(self, erd_values: np.ndarray, trial_indices: list) -> float:
        """Mixed effects model: ERD ~ 1 + (1|trial).

        Accounts for non-independence of epochs within the same trial
        (shared brain state, recording conditions). Returns intercept p-value.
        """
        n_unique_trials = len(set(trial_indices))
        if n_unique_trials < 2 or len(erd_values) < 3:
            return np.nan

        df = pd.DataFrame(
            {
                "erd_dB": erd_values,
                "trial": [str(t) for t in trial_indices],
            }
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm("erd_dB ~ 1", df, groups=df["trial"])
                result = model.fit(disp=False, reml=True)
            return result.pvalues["Intercept"]
        except Exception:
            logger.debug("Mixed model failed to converge, returning NaN")

    def generate_summary(
        self,
        erd_threshold_dB: float = -1.0,
        d_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate classification summary from ERD results."""
        if self.erd_results is None:
            return {
                "cmd_status": "ERROR: No results",
                "n_pairs": 0,
                "n_significant_contra": 0,
                "significant_results": [],
            }

        df = self.erd_results

        # Filter for significant contralateral desynchronization
        # Criteria:
        # 1. Contralateral channel
        # 2. Significant (p_corrected < alpha)
        # 3. ERD < threshold (negative = desync)
        # 4. Effect size magnitude > threshold (optional, but good for robustness)

        # Note: cohens_d sign matches ERD sign (keep - stop).
        # So if ERD is negative, d is negative. We check abs(d) > threshold.

        sig_contra = df[
            (df["is_contralateral"])
            & (df["significant"])
            & (df["erd_dB"] < erd_threshold_dB)
            & (df["cohens_d"].abs() > d_threshold)
        ]

        is_cmd_positive = len(sig_contra) > 0

        return {
            "cmd_status": "CMD+" if is_cmd_positive else "CMD-",
            "n_pairs": len(self.pairs),
            "n_significant_contra": len(sig_contra),
            "significant_results": sig_contra.to_dict("records"),
        }

    def _plot_erd_bar(self, df: pd.DataFrame) -> plt.Figure:
        """Bar plot showing ERD (dB) by channel and frequency band, faceted by side."""
        sides = df["side"].unique()
        n_sides = len(sides)

        if n_sides == 0:
            return plt.figure()

        # Create subplots: 1 row, n_sides columns
        fig, axes = plt.subplots(1, n_sides, figsize=(6 * n_sides, 6), sharey=True, squeeze=False)
        axes = axes.flatten()

        for i, side in enumerate(sides):
            side_df = df[df["side"] == side]
            ax = axes[i]

            sns.barplot(
                data=side_df,
                x="channel",
                y="erd_dB",
                hue="band",
                palette="viridis",
                ax=ax,
            )

            # Determine expected contralateral channel
            contra_ch = CONTRALATERAL_MAP.get(side.lower())

            # Add title indicating expected effect
            ax.set_title(f"Command: {side.capitalize()}\n(Expect {contra_ch} Desync)")
            ax.set_xlabel("Channel")
            if i == 0:
                ax.set_ylabel("ERD (dB)\n(Negative = Desynchronization)")
            else:
                ax.set_ylabel("")

            # Add reference line at 0
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        plt.tight_layout()
        return fig

    def visualize_trial(
        self,
        trial_idx: int = 0,
        trial_type: str = "right_command",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot a single command trial: DC channel + alpha/beta power with labeled regions.

        Plots raw audio waveform to verify alignment and band power envelopes to visualize ERD.
        """
        if self.patient_id is None:
            raise ValueError("Patient ID not set. Run .run() or set .patient_id first.")

        if self.aligned_events is None:
            self.aligned_events = self.loader.load_aligned_events(self.patient_id)

        # Find the specific trial metadata
        cmd_trials = self.aligned_events[self.aligned_events["trial_type"].str.contains("command")].reset_index(
            drop=True
        )

        # Filter by requested trial type if provided, otherwise just index into all command trials
        if trial_type:
            cmd_trials = cmd_trials[cmd_trials["trial_type"] == trial_type].reset_index(drop=True)

        if trial_idx >= len(cmd_trials):
            logger.error(f"Trial index {trial_idx} out of range for {trial_type} (max {len(cmd_trials) - 1})")
            return None

        trial = cmd_trials.iloc[trial_idx]
        date = trial["date"]
        actual_type = trial["trial_type"]

        # Load raw data for this specific trial's session
        raw = self.loader.get_patient(self.patient_id).raw
        if isinstance(raw, dict):
            raw = raw.get(date)

        if raw is None:
            logger.error(f"Could not load raw data for {self.patient_id} on {date}")
            return None

        # Pre-calculate time conversions
        edf_start_unix = raw.info["meas_date"].timestamp()

        # We need session-specific events to get timezone offset
        session_events = self.aligned_events[self.aligned_events["date"] == date]
        tz_offset = detect_timezone_offset(raw, session_events)

        trial_start_edf = unix_to_edf(trial["start_time"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset)
        trial_end_edf = unix_to_edf(trial["end_time"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset)

        # Crop to trial
        raw_trial = raw.copy().crop(tmin=trial_start_edf, tmax=trial_end_edf)
        sfreq = raw_trial.info["sfreq"]
        times = raw_trial.times

        # Extract events for plotting
        raw_events = [
            {
                "event_start": e["event_start"],
                "event_end": e["event_end"],
                "correlation_score": e.get("correlation_score", 0),
            }
            for e in trial["sentences"]
            if e.get("event_start") and e.get("event_end")
        ]

        positions = deduplicate_and_label(raw_events)

        # Convert event times to trial-relative
        events = []
        for pos in positions:
            start_rel = (
                unix_to_edf(pos["start"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset) - trial_start_edf
            )
            end_rel = (
                unix_to_edf(pos["end"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset) - trial_start_edf
            )
            events.append({"start": max(0, start_rel), "end": min(raw_trial.times[-1], end_rel), "type": pos["type"]})

        instruction_end = events[0]["start"] if events else 0

        # Select channels
        dc_channels = [ch for ch in raw_trial.ch_names if "DC" in ch.upper()]
        dc_ch = "DC5" if "DC5" in raw_trial.ch_names else (dc_channels[0] if dc_channels else None)

        side = actual_type.split("_")[0]
        motor_ch = "C4" if side == "left" else "C3"
        if motor_ch not in raw_trial.ch_names:
            motor_ch = "C3" if motor_ch == "C4" else "C4"  # Fallback

        # Get data
        dc_data = raw_trial.get_data(picks=[dc_ch])[0] if dc_ch else np.zeros_like(times)
        motor_data = raw_trial.get_data(picks=[motor_ch])[0]

        # Compute envelopes
        alpha_env = compute_band_envelope(motor_data, sfreq, self.bands.get("Alpha", (8, 13)))
        beta_env = compute_band_envelope(motor_data, sfreq, self.bands.get("Beta", (13, 30)))

        # Plot
        n_keep = sum(1 for e in events if e["type"] == "keep")
        n_stop = sum(1 for e in events if e["type"] == "stop")

        fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
        fig.suptitle(
            f"{self.patient_id} — {actual_type} (Trial {trial_idx}) | {motor_ch} (contralateral)\n"
            f"{n_keep} keep + {n_stop} stop events",
            fontsize=13,
            fontweight="bold",
        )

        # 1. Audio
        axes[0].plot(times, dc_data, "k", linewidth=0.4, alpha=0.7)
        axes[0].set_ylabel(f"{dc_ch or 'Audio'}")
        axes[0].set_title("Audio Signal (DC Channel)")

        # 2. Alpha
        axes[1].plot(times, alpha_env, color="#2563eb", linewidth=0.8)
        axes[1].set_ylabel(f"{motor_ch} Alpha ({self.bands['Alpha'][0]}–{self.bands['Alpha'][1]} Hz)")
        axes[1].set_title("Alpha Band Power")

        # 3. Beta
        axes[2].plot(times, beta_env, color="#7c3aed", linewidth=0.8)
        axes[2].set_ylabel(f"{motor_ch} Beta ({self.bands['Beta'][0]}–{self.bands['Beta'][1]} Hz)")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].set_title("Beta Band Power")

        # Highlights
        if instruction_end > 0:
            for ax in axes:
                ax.axvspan(0, instruction_end, color=INSTRUCTION_COLOR, alpha=0.25)

        for i, event in enumerate(events):
            is_keep = event["type"] == "keep"
            cmd_color = KEEP_COLOR if is_keep else STOP_COLOR

            # Command audio window
            for ax in axes:
                ax.axvspan(event["start"], event["end"], color=cmd_color, alpha=0.3)

            # Response period
            if i < len(events) - 1:
                resp_start = event["end"]
                resp_end = events[i + 1]["start"]
                resp_color = RESPONSE_KEEP_COLOR if is_keep else RESPONSE_STOP_COLOR
                for ax in axes:
                    ax.axvspan(resp_start, resp_end, color=resp_color, alpha=0.15)

        patches = [
            mpatches.Patch(color=INSTRUCTION_COLOR, alpha=0.25, label="Instruction"),
            mpatches.Patch(color=KEEP_COLOR, alpha=0.3, label="KEEP command"),
            mpatches.Patch(color=RESPONSE_KEEP_COLOR, alpha=0.3, label="KEEP response"),
            mpatches.Patch(color=STOP_COLOR, alpha=0.3, label="STOP command"),
            mpatches.Patch(color=RESPONSE_STOP_COLOR, alpha=0.3, label="STOP response"),
        ]
        axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=2)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved trial plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_summary(self, save_dir: Optional[Union[str, Path]] = None, show: bool = True) -> List[plt.Figure]:
        """
        Generate summary plots: Topomaps, Boxplots, and Bar charts of ERD results.
        """
        if self.erd_results is None or self.erd_results.empty:
            logger.warning("No ERD results to visualize. Run calculate_erd() first.")
            return []

        figs = []
        save_dir = Path(save_dir) if save_dir else None

        # 1. ERD by Channel (Bar Plot)
        fig_bar = self._plot_erd_bar(self.erd_results)
        figs.append(fig_bar)
        if save_dir:
            fig_bar.savefig(save_dir / f"{self.patient_id}_erd_bar.png", dpi=300, bbox_inches="tight")

        if not show:
            plt.close(fig_bar)

        # 2. Topomaps: Series (Before/After) for Keep and Stop
        if self.pairs and hasattr(self.pairs[0].keep, "info"):
            for band in self.bands:
                for side in ["left", "right"]:
                    # Check if we have data for this side
                    start_pairs = [p for p in self.pairs if p.side == side]
                    if not start_pairs:
                        continue

                    fig_series = self._plot_topomap_series(side, band)
                    if fig_series:
                        figs.append(fig_series)
                        if save_dir:
                            fig_series.savefig(
                                save_dir / f"{self.patient_id}_topo_series_{side}_{band}.png",
                                dpi=300,
                                bbox_inches="tight",
                            )
                        if not show:
                            plt.close(fig_series)

        return figs

    def _plot_topomap_series(self, side: str, band: str) -> Optional[plt.Figure]:
        """Plot a series of topomaps: Before vs After for Keep and Stop conditions."""

        # Filter pairs for this side
        pairs = [p for p in self.pairs if p.side == side]
        if not pairs:
            return None

        info = pairs[0].keep.info
        band_freqs = self.bands[band]

        # Define time windows (relative to epoch start)
        # Assuming minimal epoch is ~1.5s.
        # "Part 1" = Initial part of the command execution
        # "Part 2" = Later part of the command execution
        # Note: These are relative to the *cropped* command epoch, which captures the
        # active command following period.

        tmin = pairs[0].keep.times[0]
        tmax = pairs[0].keep.times[-1]
        midpoint = (tmin + tmax) / 2

        windows = {"Part 1": (tmin, midpoint), "Part 2": (midpoint, tmax)}

        conditions = ["keep", "stop"]
        n_rows = len(conditions)
        n_cols = len(windows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        fig.suptitle(f"Spatial Evolution ({side.capitalize()} {band})", fontsize=12, fontweight="bold")

        # Prepare data containers
        # We need to aggregate data across all pairs for this side
        # Structure: condition -> window -> [values per channel]

        for r, cond in enumerate(conditions):
            for c, (win_name, (w_start, w_end)) in enumerate(windows.items()):
                ax = axes[r, c] if n_rows > 1 else axes[c]

                # aggregate power
                power_values = []
                for p in pairs:
                    epoch = getattr(p, cond)
                    # Crop logic: complex to crop every single epoch again.
                    # Instead, get data and time-mask it.
                    data = epoch.get_data(copy=False)[0]  # (n_channels, n_times)
                    times = epoch.times
                    mask = (times >= w_start) & (times <= w_end)

                    if not np.any(mask):
                        continue

                    # Calculate band power for this segment
                    # compute_welch_psd requires (n_channels, n_times)
                    segment = data[:, mask]
                    sfreq = epoch.info["sfreq"]

                    freqs, psd = compute_welch_psd(segment, sfreq, n_per_seg=int(sfreq / 2))
                    bp = calculate_band_power(psd, freqs, {band: band_freqs}, relative=False)[band]

                    # Log transform for dB-like visualization (though raw power is fine for topo)
                    # Let's use raw power but standardized? Or just log.
                    # 10 * log10(power)
                    bp_db = 10 * np.log10(bp + 1e-9)
                    power_values.append(bp_db)

                if not power_values:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                    continue

                # Average across pairs
                avg_power = np.mean(np.array(power_values), axis=0)

                # Plot using helper
                self._plot_topomap(
                    avg_power,
                    info,
                    ax=ax,
                    title=f"{cond.capitalize()}: {win_name}\n({w_start:.1f}-{w_end:.1f}s)",
                    vmin=None,
                    vmax=None,
                )

        plt.tight_layout()
        return fig

    def _plot_topomap(
        self,
        values: np.ndarray,
        info: mne.Info,
        ax: plt.Axes,
        title: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """Helper to plot a single topomap on a given axes."""
        if info.get_montage() is None:
            try:
                info.set_montage("standard_1020", match_case=False)
            except Exception as e:
                logger.warning(f"Failed to set montage: {e}")

        mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            show=False,
            cmap="viridis",
            vlim=(vmin, vmax) if vmin is not None else (None, None),
            contours=0,
            image_interp="cubic",
            sensors=True,
        )
        ax.set_title(title, fontsize=10)
