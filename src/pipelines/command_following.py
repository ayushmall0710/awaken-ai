"""
Command Following Analysis for Motor Imagery Paradigm

Detects Event-Related Desynchronization (ERD) in Alpha/Beta bands during
motor command/imagery tasks to identify Covert Command Following (CMD).

Approach:
- Each trial contains alternating 'keep' (motor imagery) and 'stop' (rest) commands
- Events are deduplicated and paired: each keep is paired with its adjacent stop (assumption: keep first)
- Each segment spans from command onset to the next command onset (~12-13s)
- ERD computed per-pair in dB: ERD_dB = stop_power - keep_power
- Positive ERD_dB = power decrease during imagery (desynchronization) — literature convention
- Statistical testing: paired one-sided t-test (H1: stop > keep) + mixed effects model
- Classification: contralateral-first with effect size requirement
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.data_loading import UnifiedDataLoader
from src.pipelines.base import BasePipeline
from src.utils.signal_processing import (
    calculate_band_power,
    compute_welch_psd,
)
from src.utils.time_utils import detect_timezone_offset, unix_to_edf
from src.viz.command_following_viz import CommandFollowingVisualizer

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

EPOCH_TRIM_START = 0.5  # seconds to skip at segment start (audio onset + FIR edge)
EPOCH_TRIM_END = 0.4  # seconds to skip at segment end (FIR edge artifacts)
MIN_EPOCH_DURATION = 1.5  # minimum usable segment length after trimming (seconds)

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
    """Paired keep-stop segments cropped from clean preprocessed epochs."""

    keep: mne.Epochs
    stop: mne.Epochs
    side: str  # "left" or "right"
    trial_id: str
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


class CommandFollowingAnalysis(BasePipeline):
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
        super().__init__(loader=loader)
        self.bands = bands or DEFAULT_BANDS.copy()
        self.command_types = command_types or ["left_command", "right_command"]
        self.roi_channels = roi_channels or ROI_CHANNELS.copy()
        self.pairs: List[CommandPair] = []
        self.erd_results: Optional[pd.DataFrame] = None

    # ==================== Public API ====================

    def run(
        self,
        patient_id: str,
        session_id: Optional[str] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Run the full analysis pipeline for a patient (or single session)."""
        self.pairs = []
        self.erd_results = None
        self.viz = CommandFollowingVisualizer(self.bands)
        return super().run(patient_id, session_id=session_id, alpha=alpha)

    def analyze(self, alpha: float = 0.05, **kwargs) -> Any:
        """Delegate to domain-specific calculate_erd()."""
        return self.calculate_erd(bands=self.bands, alpha=alpha)

    def get_stacked_epochs(self, side: str) -> Tuple[mne.EpochsArray, mne.EpochsArray]:
        """Return stacked keep and stop EpochsArray objects for one command side.

        Each CommandPair holds a single-epoch object whose duration varies slightly
        (inter-command gap differs pair to pair). All pairs are trimmed to the
        shortest common length before stacking so the resulting arrays are
        rectangular — no padding that would distort PSD estimates.

        A standard_1020 montage is applied here so that callers (e.g. report
        visualizations) can immediately pass the result to plot_topomap, which
        needs 3-D electrode positions for scalp interpolation.
        """
        side_pairs = [p for p in self.pairs if p.side == side]
        if not side_pairs:
            raise ValueError(f"No pairs found for side '{side}'. Run the pipeline first.")

        info = side_pairs[0].keep.info
        min_samples = min(
            min(len(p.keep.times) for p in side_pairs),
            min(len(p.stop.times) for p in side_pairs),
        )

        keep_data = np.array([p.keep.get_data(copy=False)[0, :, :min_samples] for p in side_pairs])
        stop_data = np.array([p.stop.get_data(copy=False)[0, :, :min_samples] for p in side_pairs])

        tmin = side_pairs[0].keep.times[0]
        montage = mne.channels.make_standard_montage("standard_1020")

        epochs_list = []
        for epochs in (keep_data, stop_data):
            epochs = mne.EpochsArray(epochs, info, tmin=tmin, verbose=False)
            epochs.set_montage(montage, on_missing="ignore", verbose=False)
            epochs_list.append(epochs)

        return tuple(epochs_list)

    # ==================== Data Loading ====================

    def load(self) -> None:
        """
        Load clean epochs (from artifact rejection) and extract
        paired keep-stop command segments.

        Preprocessing saves artifact-rejected epochs per trial type as .fif files.
        This method loads those clean epochs, picks ROI channels, then uses
        Epochs.crop() to extract individual keep/stop segments directly.
        """
        cmd_trials = self.aligned_events[self.aligned_events["trial_type"].isin(self.command_types)]

        if len(cmd_trials) == 0:
            raise ValueError(f"No command trials found for patient {self.patient_id}")

        for (session_id, trial_type), group in cmd_trials.groupby(["session_id", "trial_type"]):
            all_epochs = self.loader.load_clean_epochs(self.patient_id, session_id, trial_type)
            if len(all_epochs) == 0:
                continue

            # .copy() before .pick() so that any future cache layer won't receive a mutated object.
            all_epochs = all_epochs.copy()
            all_epochs.pick(self.roi_channels)

            session_trials = self.aligned_events[self.aligned_events["session_id"] == session_id]
            tz_offset = detect_timezone_offset(all_epochs, session_trials)
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
                        trial_id = trial["trial_id"]
                        pairs = self._extract_trial_pairs(match_epoch, tz_offset, trial, side, trial_id)
                        self.pairs.extend(pairs)
                    except Exception as e:
                        logger.warning(f"Failed to extract pairs for trial {trial.get('trial_id', 'unknown')}: {e}")

        logger.info(f"Loaded {len(self.pairs)} keep-stop pairs")

    # ==================== Epoch Extraction ====================

    def _extract_trial_pairs(
        self,
        epochs: mne.Epochs,
        tz_offset: float,
        trial: pd.Series,
        side: str,
        trial_id: str,
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
        for k, (keep_pos, stop_pos) in enumerate(zip(keep_positions, stop_positions)):
            # Keep window extends to the start of the adjacent stop command.
            # Stop window extends to the start of the next keep command (or uses
            # audio-only window for the last pair, which has no following keep).
            keep_seg = self._crop_segment(epochs, keep_pos, tz_offset, segment_end_unix=stop_pos["start"])
            stop_response_end = keep_positions[k + 1]["start"] if k + 1 < len(keep_positions) else None
            stop_seg = self._crop_segment(epochs, stop_pos, tz_offset, segment_end_unix=stop_response_end)

            if keep_seg is None or stop_seg is None:
                continue

            pairs.append(
                CommandPair(
                    keep=keep_seg,
                    stop=stop_seg,
                    side=side,
                    trial_id=trial_id,
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
        segment_end_unix: Optional[float] = None,
    ) -> Optional[mne.Epochs]:
        """Crop a single keep/stop segment from the trial epochs with transition trimming.

        Converts Unix timestamps to epoch-relative times, then uses Epochs.crop().

        Args:
            epochs: Single-epoch MNE object (from _find_matching_epoch).
            position: Position dict with 'start' and 'end' Unix timestamps.
            tz_offset: Timezone correction (seconds) from detect_timezone_offset.
            segment_end_unix: If provided, crop to this Unix timestamp instead of
                position['end']. Used to extend the window to the next command onset,
                capturing the full response period (~12s) rather than just the
                audio command window (~3s).
        """
        edf_start_unix = epochs.info["meas_date"].timestamp()
        event_edf_time = epochs.events[0, 0] / epochs.info["sfreq"]

        seg_start_edf = unix_to_edf(position["start"], edf_start_unix=edf_start_unix, timezone_offset=tz_offset)
        end_unix = segment_end_unix if segment_end_unix is not None else position["end"]
        seg_end_edf = unix_to_edf(end_unix, edf_start_unix=edf_start_unix, timezone_offset=tz_offset)

        # Convert EDF-relative seconds to epoch-relative seconds
        tmin = (seg_start_edf - event_edf_time) + EPOCH_TRIM_START
        tmax = (seg_end_edf - event_edf_time) - EPOCH_TRIM_END

        if tmax - tmin < MIN_EPOCH_DURATION:
            logger.debug("Segment too short after trimming: %.2fs (min=%.2fs)", tmax - tmin, MIN_EPOCH_DURATION)
            return None

        if tmin < epochs.times[0] or tmax > epochs.times[-1]:
            logger.debug(
                "Segment out of epoch bounds: tmin=%.2f tmax=%.2f epoch=[%.2f, %.2f]",
                tmin,
                tmax,
                epochs.times[0],
                epochs.times[-1],
            )
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

        Note: Artifact rejection (ICA) is already done during preprocessing.
        Bandpass is still needed to isolate motor-relevant frequencies.
        """
        if len(self.pairs) == 0:
            raise ValueError("No pairs loaded. Call load() first.")

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

        ERD_dB = stop_power_dB - keep_power_dB  (literature convention)
        Positive ERD_dB indicates desynchronization during motor imagery (keep < stop).
        """
        if len(self.pairs) == 0:
            raise ValueError("No pairs loaded. Call load() first.")

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
        trial_ids = [p.trial_id for p in pairs]

        channels = pairs[0].keep.ch_names
        results = []

        for band_name in bands:
            for ch_idx, ch in enumerate(channels):
                keep_arr = np.array([kp[band_name][ch_idx] for kp in keep_powers])
                stop_arr = np.array([sp[band_name][ch_idx] for sp in stop_powers])

                # Positive = desynchronization during keep (literature convention)
                erd_per_pair = stop_arr - keep_arr
                mean_erd = np.mean(erd_per_pair)

                # One-sided paired t-test: H1: stop > keep (ERD_dB > 0)
                _, p_val = stats.ttest_rel(stop_arr, keep_arr, alternative="greater")
                d = compute_cohens_d(erd_per_pair)

                # Mixed effects: accounts for within-trial correlation (epochs from
                # same trial share brain state, so are not independent)
                p_mixed = self._run_mixed_model(erd_per_pair, trial_ids)
                is_contralateral = ch == CONTRALATERAL_MAP.get(side)

                results.append(
                    {
                        "side": side,
                        "channel": ch,
                        "band": band_name,
                        "keep_mean_dB": float(np.mean(keep_arr)),
                        "stop_mean_dB": float(np.mean(stop_arr)),
                        "erd_dB": mean_erd,
                        "erd_std": np.std(erd_per_pair, ddof=1) if len(pairs) > 1 else np.nan,
                        "n_pairs": len(pairs),
                        "accuracy": float(np.mean(erd_per_pair > 0)),
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
        if n_unique_trials < 3:
            logger.warning(
                "Mixed model has only %d unique trial groups — random-effect estimates "
                "are unreliable with fewer than 3 groups. Interpret p_mixed cautiously.",
                n_unique_trials,
            )

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
            return np.nan

    def generate_summary(
        self,
        erd_threshold_dB: float = 1.0,
        d_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Generate classification summary from ERD results, including binomial chance level."""
        if self.erd_results is None or self.erd_results.empty:
            return {
                "cmd_status": "ERROR: No results",
                "n_pairs": 0,
                "left_pairs": 0,
                "right_pairs": 0,
                "n_significant_contra": 0,
                "classification_chance_level": 0.0,
                "significant_results": [],
            }

        n_pairs = len(self.pairs)
        left_pairs = sum(1 for p in self.pairs if p.side == "left")
        right_pairs = sum(1 for p in self.pairs if p.side == "right")

        # Theoretical binomial chance level: minimum accuracy to beat random at alpha=0.05.
        chance_level = stats.binom.ppf(0.95, n_pairs, 0.5) / n_pairs if n_pairs > 0 else 0.5

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
            & (df["erd_dB"] > erd_threshold_dB)
            & (df["cohens_d"] > d_threshold)
        ]

        is_cmd_positive = len(sig_contra) > 0

        return {
            "cmd_status": "CMD+" if is_cmd_positive else "CMD-",
            "n_pairs": n_pairs,
            "left_pairs": left_pairs,
            "right_pairs": right_pairs,
            "n_significant_contra": len(sig_contra),
            "classification_chance_level": chance_level,
            "significant_results": sig_contra.to_dict("records"),
        }
