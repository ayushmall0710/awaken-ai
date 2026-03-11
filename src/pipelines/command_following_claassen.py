"""
Claassen-inspired SVM Command Following Classifier

Implements the SVM based approach described in Claassen et al. 2019.
Instead of testing specific channels against a power-drop threshold,
a per-patient Linear SVM is trained on spectral power features across
*all* available EEG channels to discriminate keep (motor imagery)
from stop (rest) segments.

Key differences from the parent ERD pipeline:
- Uses all available EEG channels (not just C3/C4/Cz)
- Extracts per-channel band-power feature vectors
- Per-patient Leave-One-Out cross-validation with Linear SVM
- Classification via ROC AUC + permutation-based significance testing
"""

import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data_loading import config
from src.pipelines.command_following import CommandFollowingAnalysis, CommandPair, compute_log_band_power
from src.utils.signal_processing import normalize_channel_names

logger = logging.getLogger(__name__)

N_PERMUTATIONS = 1000
MIN_PAIRS_FOR_SVM = 4

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_feature_vector(
    segment: mne.Epochs,
    bands: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Extract a flat feature vector from a single-epoch segment.

    Features = log10 band power per channel per band, concatenated.
    Shape: (n_channels * n_bands,)
    """
    band_power = compute_log_band_power(segment, bands)
    return np.concatenate([band_power[band] for band in sorted(bands)])


def build_feature_matrix(
    pairs: List[CommandPair],
    bands: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix X and label vector y from keep/stop pairs.

    Each pair contributes TWO rows: one for keep (label=1) and one for stop (label=0).

    Returns:
        X: (2 * n_pairs, n_features) feature matrix
        y: (2 * n_pairs,) binary labels
        trial_ids: (2 * n_pairs,) trial ID for each row (for LOO grouping)
    """
    features, labels, trial_ids = [], [], []

    for pair in pairs:
        keep_feat = extract_feature_vector(pair.keep, bands)
        stop_feat = extract_feature_vector(pair.stop, bands)

        features.extend([keep_feat, stop_feat])
        labels.extend([1, 0])  # 1 = keep (motor imagery), 0 = stop (rest)
        trial_ids.extend([pair.trial_id, pair.trial_id])

    return np.array(features), np.array(labels), trial_ids


# ---------------------------------------------------------------------------
# Leave-One-Out SVM Classification
# ---------------------------------------------------------------------------


def run_loo_svm(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Run Leave-One-Out cross-validation with a Linear SVM.

    Returns:
        y_scores: (n_samples,) decision function scores per fold
        auc: ROC AUC score
        accuracy: overall classification accuracy
        mean_coefs: (n_features,) mean SVM weight vector across folds
    """
    loo = LeaveOneOut()
    y_scores = np.zeros(len(y))
    coefs = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        svm = LinearSVC(dual="auto", random_state=42)
        svm.fit(X_train_s, y_train)
        y_scores[test_idx] = svm.decision_function(X_test_s)
        coefs.append(svm.coef_.ravel())

    auc = roc_auc_score(y, y_scores)
    y_pred = (y_scores > 0).astype(int)
    accuracy = np.mean(y_pred == y)
    mean_coefs = np.mean(coefs, axis=0)

    return y_scores, auc, accuracy, mean_coefs


# ---------------------------------------------------------------------------
# Permutation Testing
# ---------------------------------------------------------------------------


def _run_single_permutation(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y)
    _, perm_auc, _, _ = run_loo_svm(X, y_perm)
    return perm_auc


def permutation_test_auc(
    X: np.ndarray,
    y: np.ndarray,
    observed_auc: float,
    n_permutations: int = N_PERMUTATIONS,
    rng_seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Compute empirical p-value for observed AUC via label permutation.

    Shuffles labels, re-runs LOO-SVM, and counts how often the permuted.
    """
    master_rng = np.random.default_rng(rng_seed)
    # Generate unique seeds for each permutation so parallel workers have independent RNG streams
    seeds = master_rng.integers(0, 2**32 - 1, size=n_permutations)

    n_jobs = multiprocessing.cpu_count()
    logger.debug("Running %d permutations across %d cores...", n_permutations, n_jobs)

    perm_aucs = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_run_single_permutation)(X, y, seed) for seed in seeds)
    perm_aucs = np.array(perm_aucs)

    p_value = (np.sum(perm_aucs >= observed_auc) + 1) / (n_permutations + 1)
    return p_value, perm_aucs


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------


class CommandFollowingClaassen(CommandFollowingAnalysis):
    """Claassen-inspired SVM classification for command following."""

    def __init__(self, n_permutations: int = N_PERMUTATIONS, **kwargs):
        super().__init__(roi_channels=config.CLINICAL_20, **kwargs)
        self.n_permutations = n_permutations
        self.svm_results: Optional[Dict[str, Any]] = None

    def _select_channels(self, epochs: mne.Epochs) -> None:
        """Use all available CLINICAL_20 channels instead of just C3/C4/Cz."""
        available_chs = epochs.ch_names
        normalized_names = normalize_channel_names(available_chs)
        clean_map = {clean.upper(): orig for orig, clean in zip(available_chs, normalized_names)}

        picks = []
        missing = []
        for target in self.roi_channels:
            if target in available_chs:
                picks.append(target)
            elif target.upper() in clean_map:
                picks.append(clean_map[target.upper()])
            else:
                missing.append(target)

        if missing:
            logger.warning(f"Missing CLINICAL_20 channels: {missing}")

        if not picks:
            raise ValueError(f"No CLINICAL_20 channels found in data. Available: {available_chs}")

        epochs.pick(picks)

    def analyze(self, alpha: float = 0.05, **kwargs) -> pd.DataFrame:
        """Run SVM-based classification instead of ERD threshold testing."""
        if len(self.pairs) < MIN_PAIRS_FOR_SVM:
            logger.warning(
                f"Only {len(self.pairs)} pairs available (min {MIN_PAIRS_FOR_SVM}). Cannot run SVM. Returning empty.",
            )
            self.svm_results = self._empty_results("Not enough pairs")
            self.erd_results = pd.DataFrame()
            return self.erd_results

        sides = sorted({ct.split("_")[0] for ct in self.command_types})
        results_per_side = []
        for side in sides:
            side_pairs = [p for p in self.pairs if p.side == side]
            if not side_pairs:
                continue
            results_per_side.append(self._classify_side(side, side_pairs, alpha))

        if not results_per_side:
            self.svm_results = self._empty_results("No pairs found for any side")
            self.erd_results = pd.DataFrame()
            return self.erd_results

        self.svm_results = self._aggregate_results(results_per_side, alpha)
        self.erd_results = self.svm_results["details"]
        return self.erd_results

    def _classify_side(
        self,
        side: str,
        pairs: List[CommandPair],
        alpha: float,
    ) -> Dict[str, Any]:
        """Run feature extraction → LOO SVM → permutation test for one side."""
        logger.info("SVM classification for '%s' command (%d pairs)...", side, len(pairs))

        X, y, trial_ids = build_feature_matrix(pairs, self.bands)
        y_scores, auc, accuracy, mean_coefs = run_loo_svm(X, y)

        logger.info("  ROC AUC: %.3f  |  Accuracy: %.1f%%", auc, accuracy * 100)

        p_value, perm_aucs = permutation_test_auc(X, y, auc, n_permutations=self.n_permutations)
        logger.info("  Permutation p-value: %.4f (n_perms=%d)", p_value, self.n_permutations)

        n_pairs = len(pairs)
        chance_level = stats.binom.ppf(0.95, n_pairs * 2, 0.5) / (n_pairs * 2)

        return {
            "side": side,
            "n_pairs": n_pairs,
            "n_features": X.shape[1],
            "n_channels": len(pairs[0].keep.ch_names),
            "channels_used": list(pairs[0].keep.ch_names),
            "auc": auc,
            "accuracy": accuracy,
            "chance_level": chance_level,
            "p_value_perm": p_value,
            "significant": p_value < alpha,
            "y_true": y,
            "y_scores": y_scores,
            "perm_aucs": perm_aucs,
            "svm_coefs": mean_coefs,
        }

    def _aggregate_results(self, side_results: List[Dict[str, Any]], alpha: float) -> Dict[str, Any]:
        """Combine per-side results into the final classification output."""
        any_significant = any(r["significant"] for r in side_results)

        detail_keys = [
            "side",
            "n_pairs",
            "n_features",
            "n_channels",
            "auc",
            "accuracy",
            "chance_level",
            "p_value_perm",
            "significant",
        ]

        return {
            "cmd_status": "CMD+" if any_significant else "CMD-",
            "method": "svm_claassen",
            "alpha": alpha,
            "n_permutations": self.n_permutations,
            "side_results": side_results,
            "details": pd.DataFrame([{k: r[k] for k in detail_keys} for r in side_results]),
        }

    def _empty_results(self, reason: str) -> Dict[str, Any]:
        return {
            "cmd_status": f"CMD- ({reason})",
            "method": "svm_claassen",
            "alpha": np.nan,
            "n_permutations": self.n_permutations,
            "side_results": [],
            "details": pd.DataFrame(),
        }

    def generate_summary(self, **kwargs) -> Dict[str, Any]:
        """Generate summary dict from SVM classification results."""
        if self.svm_results is None:
            return {
                "cmd_status": "ERROR: No results",
                "method": "svm_claassen",
                "n_pairs": 0,
                "left_pairs": 0,
                "right_pairs": 0,
            }

        n_pairs = len(self.pairs)
        left_pairs = sum(1 for p in self.pairs if p.side == "left")
        right_pairs = sum(1 for p in self.pairs if p.side == "right")

        summary = {
            "cmd_status": self.svm_results["cmd_status"],
            "method": "svm_claassen",
            "n_pairs": n_pairs,
            "left_pairs": left_pairs,
            "right_pairs": right_pairs,
            "n_permutations": self.svm_results["n_permutations"],
            "side_results": self.svm_results.get("side_results", []),
        }

        return summary
