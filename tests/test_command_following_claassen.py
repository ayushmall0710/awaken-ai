"""
Tests for Claassen SVM Command Following pipeline.

Tests the standalone functions (feature extraction, LOO-SVM, permutation)
and the pipeline class (channel selection, analyze, generate_summary),
using synthetic EEG data with known separable/non-separable patterns.
"""

import mne
import numpy as np
import pytest

from src.data_loading import config
from src.pipelines.command_following import CommandFollowingAnalysis, CommandPair
from src.pipelines.command_following_claassen import (
    CommandFollowingClaassen,
    build_feature_matrix,
    extract_feature_vector,
    permutation_test_auc,
    run_loo_svm,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------

SFREQ = 100
DURATION_SEC = 2
N_SAMPLES = SFREQ * DURATION_SEC
CHANNELS = ["C3", "C4", "Cz", "Fz", "Pz"]
BANDS = {"Alpha": (8, 13), "Beta": (13, 30)}


def _make_epoch(channels=CHANNELS, amplitude=1.0, freq=10.0, noise=0.05):
    """Create a single-epoch MNE Epochs object with a sine + noise signal."""
    info = mne.create_info(ch_names=channels, sfreq=SFREQ, ch_types="eeg")
    t = np.linspace(0, DURATION_SEC, N_SAMPLES)
    data = np.array([np.sin(2 * np.pi * freq * t) * amplitude + np.random.randn(N_SAMPLES) * noise for _ in channels])
    return mne.EpochsArray(data[np.newaxis], info, tmin=0, verbose=False)


def _make_pair(side="left", trial_id="t1", keep_amp=1.0, stop_amp=2.0):
    """Create a CommandPair with keep/stop at different amplitudes."""
    return CommandPair(
        keep=_make_epoch(amplitude=keep_amp),
        stop=_make_epoch(amplitude=stop_amp),
        side=side,
        trial_id=trial_id,
        keep_start=0.0,
        stop_start=10.0,
    )


def _make_separable_pairs(n=8, keep_amp=1.0, stop_amp=3.0, side="left"):
    """Create N pairs where keep and stop have clearly different amplitudes."""
    return [_make_pair(side=side, trial_id=f"t{i}", keep_amp=keep_amp, stop_amp=stop_amp) for i in range(n)]


def _make_random_pairs(n=8, side="left"):
    """Create N pairs where keep and stop are indistinguishable (same amplitude)."""
    return [_make_pair(side=side, trial_id=f"t{i}", keep_amp=1.0, stop_amp=1.0) for i in range(n)]


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_vector_shape(self):
        """Feature vector should have n_channels * n_bands elements."""
        epoch = _make_epoch()
        vec = extract_feature_vector(epoch, BANDS)
        assert vec.shape == (len(CHANNELS) * len(BANDS),)

    def test_vector_dtype(self):
        """Features should be finite float values (log10 power)."""
        epoch = _make_epoch()
        vec = extract_feature_vector(epoch, BANDS)
        assert np.issubdtype(vec.dtype, np.floating)
        assert np.all(np.isfinite(vec))

    def test_higher_amplitude_gives_higher_power(self):
        """Doubling amplitude → higher log10 power across all features."""
        low = extract_feature_vector(_make_epoch(amplitude=1.0, noise=0.01), BANDS)
        high = extract_feature_vector(_make_epoch(amplitude=2.0, noise=0.01), BANDS)
        assert np.mean(high) > np.mean(low)

    def test_different_channel_counts_give_different_shapes(self):
        """3 channels should give different feature length than 5 channels."""
        vec3 = extract_feature_vector(_make_epoch(channels=["C3", "C4", "Cz"]), BANDS)
        vec5 = extract_feature_vector(_make_epoch(channels=CHANNELS), BANDS)
        assert vec3.shape[0] == 3 * len(BANDS)
        assert vec5.shape[0] == 5 * len(BANDS)


class TestBuildFeatureMatrix:
    def test_matrix_shape(self):
        """Each pair produces 2 rows (keep + stop)."""
        pairs = _make_separable_pairs(n=6)
        X, y, trial_ids = build_feature_matrix(pairs, BANDS)
        assert X.shape == (12, len(CHANNELS) * len(BANDS))
        assert y.shape == (12,)
        assert len(trial_ids) == 12

    def test_labels_alternate_keep_stop(self):
        """Labels should be [1, 0, 1, 0, ...] — keep=1, stop=0."""
        pairs = _make_separable_pairs(n=4)
        _, y, _ = build_feature_matrix(pairs, BANDS)
        expected = np.tile([1, 0], 4)
        np.testing.assert_array_equal(y, expected)

    def test_trial_ids_paired(self):
        """Each pair's trial_id should appear exactly twice (keep and stop rows)."""
        pairs = _make_separable_pairs(n=5)
        _, _, trial_ids = build_feature_matrix(pairs, BANDS)
        for i in range(5):
            assert trial_ids[2 * i] == trial_ids[2 * i + 1] == f"t{i}"


# ---------------------------------------------------------------------------
# LOO-SVM Tests
# ---------------------------------------------------------------------------


class TestLooSvm:
    def test_separable_data_high_auc(self):
        """Clearly separable classes → AUC should be well above chance."""
        pairs = _make_separable_pairs(n=10, keep_amp=0.5, stop_amp=4.0)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, auc, accuracy = run_loo_svm(X, y)
        assert auc > 0.7
        assert accuracy > 0.6

    def test_random_data_near_chance(self):
        """Identical keep/stop → AUC should not be consistently high like separable data."""
        np.random.seed(123)
        pairs = _make_random_pairs(n=10)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, auc, accuracy = run_loo_svm(X, y)
        # With identical classes, AUC can swing widely on small samples,
        # but should never reach the levels of truly separable data (>0.9)
        assert auc < 0.9

    def test_output_shapes(self):
        """y_scores should have one score per sample."""
        pairs = _make_separable_pairs(n=6)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        y_scores, _, _ = run_loo_svm(X, y)
        assert y_scores.shape == y.shape

    def test_accuracy_is_fraction(self):
        """Accuracy should be between 0 and 1."""
        pairs = _make_separable_pairs(n=6)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, _, accuracy = run_loo_svm(X, y)
        assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    def test_separable_gives_low_p(self):
        """Clearly separable data → permutation p should be low."""
        np.random.seed(42)
        pairs = _make_separable_pairs(n=10, keep_amp=0.5, stop_amp=5.0)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, observed_auc, _ = run_loo_svm(X, y)
        p = permutation_test_auc(X, y, observed_auc, n_permutations=50)
        assert p < 0.2  # with only 50 perms, can't be super precise

    def test_random_gives_high_p(self):
        """Identical classes → permutation p should be high (non-significant)."""
        np.random.seed(42)
        pairs = _make_random_pairs(n=8)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, observed_auc, _ = run_loo_svm(X, y)
        p = permutation_test_auc(X, y, observed_auc, n_permutations=50)
        assert p > 0.05

    def test_p_value_bounds(self):
        """p-value must be in (0, 1] — never exactly 0 due to +1 correction."""
        pairs = _make_separable_pairs(n=6)
        X, y, _ = build_feature_matrix(pairs, BANDS)
        _, auc, _ = run_loo_svm(X, y)
        p = permutation_test_auc(X, y, auc, n_permutations=20)
        assert 0 < p <= 1.0


# ---------------------------------------------------------------------------
# Channel Selection Tests
# ---------------------------------------------------------------------------


class TestChannelSelection:
    def test_picks_matching_channels(self):
        """Should select channels present in both CLINICAL_20 and data."""
        pipeline = CommandFollowingClaassen()
        epochs = _make_epoch(channels=["C3", "C4", "Fz", "DC1", "AUX2"])
        pipeline._select_channels(epochs)
        assert set(epochs.ch_names) == {"C3", "C4", "Fz"}  # DC1, AUX2 dropped

    def test_handles_prefixed_channels(self):
        """Should match 'EEG C3' to CLINICAL_20 'C3' via normalize_channel_names."""
        pipeline = CommandFollowingClaassen()
        info = mne.create_info(ch_names=["EEG C3", "EEG C4", "EEG Fz", "DC1"], sfreq=SFREQ, ch_types="eeg")
        data = np.random.randn(1, 4, N_SAMPLES)
        epochs = mne.EpochsArray(data, info, tmin=0, verbose=False)
        pipeline._select_channels(epochs)
        assert len(epochs.ch_names) == 3

    def test_raises_on_no_match(self):
        """Should raise ValueError if zero CLINICAL_20 channels found."""
        pipeline = CommandFollowingClaassen()
        epochs = _make_epoch(channels=["DC1", "DC2", "AUX1"])
        with pytest.raises(ValueError, match="No CLINICAL_20"):
            pipeline._select_channels(epochs)


# ---------------------------------------------------------------------------
# Pipeline Analyze + Summary Tests
# ---------------------------------------------------------------------------


class TestClaassenPipeline:
    def test_cmd_positive_with_separable_data(self):
        """Pipeline should classify as CMD+ when keep/stop are clearly different."""
        pipeline = CommandFollowingClaassen(n_permutations=50)
        pipeline.pairs = _make_separable_pairs(n=10, keep_amp=0.5, stop_amp=5.0, side="left")
        pipeline.pairs += _make_separable_pairs(n=10, keep_amp=0.5, stop_amp=5.0, side="right")

        result_df = pipeline.analyze(alpha=0.05)

        assert not result_df.empty
        assert pipeline.svm_results["cmd_status"] == "CMD+"

    def test_cmd_negative_with_random_data(self):
        """Pipeline should classify as CMD- when data is indistinguishable."""
        pipeline = CommandFollowingClaassen(n_permutations=50)
        pipeline.pairs = _make_random_pairs(n=8, side="left")
        pipeline.pairs += _make_random_pairs(n=8, side="right")

        pipeline.analyze(alpha=0.05)

        assert pipeline.svm_results["cmd_status"] == "CMD-"

    def test_too_few_pairs_gives_empty(self):
        """Should return CMD- with reason if fewer than MIN_PAIRS_FOR_SVM pairs."""
        pipeline = CommandFollowingClaassen(n_permutations=10)
        pipeline.pairs = _make_separable_pairs(n=2)

        result_df = pipeline.analyze(alpha=0.05)

        assert result_df.empty
        assert "Not enough pairs" in pipeline.svm_results["cmd_status"]

    def test_details_dataframe_columns(self):
        """Result details DataFrame should have the expected columns."""
        pipeline = CommandFollowingClaassen(n_permutations=10)
        pipeline.pairs = _make_separable_pairs(n=6, side="left")

        result_df = pipeline.analyze(alpha=0.05)

        expected_cols = {
            "side",
            "n_pairs",
            "n_features",
            "n_channels",
            "auc",
            "accuracy",
            "chance_level",
            "p_value_perm",
            "significant",
        }
        assert expected_cols == set(result_df.columns)

    def test_sides_from_command_types(self):
        """Should iterate over sides derived from command_types, not pair data."""
        pipeline = CommandFollowingClaassen(n_permutations=10)
        pipeline.command_types = ["left_command", "right_command"]
        pipeline.pairs = _make_separable_pairs(n=6, side="left")
        # No right-side pairs → right should just be skipped, not error

        result_df = pipeline.analyze(alpha=0.05)
        assert len(result_df) == 1
        assert result_df.iloc[0]["side"] == "left"

    def test_generate_summary_structure(self):
        """Summary should contain cmd_status, method, pair counts, side_results."""
        pipeline = CommandFollowingClaassen(n_permutations=10)
        pipeline.pairs = _make_separable_pairs(n=6, side="left")
        pipeline.pairs += _make_separable_pairs(n=6, side="right")
        pipeline.analyze(alpha=0.05)

        summary = pipeline.generate_summary()

        assert summary["method"] == "svm_claassen"
        assert summary["n_pairs"] == 12
        assert summary["left_pairs"] == 6
        assert summary["right_pairs"] == 6
        assert isinstance(summary["side_results"], list)
        assert len(summary["side_results"]) == 2

    def test_generate_summary_without_analyze_returns_error(self):
        """Calling generate_summary before analyze should return an error status."""
        pipeline = CommandFollowingClaassen()
        summary = pipeline.generate_summary()
        assert "ERROR" in summary["cmd_status"]

    def test_inherits_from_command_following(self):
        """Should be a subclass of CommandFollowingAnalysis."""
        assert issubclass(CommandFollowingClaassen, CommandFollowingAnalysis)

    def test_uses_clinical_20_channels(self):
        """Default roi_channels should be CLINICAL_20."""
        pipeline = CommandFollowingClaassen()
        assert pipeline.roi_channels == config.CLINICAL_20
