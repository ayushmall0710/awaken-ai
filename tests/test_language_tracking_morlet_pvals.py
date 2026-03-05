"""Tests for Morlet permutation p-values in LanguageTrackingAnalysis."""

import numpy as np
import pytest

from src.pipelines.language_tracking import LanguageTrackingAnalysis


@pytest.fixture
def pipeline_with_phases():
    """Pipeline instance with synthetic _morlet_phases stored."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    rng = np.random.default_rng(0)
    # Shape: (n_trials=10, n_channels=5, n_target_freqs=3)
    # axis-2 order: [0]=word, [1]=phrase, [2]=sentence
    lt._morlet_phases = rng.uniform(-np.pi, np.pi, size=(10, 5, 3))
    return lt


def test_compute_null_morlet_returns_correct_shape(pipeline_with_phases):
    """compute_trial_shuffled_null_itpc(method='morlet') returns (n_permutations,)."""
    null = pipeline_with_phases.compute_trial_shuffled_null_itpc(
        epochs=None, n_permutations=50, metric="word", seed=0, method="morlet"
    )
    assert null.shape == (50,)
    assert np.all(null >= 0) and np.all(null <= 1)


def test_compute_null_morlet_all_metrics(pipeline_with_phases):
    """All metric types work for morlet method."""
    for metric in ("word", "phrase", "sentence", "comprehension"):
        null = pipeline_with_phases.compute_trial_shuffled_null_itpc(
            epochs=None, n_permutations=20, metric=metric, seed=0, method="morlet"
        )
        assert null.shape == (20,)


def test_compute_null_morlet_raises_if_no_phases():
    """Raises ValueError when _morlet_phases is not stored."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    lt._morlet_phases = None
    with pytest.raises(ValueError, match="_morlet_phases"):
        lt.compute_trial_shuffled_null_itpc(epochs=None, n_permutations=10, metric="word", seed=0, method="morlet")


def test_compute_null_invalid_method():
    """Raises ValueError for unknown method."""
    lt = LanguageTrackingAnalysis.__new__(LanguageTrackingAnalysis)
    with pytest.raises(ValueError, match="method"):
        lt.compute_trial_shuffled_null_itpc(epochs=None, n_permutations=10, metric="word", seed=0, method="unknown")
