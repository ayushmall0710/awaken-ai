"""Tests for adaptive Morlet TFR cropping."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.viz.language_plots import plot_itpc_results


def test_morlet_tfr_cropping_is_applied(tmp_path):
    """Verify that plot_itpc_results crops the TFR object based on n_cycles."""
    # Create mock AverageTFR
    n_ch, n_freqs, n_times = 1, 10, 1000
    sfreq = 100.0
    times = np.arange(n_times) / sfreq
    freqs = np.linspace(0.5, 5.0, n_freqs)
    data = np.random.rand(n_ch, n_freqs, n_times)

    # We need a real AverageTFR or a good mock
    # Using a mock that records calls to crop()
    itc = MagicMock()
    itc.data = data
    itc.freqs = freqs
    itc.times = times

    # mock copy() to return another mock
    itc_plot = MagicMock()
    itc_plot.data = data
    itc_plot.freqs = freqs
    itc_plot.times = times
    itc.copy.return_value = itc_plot

    # n_cycles = 2f logic
    n_cycles = freqs * 2.0
    # expected crop_val = max( (f*2) / (2*f) ) = 1.0

    metrics = {"freq_sentence_hz": 0.78, "freq_phrase_hz": 1.56, "freq_word_hz": 3.125}

    plot_itpc_results(itc, "TEST", str(tmp_path), metrics, n_cycles=n_cycles)

    # Check if crop was called on itc_plot
    assert itc_plot.crop.called
    args, kwargs = itc_plot.crop.call_args
    assert pytest.approx(kwargs["tmin"], 0.01) == 1.0
    assert pytest.approx(kwargs["tmax"], 0.01) == times[-1] - 1.0


def test_morlet_tfr_cropping_no_cycles_no_crop(tmp_path):
    """Verify that no cropping occurs if n_cycles is None."""
    itc = MagicMock()
    itc.data = np.random.rand(1, 10, 100)
    itc.freqs = np.linspace(0.5, 5.0, 10)
    itc.times = np.linspace(0, 10, 100)

    itc_plot = MagicMock()
    itc_plot.data = itc.data
    itc_plot.freqs = itc.freqs
    itc_plot.times = itc.times
    itc.copy.return_value = itc_plot

    metrics = {"freq_sentence_hz": 1.0, "freq_phrase_hz": 2.0, "freq_word_hz": 3.0}

    plot_itpc_results(itc, "TEST", str(tmp_path), metrics, n_cycles=None)

    assert not itc_plot.crop.called
