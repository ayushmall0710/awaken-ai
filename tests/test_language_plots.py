"""Tests for language_plots visualization functions."""

import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest

from src.viz.language_plots import plot_dft_spectrum, plot_dft_topomap


@pytest.fixture
def fake_dft_data():
    """Minimal DFT spectrum: 7 channels, 500 frequency bins."""
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"]
    sfreq = 256.0
    n_fft = 500
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)[:500]
    itpc_spectrum = np.full((len(ch_names), len(freqs)), 0.05)
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return itpc_spectrum, freqs, info, ch_names


def test_plot_dft_spectrum_saves_file(fake_dft_data):
    """plot_dft_spectrum saves a PNG to output_dir."""
    itpc_spectrum, freqs, info, _ = fake_dft_data
    metrics = {
        "dft_p_word": 0.001,
        "dft_p_phrase": 0.04,
        "dft_p_sentence": 0.03,
        "itpc_word": 0.12,
        "itpc_phrase": 0.07,
        "itpc_sentence": 0.06,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_dft_spectrum(itpc_spectrum, freqs, "CON008", tmpdir, metrics)
        assert Path(path).exists()
        assert Path(path).suffix == ".png"


def test_plot_dft_topomap_saves_file(fake_dft_data):
    """plot_dft_topomap saves a PNG for a target frequency."""
    itpc_spectrum, freqs, info, _ = fake_dft_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_dft_topomap(
            itpc_spectrum,
            freqs,
            info,
            target_freq=0.78,
            label="Sentence",
            patient_id="CON008",
            output_dir=tmpdir,
        )
        assert Path(path).exists()
        assert "sentence" in Path(path).name.lower()


def test_plot_dft_topomap_respects_vlim(fake_dft_data):
    """vlim parameter is respected when passed."""
    itpc_spectrum, freqs, info, _ = fake_dft_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_dft_topomap(
            itpc_spectrum,
            freqs,
            info,
            target_freq=3.125,
            label="Word",
            patient_id="CON008",
            output_dir=tmpdir,
            vlim=(0.0, 0.2),
        )
        assert Path(path).exists()
