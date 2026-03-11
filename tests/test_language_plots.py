"""Tests for language_plots visualization functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from src.viz.language_plots import (
    plot_focus_comparison_bar,
    plot_itpc_channels_horizontal,
    plot_itpc_spectrum,
    plot_itpc_topomap,
)


@pytest.fixture
def fake_dft_data():
    """Minimal DFT spectrum: 7 channels, ~251 frequency bins at 256 Hz."""
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"]
    sfreq = 256.0
    freqs = np.fft.rfftfreq(500, d=1.0 / sfreq)
    itpc_spectrum = np.full((len(ch_names), len(freqs)), 0.05)
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")
    return itpc_spectrum, freqs, info, ch_names


@pytest.fixture
def mock_info():
    """MNE Info with a full LH+RH channel set and standard montage."""
    ch_names = ["F7", "T7", "P7", "F3", "C3", "P3", "Fz", "Cz", "Pz", "F8", "T8", "F4", "C4"]
    info = mne.create_info(ch_names, sfreq=256.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


# --- plot_itpc_spectrum ---


def test_plot_itpc_spectrum_saves_file(fake_dft_data):
    """plot_itpc_spectrum saves a PNG to output_dir."""
    itpc_spectrum, freqs, _, _ = fake_dft_data
    metrics = {
        "dft_p_word": 0.001,
        "dft_p_phrase": 0.04,
        "dft_p_sentence": 0.03,
        "itpc_word": 0.12,
        "itpc_phrase": 0.07,
        "itpc_sentence": 0.06,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_itpc_spectrum(itpc_spectrum, freqs, "CON008", tmpdir, metrics)
        assert Path(path).exists()
        assert Path(path).suffix == ".png"


def test_plot_itpc_spectrum_with_focus_label(tmp_path):
    """plot_itpc_spectrum encodes focus_label in the output filename."""
    freqs = np.linspace(0, 10, 100)
    itpc_spectrum = np.random.rand(10, 100)
    metrics = {
        "itpc_sentence": 0.2,
        "itpc_phrase": 0.1,
        "itpc_word": 0.05,
        "freq_sentence_hz": 0.78,
        "freq_phrase_hz": 1.56,
        "freq_word_hz": 3.125,
        "dft_p_sentence": 0.01,
        "dft_p_phrase": 0.05,
        "dft_p_word": 0.5,
    }
    out_path = plot_itpc_spectrum(
        itpc_spectrum=itpc_spectrum,
        freqs=freqs,
        patient_id="CON008",
        output_dir=tmp_path,
        metrics=metrics,
        method_label="DFT",
        focus_label="Optimal",
    )
    assert out_path.exists()
    assert "optimal" in out_path.name.lower()


# --- plot_itpc_topomap ---


def test_plot_itpc_topomap_saves_file(fake_dft_data):
    """plot_itpc_topomap saves a PNG named after the target frequency label."""
    itpc_spectrum, freqs, info, _ = fake_dft_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_itpc_topomap(
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


def test_plot_itpc_topomap_respects_vlim(fake_dft_data):
    """vlim parameter is accepted without error."""
    itpc_spectrum, freqs, info, _ = fake_dft_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_itpc_topomap(
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


def test_plot_itpc_topomap_with_highlight_channels(mock_info, tmp_path):
    """highlight_channels parameter is accepted without TypeError."""
    n_ch = len(mock_info.ch_names)
    freqs = np.linspace(0, 10, 100)
    itpc_spectrum = np.random.rand(n_ch, 100)
    out_path = plot_itpc_topomap(
        itpc_spectrum=itpc_spectrum,
        freqs=freqs,
        info=mock_info,
        target_freq=0.78,
        label="Sentence",
        patient_id="CON008",
        output_dir=tmp_path,
        highlight_channels=["F7", "T7"],
    )
    assert out_path.exists()


# --- plot_focus_comparison_bar ---


def test_plot_focus_comparison_bar_saves_file(tmp_path):
    """plot_focus_comparison_bar saves a PNG with the patient ID in the filename."""
    df = pd.DataFrame(
        {
            "focus": ["clinical", "lh", "rh", "optimal"],
            "itpc_comprehension": [0.2, 0.15, 0.1, 0.3],
            "dft_p_comprehension": [0.01, 0.1, 0.5, 0.001],
        }
    )
    out_path = plot_focus_comparison_bar(df=df, patient_id="CON008", output_dir=tmp_path)
    assert out_path.exists()
    assert out_path.name == "CON008_lang_focus_comparison.png"


# --- plot_itpc_channels_horizontal ---


def test_plot_itpc_channels_horizontal_morlet(tmp_path):
    """Horizontal bar chart accepts 3D Morlet data and saves a file."""
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"]
    itpc_data = np.random.rand(len(ch_names), 50, 100)
    freqs = np.linspace(0.5, 5, 50)
    path = plot_itpc_channels_horizontal(
        itpc_data=itpc_data,
        ch_names=ch_names,
        patient_id="TEST",
        output_dir=tmp_path,
        n_trials=20,
        freqs=freqs,
        sentence_band=(0.7, 0.9),
        phrase_band=(1.4, 1.7),
        word_band=(2.8, 3.4),
        method_label="Morlet",
    )
    assert Path(path).exists()
    assert "morlet" in Path(path).name.lower()


def test_plot_itpc_channels_horizontal_dft(tmp_path):
    """Horizontal bar chart accepts 2D DFT data and saves a file."""
    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"]
    itpc_data = np.random.rand(len(ch_names), 500)
    freqs = np.linspace(0.1, 10, 500)
    path = plot_itpc_channels_horizontal(
        itpc_data=itpc_data,
        ch_names=ch_names,
        patient_id="TEST",
        output_dir=tmp_path,
        n_trials=20,
        freqs=freqs,
        sentence_band=(0.7, 0.9),
        phrase_band=(1.4, 1.7),
        word_band=(2.8, 3.4),
        method_label="DFT",
    )
    assert Path(path).exists()
    assert "dft" in Path(path).name.lower()
