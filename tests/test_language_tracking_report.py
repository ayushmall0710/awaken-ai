"""Tests for LanguageTrackingReport."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.reports.language_tracking_report import LanguageTrackingReport


@pytest.fixture
def mock_pipeline():
    """Minimal mock LanguageTrackingAnalysis after analyze() has run."""
    pipeline = MagicMock()
    pipeline.patient_id = "CON008"

    pipeline.results = pd.DataFrame(
        [
            {
                "patient_id": "CON008",
                "n_trials": 20,
                "itpc_word": 0.14,
                "itpc_phrase": 0.08,
                "itpc_sentence": 0.07,
                "itpc_comprehension_combined": 0.075,
                "ratio_cognitive_acoustic": 0.54,
                "lh_itpc_word": 0.15,
                "lh_itpc_phrase": 0.09,
                "lh_itpc_sentence": 0.08,
                "rh_itpc_word": 0.13,
                "rh_itpc_phrase": 0.07,
                "rh_itpc_sentence": 0.06,
                "lateralization_index_word": 0.07,
                "lateralization_index_phrase": 0.12,
                "lateralization_index_sentence": 0.14,
                "lateralization_index_comprehension": 0.13,
                "morlet_itpc_word": 0.11,
                "morlet_itpc_phrase": 0.06,
                "morlet_itpc_sentence": 0.05,
                "dft_p_word": 0.001,
                "dft_p_phrase": 0.032,
                "dft_p_sentence": 0.028,
                "dft_p_comprehension": 0.015,
                "morlet_p_word": 0.01,
                "morlet_p_phrase": 0.04,
                "morlet_p_sentence": 0.02,
                "morlet_p_comprehension": 0.03,
            }
        ]
    )

    n_ch, n_freqs = 7, 300
    pipeline._dft_spectrum_full = np.random.rand(n_ch, n_freqs) * 0.1
    pipeline._dft_freqs = np.linspace(0.01, 4.0, n_freqs)
    pipeline._dft_ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"]
    pipeline._dft_info = mne.create_info(["Fp1", "Fp2", "F3", "F4", "Fz", "T7", "T8"], sfreq=256.0, ch_types="eeg")
    pipeline._morlet_itc = None  # skip Morlet plot in tests
    return pipeline


def test_generate_creates_html_file(mock_pipeline):
    """generate() writes an HTML file to output_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reports.language_tracking_report.plot_itpc_spectrum", return_value=Path(tmpdir) / "s.png"),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            path = rpt.generate()
            assert Path(path).exists()
            assert Path(path).suffix == ".html"


def test_generate_html_contains_key_sections(mock_pipeline):
    """generate() HTML contains entrainment table and lateralization section."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reports.language_tracking_report.plot_itpc_spectrum", return_value=Path(tmpdir) / "s.png"),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            path = rpt.generate()
            html = Path(path).read_text()
    assert "lateralization" in html.lower()
    assert "<table" in html
    assert "CON008" in html


def test_build_session_html_returns_details_element(mock_pipeline):
    """build_session_html() returns a collapsible <details> fragment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reports.language_tracking_report.plot_itpc_spectrum", return_value=Path(tmpdir) / "s.png"),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            fragment = rpt.build_session_html()
    assert "<details" in fragment
    assert "session-wrapper" in fragment


def test_report_missing_patient_id_raises(mock_pipeline):
    """Report raises ValueError if pipeline has no patient_id."""
    mock_pipeline.patient_id = None
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="patient_id"):
            LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))


def test_report_html_contains_morlet_section(mock_pipeline):
    """Report HTML contains a Morlet ITPC section when _morlet_itc is set."""
    morlet_itc = MagicMock()
    n_ch, n_freqs, n_times = 7, 60, 50
    morlet_itc.data = np.random.rand(n_ch, n_freqs, n_times) * 0.1
    morlet_itc.freqs = np.logspace(np.log10(0.5), np.log10(5.0), num=n_freqs)
    mock_pipeline._morlet_itc = morlet_itc
    # mock_pipeline.results already has morlet_p_* from fixture

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("src.reports.language_tracking_report.plot_itpc_results"),
            patch("src.reports.language_tracking_report.plot_itpc_spectrum", return_value=Path(tmpdir) / "s.png"),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            path = rpt.generate()
            html = Path(path).read_text()
    assert "ITPC Topographic Maps (Morlet)" in html
