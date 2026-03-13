"""Tests for LanguageTrackingReport."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.pipelines.language_tracking import LanguageConfig
from src.reports.language_tracking_report import LanguageTrackingReport


@pytest.fixture
def mock_pipeline():
    """Minimal mock LanguageTrackingAnalysis after analyze() has run."""
    pipeline = MagicMock()
    pipeline.patient_id = "CON008"
    pipeline.cfg = LanguageConfig()

    pipeline.results = pd.DataFrame(
        [
            {
                "focus": "clinical",
                "patient_id": "CON008",
                "n_trials": 20,
                "itpc_word": 0.14,
                "itpc_phrase": 0.08,
                "itpc_sentence": 0.07,
                "itpc_comprehension": 0.075,
                "ratio_cognitive_acoustic": 0.54,
                "channels": ["Fp1", "Fp2", "F3", "F4", "Fz"],
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
            },
            {
                "focus": "lh",
                "patient_id": "CON008",
                "itpc_word": 0.15,
                "itpc_phrase": 0.09,
                "itpc_sentence": 0.08,
                "itpc_comprehension": 0.085,
                "dft_p_comprehension": 0.01,
            },
            {
                "focus": "rh",
                "patient_id": "CON008",
                "itpc_word": 0.13,
                "itpc_phrase": 0.07,
                "itpc_sentence": 0.06,
                "itpc_comprehension": 0.065,
                "dft_p_comprehension": 0.05,
            },
            {
                "focus": "optimal",
                "patient_id": "CON008",
                "itpc_word": 0.18,
                "itpc_phrase": 0.12,
                "itpc_sentence": 0.11,
                "itpc_comprehension": 0.115,
                "dft_p_comprehension": 0.001,
                "channels": ["F3", "T7"],
            },
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
            patch(
                "src.reports.language_tracking_report.plot_itpc_spectrum_stacked", return_value=Path(tmpdir) / "s.png"
            ),
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
            patch(
                "src.reports.language_tracking_report.plot_itpc_spectrum_stacked", return_value=Path(tmpdir) / "s.png"
            ),
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
            patch(
                "src.reports.language_tracking_report.plot_itpc_spectrum_stacked", return_value=Path(tmpdir) / "s.png"
            ),
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
            patch("src.reports.language_tracking_report.plot_itpc_results", return_value=Path(tmpdir) / "m.png"),
            patch(
                "src.reports.language_tracking_report.plot_itpc_spectrum_stacked", return_value=Path(tmpdir) / "s.png"
            ),
            patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=Path(tmpdir) / "t.png"),
        ):
            rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=Path(tmpdir))
            path = rpt.generate()
            html = Path(path).read_text()
    assert "ITPC Topographic Maps (Morlet)" in html


def test_build_entrainment_table_has_all_focuses(mock_pipeline, tmp_path):
    """_build_entrainment_table renders a row for each focus in results."""
    rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=tmp_path)
    html = rpt._build_entrainment_table()
    assert "clinical" in html.lower()
    assert "lh" in html.lower()
    assert html.count("<tr") >= 3  # header + at least 2 data rows


def test_build_optimal_focus_section_shows_channels(mock_pipeline, tmp_path):
    """_build_optimal_focus_section lists the optimal cluster channels."""
    rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=tmp_path)
    html = rpt._build_optimal_focus_section(plot_paths={})
    assert "Optimal Focus" in html
    assert "F3" in html and "T7" in html


def test_build_overview_cards_includes_optimal_in_significance(mock_pipeline, tmp_path):
    """_build_overview_cards includes 'OPTIMAL' in the significance badge if significant."""
    # Ensure optimal focus is significant in mock data (it already is p=0.001 in fixture)
    rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=tmp_path)
    html = rpt._build_overview_cards()
    # Currently it only checks ["clinical", "lh", "rh"]
    assert "OPTIMAL" in html


def test_save_plots_conditional_highlighting_for_optimal_word(mock_pipeline, tmp_path):
    """_save_plots passes highlight_channels=None for Word rate in Optimal focus."""
    with (
        patch("src.reports.language_tracking_report.plot_itpc_spectrum_stacked", return_value=tmp_path / "s.png"),
        patch("src.reports.language_tracking_report.plot_itpc_topomap", return_value=tmp_path / "t.png") as mock_topo,
        patch("src.reports.language_tracking_report.plot_focus_comparison_bar", return_value=tmp_path / "c.png"),
    ):
        rpt = LanguageTrackingReport(mock_pipeline, session_id="sess_01", output_dir=tmp_path)
        rpt._save_plots()

    # Verify calls to plot_itpc_topomap
    # Signature: (spectrum, freqs, info, freq, label, patient_id, output_dir, vlim, highlight_channels, ...)
    # highlight_channels is the 9th argument (index 8)

    optimal_word_called = False
    for call in mock_topo.call_args_list:
        label = call.args[4]
        highlight = call.kwargs.get("highlight_channels")

        if label == "Word" and highlight is not None:
            pytest.fail(f"Word rate topomap should not have highlighted channels, but got {highlight}")

        if label == "Word":
            optimal_word_called = True

    assert optimal_word_called, "Word rate topomap was never called"


def test_morlet_tfr_cropping_is_applied(tmp_path):
    """Verify that plot_itpc_results crops the TFR object based on n_cycles."""
    from src.viz.language_plots import plot_itpc_results

    # Create mock AverageTFR
    n_ch, n_freqs, n_times = 1, 10, 1000
    sfreq = 100.0
    times = np.arange(n_times) / sfreq
    freqs = np.linspace(0.5, 5.0, n_freqs)
    data = np.random.rand(n_ch, n_freqs, n_times)

    # We need a mock that records calls to copy() and crop()
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
    _, kwargs = itc_plot.crop.call_args
    assert pytest.approx(kwargs["tmin"], 0.01) == 1.0
    assert pytest.approx(kwargs["tmax"], 0.01) == times[-1] - 1.0


def test_morlet_tfr_cropping_no_cycles_no_crop(tmp_path):
    """Verify that no cropping occurs if n_cycles is None."""
    from src.viz.language_plots import plot_itpc_results

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
