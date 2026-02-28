"""Tests for ENG-06: QC Report Generation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.qc_report import (
    QCDataCollector,
    QCMetricsCalculator,
    QCReportGenerator,
    _aggregate_summary,
    _build_html_table,
    _count_iclabel_artifact_types,
    _empty_qc_dataframe,
    _format_cell,
    _parse_single_ica_json,
    _safe_id,
    generate_qc_report,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ica_json(
    method: str = "infomax",
    classification_method: str = "iclabel",
    excluded: list | None = None,
    eog_components: list | None = None,
    ecg_components: list | None = None,
    muscle_components: list | None = None,
    line_noise_components: list | None = None,
    channel_noise_components: list | None = None,
) -> str:
    """Build a valid ICA summary JSON string matching ENG-03's schema."""
    return json.dumps(
        {
            "method": method,
            "classification_method": classification_method,
            "n_components": 15,
            "n_components_selected": None,
            "excluded": excluded or [1, 3],
            "eog_channels_used": ["Fp1", "Fp2"],
            "eog_components": eog_components or [1],
            "ecg_channels_used": [],
            "ecg_components": ecg_components or [],
            "muscle_components": muscle_components or [3],
            "line_noise_components": line_noise_components or [],
            "channel_noise_components": channel_noise_components or [],
            "iclabel_labels": ["brain", "eye", "brain", "muscle"],
            "iclabel_probs": None,
            "notes": [],
        }
    )


@pytest.fixture
def sample_ica_json():
    """A valid ICA summary JSON string."""
    return _make_ica_json()


@pytest.fixture
def sample_qc_df(sample_ica_json):
    """Synthetic QC DataFrame matching the ENG-03 output schema."""
    return pd.DataFrame(
        [
            {
                "patient_id": "CON008",
                "date": "2025-08-14",
                "trial_type": "language",
                "window_sec": 16.0,
                "reject_ptp_percentile": 95.0,
                "reject_ptp_threshold_uv": 120.5,
                "n_epochs_total": 72,
                "n_epochs_dropped": 4,
                "n_epochs_kept": 68,
                "drop_reason": "ENG03_PTP_GT_P95",
                "ica": sample_ica_json,
                "notes": json.dumps(["all good"]),
                "ptp_uv_p50": 45.0,
                "ptp_uv_p95": 110.0,
                "ptp_uv_p99": 150.0,
                "ptp_uv_max": 200.0,
                "ptp_uv_mean": 55.0,
            },
            {
                "patient_id": "CON008",
                "date": "2025-08-14",
                "trial_type": "oddball",
                "window_sec": 35.0,
                "reject_ptp_percentile": 95.0,
                "reject_ptp_threshold_uv": 95.3,
                "n_epochs_total": 4,
                "n_epochs_dropped": 0,
                "n_epochs_kept": 4,
                "drop_reason": None,
                "ica": sample_ica_json,
                "notes": json.dumps([]),
                "ptp_uv_p50": 30.0,
                "ptp_uv_p95": 80.0,
                "ptp_uv_p99": 90.0,
                "ptp_uv_max": 100.0,
                "ptp_uv_mean": 35.0,
            },
            {
                "patient_id": "CON009",
                "date": "2025-08-26",
                "trial_type": "language",
                "window_sec": 16.0,
                "reject_ptp_percentile": 95.0,
                "reject_ptp_threshold_uv": 130.0,
                "n_epochs_total": 72,
                "n_epochs_dropped": 8,
                "n_epochs_kept": 64,
                "drop_reason": "ENG03_PTP_GT_P95",
                "ica": _make_ica_json(classification_method="correlation", excluded=[0, 2, 4]),
                "notes": json.dumps(["noisy session"]),
                "ptp_uv_p50": 50.0,
                "ptp_uv_p95": 120.0,
                "ptp_uv_p99": 160.0,
                "ptp_uv_max": 220.0,
                "ptp_uv_mean": 60.0,
            },
        ]
    )


@pytest.fixture
def temp_qc_dir(sample_qc_df):
    """Temporary directory populated with sample QC parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # CON008 session
        con008_dir = root / "CON008" / "2025-08-14"
        con008_dir.mkdir(parents=True)
        sample_qc_df[sample_qc_df["patient_id"] == "CON008"].to_parquet(con008_dir / "eng03_qc.parquet", index=False)

        # CON009 session
        con009_dir = root / "CON009" / "2025-08-26"
        con009_dir.mkdir(parents=True)
        sample_qc_df[sample_qc_df["patient_id"] == "CON009"].to_parquet(con009_dir / "eng03_qc.parquet", index=False)

        yield root


# ── QCDataCollector tests ────────────────────────────────────────────────────


class TestQCDataCollector:
    def test_discover_files_finds_parquets(self, temp_qc_dir):
        collector = QCDataCollector(qc_dir=temp_qc_dir)
        files = collector.discover_qc_files()
        assert len(files) == 2
        assert all(f.name == "eng03_qc.parquet" for f in files)

    def test_discover_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QCDataCollector(qc_dir=Path(tmpdir))
            assert collector.discover_qc_files() == []

    def test_discover_files_missing_dir(self):
        collector = QCDataCollector(qc_dir=Path("/nonexistent/path"))
        assert collector.discover_qc_files() == []

    def test_load_all_concatenates(self, temp_qc_dir):
        collector = QCDataCollector(qc_dir=temp_qc_dir)
        df = collector.load_all()
        assert len(df) == 3
        assert set(df["patient_id"].unique()) == {"CON008", "CON009"}

    def test_load_all_empty_returns_empty_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = QCDataCollector(qc_dir=Path(tmpdir))
            df = collector.load_all()
            assert df.empty
            assert "patient_id" in df.columns

    def test_load_session_specific(self, temp_qc_dir):
        collector = QCDataCollector(qc_dir=temp_qc_dir)
        df = collector.load_session("CON008", "2025-08-14")
        assert len(df) == 2
        assert all(df["patient_id"] == "CON008")

    def test_load_session_missing(self, temp_qc_dir):
        collector = QCDataCollector(qc_dir=temp_qc_dir)
        df = collector.load_session("CON999", "2099-01-01")
        assert df.empty

    def test_get_available_sessions(self, temp_qc_dir):
        collector = QCDataCollector(qc_dir=temp_qc_dir)
        sessions = collector.get_available_sessions()
        assert len(sessions) == 2
        assert ("CON008", "2025-08-14") in sessions
        assert ("CON009", "2025-08-26") in sessions


# ── QCMetricsCalculator tests ────────────────────────────────────────────────


class TestQCMetricsCalculator:
    def test_compute_drop_rates(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_drop_rates()
        assert "drop_rate" in result.columns
        # CON008 language: 4/72
        lang_row = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")]
        assert abs(lang_row.iloc[0]["drop_rate"] - 4 / 72) < 1e-6
        # CON008 oddball: 0/4
        odd_row = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "oddball")]
        assert odd_row.iloc[0]["drop_rate"] == 0.0

    def test_compute_drop_rates_zero_epochs(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 0,
                    "n_epochs_dropped": 0,
                    "n_epochs_kept": 0,
                    "ica": "{}",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.compute_drop_rates()
        assert np.isnan(result.iloc[0]["drop_rate"])

    def test_compute_snr_estimates(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_snr_estimates()
        assert "snr_db" in result.columns
        # CON008 language: 20*log10(45/110) ≈ -7.76 dB
        lang_row = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")]
        expected_snr = 20 * np.log10(45.0 / 110.0)
        assert abs(lang_row.iloc[0]["snr_db"] - expected_snr) < 0.01

    def test_compute_snr_missing_ptp(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 10,
                    "n_epochs_dropped": 1,
                    "n_epochs_kept": 9,
                    "ica": "{}",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.compute_snr_estimates()
        assert "snr_db" in result.columns
        assert np.isnan(result.iloc[0]["snr_db"])

    def test_parse_ica_summaries(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.parse_ica_summaries()
        assert "ica_method" in result.columns
        assert "ica_classification_method" in result.columns
        assert "ica_n_components_excluded" in result.columns
        # CON008 language uses iclabel with 2 excluded
        row = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")].iloc[0]
        assert row["ica_method"] == "infomax"
        assert row["ica_classification_method"] == "iclabel"
        assert row["ica_n_components_excluded"] == 2
        assert row["ica_n_eog"] == 1
        assert row["ica_n_muscle"] == 1

    def test_parse_ica_null_json(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 10,
                    "n_epochs_dropped": 1,
                    "n_epochs_kept": 9,
                    "ica": None,
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.parse_ica_summaries()
        assert result.iloc[0]["ica_method"] is None
        assert result.iloc[0]["ica_n_components_excluded"] == 0

    def test_parse_ica_invalid_json(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 10,
                    "n_epochs_dropped": 1,
                    "n_epochs_kept": 9,
                    "ica": "not valid json {{{",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.parse_ica_summaries()
        assert result.iloc[0]["ica_method"] is None

    def test_compute_all_metrics(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert "drop_rate" in result.columns
        assert "snr_db" in result.columns
        assert "ica_method" in result.columns

    def test_summary_by_patient(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        summary = calc.summary_by_patient()
        assert len(summary) == 2
        assert set(summary["patient_id"]) == {"CON008", "CON009"}
        # CON008: total epochs = 72 + 4 = 76
        con008 = summary[summary["patient_id"] == "CON008"].iloc[0]
        assert con008["n_epochs_total"] == 76

    def test_summary_by_trial_type(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        summary = calc.summary_by_trial_type()
        assert "language" in summary["trial_type"].values
        assert "oddball" in summary["trial_type"].values
        # language: 72 + 72 = 144 total
        lang = summary[summary["trial_type"] == "language"].iloc[0]
        assert lang["n_epochs_total"] == 144

    def test_raw_df_immutability(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        raw = calc.raw_df
        raw["new_col"] = 1
        # Original should be unchanged
        assert "new_col" not in calc.raw_df.columns


# ── QCReportGenerator tests ──────────────────────────────────────────────────


class TestQCReportGenerator:
    def _enriched_df(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        return calc.compute_all_metrics()

    def test_generate_creates_html_file(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            assert path.exists()
            assert path.suffix == ".html"

    def test_html_contains_patient_ids(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "CON008" in html
            assert "CON009" in html

    def test_html_contains_metrics(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Drop Rate" in html
            assert "SNR" in html
            assert "Trial Type" in html

    def test_html_contains_css(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "<style>" in html
            assert "font-family" in html

    def test_html_contains_ica_summary(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Artifact Rejection Summary" in html
            assert "iclabel" in html

    def test_save_summary_csv(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            csv_path = gen.save_summary_csv()
            assert csv_path.exists()
            loaded = pd.read_csv(csv_path)
            assert len(loaded) == len(df)
            assert "drop_rate" in loaded.columns

    def test_generate_empty_df(self):
        df = pd.DataFrame(columns=["patient_id", "date", "trial_type"])
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            assert path.exists()
            html = path.read_text()
            assert "No QC data available" in html

    def test_custom_filename(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate(filename="custom_report.html")
            assert path.name == "custom_report.html"
            assert path.exists()


# ── Helper function tests ────────────────────────────────────────────────────


class TestHelpers:
    def test_empty_qc_dataframe_has_expected_columns(self):
        df = _empty_qc_dataframe()
        assert df.empty
        assert "patient_id" in df.columns
        assert "ica" in df.columns
        assert "ptp_uv_p50" in df.columns

    def test_parse_single_ica_json_valid(self, sample_ica_json):
        result = _parse_single_ica_json(sample_ica_json)
        assert result["ica_method"] == "infomax"
        assert result["ica_classification_method"] == "iclabel"
        assert result["ica_n_components_excluded"] == 2
        assert result["ica_n_eog"] == 1
        assert result["ica_n_muscle"] == 1

    def test_parse_single_ica_json_none(self):
        result = _parse_single_ica_json(None)
        assert result["ica_method"] is None
        assert result["ica_n_components_excluded"] == 0

    def test_parse_single_ica_json_nan(self):
        result = _parse_single_ica_json(float("nan"))
        assert result["ica_method"] is None

    def test_parse_single_ica_json_invalid_string(self):
        result = _parse_single_ica_json("not json")
        assert result["ica_method"] is None

    def test_parse_single_ica_json_dict_input(self):
        d = {"method": "infomax", "excluded": [0, 1, 2], "eog_components": [0]}
        result = _parse_single_ica_json(d)
        assert result["ica_method"] == "infomax"
        assert result["ica_n_components_excluded"] == 3
        assert result["ica_n_eog"] == 1

    def test_parse_iclabel_derives_counts_from_labels(self):
        """When ICLabel is used and per-type lists are empty, counts come from labels."""
        d = {
            "method": "infomax",
            "classification_method": "iclabel",
            "excluded": [0, 1, 2, 3, 7],
            "eog_components": [],
            "ecg_components": [],
            "muscle_components": [],
            "line_noise_components": [],
            "channel_noise_components": [],
            "iclabel_labels": [
                "line noise",
                "line noise",
                "eye blink",
                "eye blink",
                "brain",
                "brain",
                "brain",
                "muscle artifact",
            ],
        }
        result = _parse_single_ica_json(d)
        assert result["ica_n_components_excluded"] == 5
        assert result["ica_n_eog"] == 2  # indices 2, 3
        assert result["ica_n_muscle"] == 1  # index 7
        assert result["ica_n_line_noise"] == 2  # indices 0, 1
        assert result["ica_n_ecg"] == 0
        assert result["ica_n_channel_noise"] == 0

    def test_parse_iclabel_prefers_per_type_lists_when_populated(self):
        """When per-type lists are populated (correlation fallback), use those."""
        d = {
            "method": "infomax",
            "classification_method": "correlation",
            "excluded": [0, 2],
            "eog_components": [0],
            "ecg_components": [],
            "muscle_components": [2],
            "line_noise_components": [],
            "channel_noise_components": [],
            "iclabel_labels": None,
        }
        result = _parse_single_ica_json(d)
        assert result["ica_n_eog"] == 1
        assert result["ica_n_muscle"] == 1

    def test_count_iclabel_artifact_types_empty_labels(self):
        d = {"excluded": [0, 1], "iclabel_labels": None}
        counts = _count_iclabel_artifact_types(d, [0, 1])
        assert counts == {"eog": 0, "ecg": 0, "muscle": 0, "line_noise": 0, "channel_noise": 0}

    def test_count_iclabel_artifact_types_index_out_of_range(self):
        """Excluded index beyond labels list should be silently skipped."""
        d = {"iclabel_labels": ["eye blink", "brain"]}
        counts = _count_iclabel_artifact_types(d, [0, 99])
        assert counts["eog"] == 1  # index 0
        # index 99 is beyond labels, silently ignored

    def test_format_cell_drop_rate(self):
        assert _format_cell("drop_rate", 0.055) == "5.5%"
        assert _format_cell("mean_drop_rate", 0.1) == "10.0%"

    def test_format_cell_snr(self):
        assert _format_cell("snr_db", -7.76) == "-7.76"
        assert _format_cell("mean_snr_db", -5.0) == "-5.00"

    def test_format_cell_none(self):
        assert _format_cell("any_col", None) == "N/A"

    def test_format_cell_nan(self):
        assert _format_cell("any_col", float("nan")) == "N/A"

    def test_format_cell_ptp_threshold(self):
        assert _format_cell("reject_ptp_threshold_uv", 120.567) == "120.6"

    def test_safe_id(self):
        assert _safe_id("CON008") == "con008"
        assert _safe_id("Patient One") == "patient-one"

    def test_build_html_table(self):
        html = _build_html_table(["A", "B"], [["1", "2"], ["3", "4"]])
        assert "<table>" in html
        assert "<th>A</th>" in html
        assert "<td>1</td>" in html
        assert "</table>" in html

    def test_build_html_table_empty_rows(self):
        html = _build_html_table(["A"], [])
        assert "<table>" in html
        assert "<th>A</th>" in html

    def test_aggregate_summary_by_patient(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        enriched = calc.compute_all_metrics()
        summary = _aggregate_summary(enriched, group_cols=["patient_id"])
        assert len(summary) == 2
        assert "n_epochs_total" in summary.columns

    def test_aggregate_summary_empty(self):
        df = pd.DataFrame()
        result = _aggregate_summary(df, group_cols=["patient_id"])
        assert result.empty


# ── Integration test ─────────────────────────────────────────────────────────


class TestIntegration:
    def test_end_to_end_from_parquets_to_html(self, temp_qc_dir):
        """Full pipeline: parquet files -> collector -> calculator -> HTML report."""
        with tempfile.TemporaryDirectory() as output_dir:
            report_path = generate_qc_report(
                qc_dir=temp_qc_dir,
                output_dir=Path(output_dir),
            )
            assert report_path.exists()
            html = report_path.read_text()

            # Check structure
            assert "<!DOCTYPE html>" in html
            assert "EEG Quality Control Report" in html
            assert "CON008" in html
            assert "CON009" in html
            assert "Executive Summary" in html
            assert "Trial Type Overview" in html
            assert "Per-Patient Details" in html

            # Check CSV was also generated
            csv_path = Path(output_dir) / "qc_summary.csv"
            assert csv_path.exists()
            csv_df = pd.read_csv(csv_path)
            assert len(csv_df) == 3
            assert "drop_rate" in csv_df.columns
            assert "snr_db" in csv_df.columns

    def test_end_to_end_empty_dir(self):
        """Pipeline should gracefully handle an empty QC directory."""
        with tempfile.TemporaryDirectory() as qc_dir, tempfile.TemporaryDirectory() as output_dir:
            report_path = generate_qc_report(
                qc_dir=Path(qc_dir),
                output_dir=Path(output_dir),
            )
            assert report_path.exists()
            html = report_path.read_text()
            assert "No QC data available" in html


# ── Lazy import tests ────────────────────────────────────────────────────────


class TestLazyImports:
    def test_import_qc_report_generator(self):
        from src.data_processing import QCReportGenerator as QRG

        assert QRG is not None
        assert QRG.__name__ == "QCReportGenerator"

    def test_import_qc_data_collector(self):
        from src.data_processing import QCDataCollector as QDC

        assert QDC is not None
        assert QDC.__name__ == "QCDataCollector"

    def test_import_qc_metrics_calculator(self):
        from src.data_processing import QCMetricsCalculator as QMC

        assert QMC is not None
        assert QMC.__name__ == "QCMetricsCalculator"

    def test_import_generate_qc_report(self):
        from src.data_processing import generate_qc_report as gqr

        assert gqr is not None
        assert callable(gqr)

    def test_import_nonexistent_raises(self):
        with pytest.raises((AttributeError, ImportError)):
            from src.data_processing import NonExistentThing  # noqa: F401


# ── New metric tests ─────────────────────────────────────────────────────────


class TestComputeRetentionRate:
    def test_retention_rate_basic(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_retention_rate()
        assert "retention_rate" in result.columns
        # CON008 language: 68/72
        lang = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")].iloc[0]
        assert abs(lang["retention_rate"] - 68 / 72) < 1e-6

    def test_retention_rate_plus_drop_rate_equals_one(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        dr = calc.compute_drop_rates()
        calc2 = QCMetricsCalculator(dr)
        rr = calc2.compute_retention_rate()
        for _, row in rr.iterrows():
            if not np.isnan(row["drop_rate"]) and not np.isnan(row["retention_rate"]):
                assert abs(row["drop_rate"] + row["retention_rate"] - 1.0) < 1e-6

    def test_retention_rate_zero_epochs(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 0,
                    "n_epochs_dropped": 0,
                    "n_epochs_kept": 0,
                    "ica": "{}",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.compute_retention_rate()
        assert np.isnan(result.iloc[0]["retention_rate"])

    def test_retention_rate_in_compute_all_metrics(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert "retention_rate" in result.columns
        assert result["retention_rate"].notna().any()


class TestComputeDataCoverage:
    def test_data_coverage_basic(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_data_coverage()
        assert "estimated_recording_min" in result.columns
        # CON008 language: window_sec=16, n_epochs=72 → 16*72/60 = 19.2 min
        lang = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")].iloc[0]
        assert abs(lang["estimated_recording_min"] - 16.0 * 72 / 60) < 0.01

    def test_data_coverage_missing_window_sec(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 10,
                    "n_epochs_dropped": 1,
                    "n_epochs_kept": 9,
                    "ica": "{}",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.compute_data_coverage()
        # No window_sec → should still add the column (filled with NaN)
        assert "estimated_recording_min" in result.columns

    def test_data_coverage_in_compute_all_metrics(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert "estimated_recording_min" in result.columns
        assert result["estimated_recording_min"].notna().any()


class TestFlagUsableSessions:
    def test_usable_flag_basic(self, sample_qc_df):
        """flag_usable_sessions requires retention_rate to already be computed."""
        calc = QCMetricsCalculator(sample_qc_df)
        # Without retention_rate → must raise
        with pytest.raises(ValueError, match="retention_rate"):
            calc.flag_usable_sessions()
        # With retention_rate present → succeeds
        df_with_retention = calc.compute_retention_rate()
        calc2 = QCMetricsCalculator(df_with_retention)
        result = calc2.flag_usable_sessions(min_retention=0.5, min_epochs_kept=1)
        assert "is_usable" in result.columns

    def test_usable_flag_all_usable_with_high_retention(self, sample_qc_df):
        """All rows in sample data have ≥94% retention → all usable."""
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert result["is_usable"].all()

    def test_usable_flag_unusable_below_threshold(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 100,
                    "n_epochs_dropped": 80,
                    "n_epochs_kept": 20,
                    "window_sec": 16.0,
                    "ica": "{}",
                    "ptp_uv_p50": 20.0,
                    "ptp_uv_p95": 90.0,
                    "ptp_uv_mean": 30.0,
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.compute_all_metrics()
        # 20% retention < 50% threshold → should be False
        assert not result.iloc[0]["is_usable"]

    def test_usable_flag_custom_threshold(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        calc2 = QCMetricsCalculator(result)
        # Require 100% retention → none should pass
        result2 = calc2.flag_usable_sessions(min_retention=1.0, min_epochs_kept=1)
        assert not result2["is_usable"].all()

    def test_usable_flag_in_compute_all_metrics(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert "is_usable" in result.columns
        assert result["is_usable"].dtype == bool


class TestParseDropReasons:
    def test_parse_drop_reasons_normalizes(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.parse_drop_reasons()
        assert "primary_drop_reason" in result.columns
        # Known value: "ENG03_PTP_GT_P95" → "eng03_ptp_gt_p95"
        lang = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "language")].iloc[0]
        assert lang["primary_drop_reason"] == "eng03_ptp_gt_p95"

    def test_parse_drop_reasons_null_becomes_none(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.parse_drop_reasons()
        # CON008 oddball has null drop_reason
        odd = result[(result["patient_id"] == "CON008") & (result["trial_type"] == "oddball")].iloc[0]
        assert odd["primary_drop_reason"] == "none"

    def test_parse_drop_reasons_missing_column(self):
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "date": "2025-01-01",
                    "trial_type": "language",
                    "n_epochs_total": 10,
                    "n_epochs_dropped": 1,
                    "n_epochs_kept": 9,
                    "ica": "{}",
                }
            ]
        )
        calc = QCMetricsCalculator(df)
        result = calc.parse_drop_reasons()
        assert "primary_drop_reason" in result.columns
        assert result.iloc[0]["primary_drop_reason"] == "none"

    def test_parse_drop_reasons_in_compute_all_metrics(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        result = calc.compute_all_metrics()
        assert "primary_drop_reason" in result.columns


class TestNewMetricsInHTML:
    def _enriched_df(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        return calc.compute_all_metrics()

    def test_html_contains_retention_rate(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Retention Rate" in html

    def test_html_contains_ptp_columns(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Median PTP" in html
            assert "P95 PTP" in html

    def test_html_contains_session_notes(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Session Notes" in html
            assert "noisy session" in html

    def test_html_contains_usable_flag(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Usable Sessions" in html

    def test_html_contains_recording_time(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            path = gen.generate()
            html = path.read_text()
            assert "Est. Total Recording Time" in html

    def test_csv_contains_new_columns(self, sample_qc_df):
        df = self._enriched_df(sample_qc_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            csv_path = gen.save_summary_csv()
            loaded = pd.read_csv(csv_path)
            assert "retention_rate" in loaded.columns
            assert "estimated_recording_min" in loaded.columns
            assert "is_usable" in loaded.columns
            assert "primary_drop_reason" in loaded.columns

    def test_aggregate_summary_includes_new_cols(self, sample_qc_df):
        from src.data_processing.qc_report import _aggregate_summary

        calc = QCMetricsCalculator(sample_qc_df)
        enriched = calc.compute_all_metrics()
        summary = _aggregate_summary(enriched, group_cols=["patient_id"])
        assert "mean_retention_rate" in summary.columns
        assert "total_recording_min" in summary.columns
        assert "n_usable" in summary.columns


# ── Filter tests ──────────────────────────────────────────────────────────────


class TestFilterBanner:
    """Unit tests for QCReportGenerator._render_filter_banner."""

    def _make_gen(self, filters):
        df = pd.DataFrame([{"patient_id": "P1", "date": "2025-01-01", "trial_type": "lang"}])
        return QCReportGenerator(df, active_filters=filters)

    def test_no_filters_returns_empty(self):
        assert self._make_gen({})._render_filter_banner() == ""

    def test_none_active_filters_returns_empty(self):
        assert self._make_gen(None)._render_filter_banner() == ""

    def test_patient_id_filter_shown(self):
        html = self._make_gen({"patient_id": ["CON008"]})._render_filter_banner()
        assert "CON008" in html and "filter-banner" in html

    def test_date_filter_shown(self):
        html = self._make_gen({"date": ["2025-08-14"]})._render_filter_banner()
        assert "2025-08-14" in html

    def test_combined_filters_shown(self):
        html = self._make_gen({"patient_id": ["CON008"], "date": ["2025-08-14"]})._render_filter_banner()
        assert "CON008" in html and "2025-08-14" in html

    def test_banner_absent_in_unfiltered_html(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        df = calc.compute_all_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir))
            html = gen.generate().read_text()
        assert "Filtered Report" not in html

    def test_banner_present_in_filtered_html(self, sample_qc_df):
        calc = QCMetricsCalculator(sample_qc_df)
        df = calc.compute_all_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = QCReportGenerator(df, output_dir=Path(tmpdir), active_filters={"patient_id": ["CON008"]})
            html = gen.generate().read_text()
        assert "Filtered Report" in html and "CON008" in html


class TestGenerateQCReportFilters:
    """Integration tests for patient_id and date filters in generate_qc_report."""

    def test_patient_id_filter_keeps_only_matching(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            html = generate_qc_report(
                qc_dir=temp_qc_dir,
                output_dir=Path(outdir),
                patient_ids=["CON008"],
            ).read_text()
        assert "CON008" in html and "CON009" not in html

    def test_date_filter_keeps_only_matching(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            html = generate_qc_report(
                qc_dir=temp_qc_dir,
                output_dir=Path(outdir),
                dates=["2025-08-14"],
            ).read_text()
        assert "2025-08-14" in html and "CON009" not in html

    def test_no_filters_includes_all(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            html = generate_qc_report(qc_dir=temp_qc_dir, output_dir=Path(outdir)).read_text()
        assert "CON008" in html and "CON009" in html

    def test_filter_banner_appears_in_filtered_report(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            html = generate_qc_report(
                qc_dir=temp_qc_dir,
                output_dir=Path(outdir),
                patient_ids=["CON008"],
            ).read_text()
        assert "Filtered Report" in html

    def test_no_filter_banner_in_unfiltered_report(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            html = generate_qc_report(qc_dir=temp_qc_dir, output_dir=Path(outdir)).read_text()
        assert "Filtered Report" not in html

    def test_csv_reflects_patient_filter(self, temp_qc_dir):
        with tempfile.TemporaryDirectory() as outdir:
            generate_qc_report(
                qc_dir=temp_qc_dir,
                output_dir=Path(outdir),
                patient_ids=["CON008"],
            )
            df = pd.read_csv(Path(outdir) / "qc_summary.csv")
        assert set(df["patient_id"].unique()) == {"CON008"}
