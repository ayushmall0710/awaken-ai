"""Tests for OddballQCReport (parquet-based oddball QC HTML)."""

import pandas as pd

from src.reports.oddball_qc_report import OddballQCReport


def _make_clinical_row(patient_id: str, session_id: str) -> pd.Series:
    return pd.Series(
        {
            "patient_id": patient_id,
            "session_id": session_id,
            "session_date": "2025-08-14",
            "n_rare_epochs": 12,
            "n_standard_epochs": 24,
            "baseline_std_uV": 2.1,
            "p300_diff_amplitude_Pz_uV": 4.5,
            "p300_diff_latency_Pz_ms": 380,
            "p300_best_electrode": "Pz",
            "p300_subtype": "P3b",
            "p300_amplitude_uV": 4.2,
            "p300_latency_ms": 385,
            "p300_n_valid_electrodes": 3,
            "qc_notes": "",
            "qc_pass": True,
        }
    )


def _make_detail_df(patient_id: str, session_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "electrode": "Fz",
                "p300_amplitude_uV": 2.1,
                "p300_latency_ms": 350,
                "is_valid": True,
                "flagged_reason": None,
                "diff_amplitude_uV": 2.0,
                "diff_latency_ms": 355,
            },
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "electrode": "Cz",
                "p300_amplitude_uV": 3.5,
                "p300_latency_ms": 370,
                "is_valid": True,
                "flagged_reason": None,
                "diff_amplitude_uV": 3.2,
                "diff_latency_ms": 372,
            },
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "electrode": "Pz",
                "p300_amplitude_uV": 4.2,
                "p300_latency_ms": 385,
                "is_valid": True,
                "flagged_reason": None,
                "diff_amplitude_uV": 4.5,
                "diff_latency_ms": 380,
            },
        ]
    )


def _make_mapping_row(patient_id: str, session_id: str) -> pd.Series:
    return pd.Series(
        {
            "patient_id": patient_id,
            "session_id": session_id,
            "n_rare_events_candidate": 15,
            "n_rare_mapped": 12,
            "n_rare_unmapped": 2,
            "n_rare_boundary_clipped": 1,
            "rare_mapping_rate": 0.8,
            "n_standard_events_candidate": 30,
            "n_standard_mapped": 24,
        }
    )


def test_build_session_html_contains_details_and_session_id():
    patient_id = "CON008"
    session_id = "s_CON008_202508140000"
    clinical_row = _make_clinical_row(patient_id, session_id)
    detail_df = _make_detail_df(patient_id, session_id)
    mapping_row = _make_mapping_row(patient_id, session_id)

    report = OddballQCReport(patient_id, session_id, clinical_row, detail_df, mapping_row)
    html = report.build_session_html()

    assert "<details class='session-wrapper'" in html
    assert session_id in html
    assert "session-content" in html
    assert "P300 QC Pass" in html or "qc_pass" in html.lower() or "P3b" in html


def test_generate_writes_html_file(tmp_path):
    patient_id = "CON008"
    session_id = "s_CON008_202508140000"
    clinical_row = _make_clinical_row(patient_id, session_id)
    detail_df = _make_detail_df(patient_id, session_id)
    mapping_row = _make_mapping_row(patient_id, session_id)

    report = OddballQCReport(
        patient_id,
        session_id,
        clinical_row,
        detail_df,
        mapping_row,
        output_dir=tmp_path,
    )
    out = report.generate()

    assert out.exists()
    assert out == tmp_path / f"{session_id}_oddball_qc.html"
    content = out.read_text(encoding="utf-8")
    assert "P300 Oddball QC Report" in content
    assert session_id in content
    assert patient_id in content
    assert "Clinical summary" in content or "metric" in content.lower()
    assert "</html>" in content


def test_format_cell_is_valid_icons():
    from src.reports.oddball_qc_report import ICON_FALSE, ICON_TRUE, OddballQCReport

    patient_id = "CON008"
    session_id = "s_CON008_202508140000"
    clinical_row = _make_clinical_row(patient_id, session_id)
    detail_df = _make_detail_df(patient_id, session_id)
    mapping_row = _make_mapping_row(patient_id, session_id)
    report = OddballQCReport(patient_id, session_id, clinical_row, detail_df, mapping_row)

    assert ICON_TRUE in report._format_cell("is_valid", True)
    assert ICON_FALSE in report._format_cell("is_valid", False)
    assert report._format_cell("is_valid", None) == "N/A"
    assert report._format_cell("other", 3.14) == "3.14"
    assert report._format_cell("other", None) == "N/A"
