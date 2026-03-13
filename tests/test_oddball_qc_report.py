"""Tests for OddballQCReport (parquet-based oddball QC HTML)."""

import math

import pandas as pd

from src.reports.oddball_qc_report import OddballQCReport


def _make_clinical_row(patient_id: str, session_id: str, **overrides) -> pd.Series:
    row = {
        "patient_id": patient_id,
        "session_id": session_id,
        "session_date": "2025-08-14",
        "n_rare_epochs": 10,
        "n_standard_epochs": 65,
        "baseline_std_uV": 1.89,
        "p300_rare_amplitude_Pz_uV": 3.08,
        "p300_rare_latency_Pz_ms": 476.56,
        "p300_diff_amplitude_Pz_uV": 5.40,
        "p300_diff_latency_Pz_ms": 467.0,
        "diff_mmn_amplitude_Fz_uV": -4.36,
        "diff_mmn_latency_Fz_ms": 152.34,
        "p300_best_electrode": "Pz",
        "p300_subtype": "P3b",
        "p300_amplitude_uV": 3.90,
        "p300_latency_ms": 430.0,
        "p300_n_valid_electrodes": 3,
        "qc_notes": "",
        "qc_pass": True,
        "p300_p_value": 0.775,
        "p300_t_stat": 0.29,
        "p300_n_rare": 10,
        "p300_n_standard": 65,
    }
    row.update(overrides)
    return pd.Series(row)


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
                "diff_mmn_amplitude_uV": -4.36,
                "diff_mmn_latency_ms": 152.34,
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
                "diff_mmn_amplitude_uV": -2.4,
                "diff_mmn_latency_ms": 165.0,
            },
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "electrode": "Pz",
                "p300_amplitude_uV": 4.2,
                "p300_latency_ms": 385,
                "is_valid": True,
                "flagged_reason": None,
                "diff_amplitude_uV": 5.4,
                "diff_latency_ms": 467,
                "diff_mmn_amplitude_uV": -1.1,
                "diff_mmn_latency_ms": 180.0,
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


def _make_report(**clinical_overrides) -> OddballQCReport:
    patient_id = "CON008"
    session_id = "s_CON008_202508140000"
    clinical_row = _make_clinical_row(patient_id, session_id, **clinical_overrides)
    detail_df = _make_detail_df(patient_id, session_id)
    mapping_row = _make_mapping_row(patient_id, session_id)
    return OddballQCReport(patient_id, session_id, clinical_row, detail_df, mapping_row)


def test_interpret_low_confidence_detected():
    report = _make_report()

    summary = report._interpret_p300_summary()

    assert summary["candidate_present"] is True
    assert summary["confidence_label"] == "Low-confidence detected"
    assert summary["stats_support"] == "not_supported"
    assert summary["trial_count_tier"] == "borderline"
    assert summary["snr_tier"] == "borderline"
    assert summary["difference_support"] == "supportive"
    assert "borderline rare-trial count" in summary["limiter_phrases"]
    assert "borderline signal-to-noise" in summary["limiter_phrases"]
    assert "non-significant rare-standard contrast" in summary["limiter_phrases"]


def test_interpret_detected():
    report = _make_report(
        n_rare_epochs=24,
        baseline_std_uV=1.4,
        p300_p_value=0.04,
        p300_n_rare=24,
        p300_n_standard=80,
    )

    summary = report._interpret_p300_summary()

    assert summary["candidate_present"] is True
    assert summary["confidence_label"] == "Detected"
    assert summary["stats_support"] == "supportive"
    assert summary["trial_count_tier"] == "good"
    assert summary["data_quality_tier"] in {"good", "borderline"}


def test_interpret_no_reliable_missing_pz():
    report = _make_report(
        p300_rare_amplitude_Pz_uV=float("nan"),
        p300_rare_latency_Pz_ms=float("nan"),
    )

    summary = report._interpret_p300_summary()

    assert summary["pz_available"] is False
    assert summary["candidate_reason"] == "pz_unavailable"
    assert summary["confidence_label"] == "No reliable P300 detected"


def test_interpret_no_reliable_due_to_poor_quality():
    report = _make_report(
        n_rare_epochs=8,
        baseline_std_uV=5.0,
        p300_p_value=0.08,
        p300_n_rare=8,
        p300_n_standard=65,
    )

    summary = report._interpret_p300_summary()

    assert summary["candidate_present"] is True
    assert summary["trial_count_tier"] == "poor"
    assert summary["snr_tier"] == "poor"
    assert summary["data_quality_tier"] == "poor"
    assert summary["confidence_label"] == "No reliable P300 detected"


def test_interpret_stats_unavailable():
    report = _make_report(
        p300_p_value=float("nan"),
        p300_n_rare=1,
        p300_n_standard=65,
    )

    summary = report._interpret_p300_summary()

    assert summary["stats_support"] == "unavailable"


def test_interpret_mmn_invalid_when_positive():
    report = _make_report(diff_mmn_amplitude_Fz_uV=1.5, diff_mmn_latency_Fz_ms=150.0)

    summary = report._interpret_p300_summary()

    assert summary["mmn_valid"] is False


def test_build_session_html_contains_new_card_titles_and_labels():
    report = _make_report()

    html = report.build_session_html()

    assert "<details class='session-wrapper'" in html
    assert "P300 Candidate at Pz" in html
    assert "Confidence" in html
    assert "Rare vs Standard Support" in html
    assert "Data Quality" in html
    assert "Low-confidence detected" in html
    assert "MMN at Fz" in html
    assert "Topography" in html


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
    assert "P300 Oddball Summary Report" in content
    assert "P300 Candidate at Pz" in content
    assert "Low-confidence detected" in content
    assert patient_id in content
    assert session_id in content
    assert "</html>" in content


def test_format_cell_is_valid_icons():
    from src.reports.oddball_qc_report import ICON_FALSE, ICON_TRUE

    report = _make_report()

    assert ICON_TRUE in report._format_cell("is_valid", True)
    assert ICON_FALSE in report._format_cell("is_valid", False)
    assert report._format_cell("is_valid", None) == "N/A"
    assert report._format_cell("other", 3.14) == "3.14"
    assert report._format_cell("other", None) == "N/A"
    assert report._format_cell("other", math.nan) == "N/A"
