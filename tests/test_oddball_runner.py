"""Tests for the oddball CLI runner report output path."""

from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.cli.runners import oddball as oddball_runner
from src.data_loading import config


def test_report_output_path_uses_timestamped_session_folder(tmp_path, monkeypatch):
    class FixedDateTime:
        @staticmethod
        def now():
            return real_datetime(2026, 3, 13, 18, 3, 0)

    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "p300_oddball_clinical.parquet",
        "p300_oddball_electrode_detail.parquet",
        "p300_oddball_mapping_qc.parquet",
    ):
        (features_dir / name).write_text("placeholder", encoding="utf-8")

    patient_id = "CON008"
    session_id = "s_CON008_202508140000"

    clinical_df = pd.DataFrame(
        [
            {
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
        ]
    )
    detail_df = pd.DataFrame(
        [
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
            }
        ]
    )
    mapping_df = pd.DataFrame(
        [
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
        ]
    )

    def fake_read_parquet(path: Path):
        path = Path(path)
        if path.name == "p300_oddball_clinical.parquet":
            return clinical_df
        if path.name == "p300_oddball_electrode_detail.parquet":
            return detail_df
        if path.name == "p300_oddball_mapping_qc.parquet":
            return mapping_df
        raise AssertionError(f"Unexpected parquet path: {path}")

    mock_loader = MagicMock()
    mock_patient = MagicMock()
    mock_patient.list_session_ids.return_value = [session_id]
    mock_loader.get_patient.return_value = mock_patient

    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = clinical_df

    captured = {}

    def fake_stitch_and_save(fragments, output_path, title, generator_name, extra_css):
        captured["output_path"] = Path(output_path)
        captured["title"] = title
        return Path(output_path)

    monkeypatch.setattr(config, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(
        config,
        "REPORT_DIR_TEMPLATE",
        str(tmp_path / "{patient_id}" / "{session_id}" / "{pipeline_name}"),
    )
    monkeypatch.setattr(oddball_runner, "datetime", FixedDateTime)
    monkeypatch.setattr(oddball_runner.pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(oddball_runner, "P300OddballPipeline", lambda loader: mock_pipeline)
    monkeypatch.setattr(oddball_runner, "print_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(oddball_runner.style_utils, "stitch_and_save", fake_stitch_and_save)
    monkeypatch.setattr(oddball_runner.typer, "echo", lambda *args, **kwargs: None)

    oddball_runner.run(
        loader=mock_loader,
        patient_ids=[patient_id],
        session=session_id,
        electrodes=None,
        report=True,
    )

    assert captured["title"] == "P300 Oddball Summary — Combined Report"
    assert captured["output_path"] == (
        tmp_path / patient_id / session_id / "oddball" / "20260313_180300" / "oddball_qc.html"
    )
