"""Oddball pipeline runner for the awakenai CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
from src.pipelines.p300_oddball import P300OddballPipeline
from src.reports import style_utils
from src.reports.oddball_report import OddballQCReport


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    electrodes: Optional[list[str]],
    report: bool = False,
) -> None:
    """Run P300OddballPipeline for the given patients/sessions.

    When report=True, one HTML report is written per session to the path
        defined by config.REPORT_DIR_TEMPLATE (patient_id/session_id/oddball/timestamp).
    """
    pipeline = P300OddballPipeline(loader=loader)

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        for sess in sessions:
            typer.echo(f"[oddball] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, session=sess, custom_electrodes=electrodes)
                if df is not None and not df.empty:
                    print_table(df, title=f"{pid} / {sess} — P300 Features")
                else:
                    typer.echo("  No oddball data or features.")
            except Exception as e:  # pragma: no cover - defensive
                typer.echo(f"  ✗ Failed: {e}", err=True)

    if report:
        # Parquet-based stitcher — one report per session with timestamped directory.
        clinical_path = config.FEATURES_DIR / "p300_oddball_clinical.parquet"
        detail_path = config.FEATURES_DIR / "p300_oddball_electrode_detail.parquet"
        mapping_path = config.FEATURES_DIR / "p300_oddball_mapping_qc.parquet"

        missing = [p for p in (clinical_path, detail_path, mapping_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Oddball feature parquets not found: {[str(p) for p in missing]}. "
                "Run the oddball pipeline first to generate them."
            )

        clinical_df = pd.read_parquet(clinical_path)
        detail_df = pd.read_parquet(detail_path)
        mapping_df = pd.read_parquet(mapping_path)

        # Apply filters: patient_ids always provided; session may be None.
        clinical_df = clinical_df[clinical_df["patient_id"].isin(patient_ids)]
        detail_df = detail_df[detail_df["patient_id"].isin(patient_ids)]
        mapping_df = mapping_df[mapping_df["patient_id"].isin(patient_ids)]
        if session:
            clinical_df = clinical_df[clinical_df["session_id"] == session]
            detail_df = detail_df[detail_df["session_id"] == session]
            mapping_df = mapping_df[mapping_df["session_id"] == session]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for pid in sorted(clinical_df["patient_id"].unique()):
            for sess in sorted(clinical_df[clinical_df["patient_id"] == pid]["session_id"].unique()):
                clinical_row = clinical_df[
                    (clinical_df["patient_id"] == pid) & (clinical_df["session_id"] == sess)
                ].iloc[0]
                sess_detail = detail_df[(detail_df["patient_id"] == pid) & (detail_df["session_id"] == sess)]
                mapping_row = mapping_df[(mapping_df["patient_id"] == pid) & (mapping_df["session_id"] == sess)].iloc[0]

                out_dir = Path(
                    config.REPORT_DIR_TEMPLATE.format(
                        patient_id=pid,
                        session_id=sess,
                        pipeline_name="oddball",
                        timestamp=timestamp,
                    )
                )
                if "{timestamp}" not in config.REPORT_DIR_TEMPLATE:
                    out_dir = out_dir / timestamp
                out_path = out_dir / "oddball_qc.html"

                html_fragments: list[str] = []
                report_obj = OddballQCReport(pid, sess, clinical_row, sess_detail, mapping_row)
                extra_css = report_obj._build_css_extensions()
                html_fragments.append(style_utils.build_patient_panel(pid))
                html_fragments.append(report_obj.build_session_html())

                out = style_utils.stitch_and_save(
                    html_fragments,
                    output_path=out_path,
                    title="P300 Oddball Summary — Combined Report",
                    generator_name="P300 Oddball Pipeline",
                    extra_css=extra_css,
                )
                typer.echo(f"Report: {out}")
