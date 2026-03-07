"""Oddball pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
from src.pipelines.p300_oddball import P300OddballPipeline
from src.reports import style_utils


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    electrodes: Optional[list[str]],
    report: bool = False,
) -> None:
    """Run P300OddballPipeline for the given patients/sessions.

    When report=True, a combined HTML report is written to
    REPORTS_DIR/combined_oddball_qc.html that contains every patient and
    session that ran successfully (mirrors the Command Following runner pattern).
    """
    pipeline = P300OddballPipeline(loader=loader)
    html_fragments: list[str] = []
    extra_css: str = ""

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        if report:
            html_fragments.append(style_utils.build_patient_panel(pid))

        for sess in sessions:
            typer.echo(f"[oddball] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, session=sess, custom_electrodes=electrodes)
                if df is not None and not df.empty:
                    print_table(df, title=f"{pid} / {sess} — P300 Features")

                    if report:
                        from src.reports.oddball_qc_report import OddballQCReport

                        qc_report = OddballQCReport(pipeline, session_id=sess)
                        extra_css = qc_report._build_css_extensions()
                        html_fragments.append(qc_report.build_session_html())
                else:
                    typer.echo("  No oddball data or features.")
            except Exception as e:  # pragma: no cover - defensive
                typer.echo(f"  ✗ Failed: {e}", err=True)

    if report and html_fragments:
        out = style_utils.stitch_and_save(
            html_fragments,
            output_path=config.REPORTS_DIR / "combined_oddball_qc.html",
            title="P300 Oddball QC — Combined Report",
            generator_name="P300 Oddball Pipeline",
            extra_css=extra_css,
        )
        typer.echo(f"Combined report: {out}")
