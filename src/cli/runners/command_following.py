"""Command Following pipeline runner for the awakenai CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.cli.cli_utils import generate_pdf_from_html, print_report_paths, print_table
from src.data_loading import UnifiedDataLoader
from src.pipelines.command_following import CommandFollowingAnalysis
from src.reports import style_utils
from src.reports.command_following_report import CommandFollowingReport


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    alpha: float,
    report: bool = False,
) -> None:
    """Run CommandFollowingAnalysis for the given patients/sessions.

    When report=True a combined HTML report is written to
    REPORTS_DIR/combined_cf_report.html that contains every patient and
    session that ran successfully.
    """
    pipeline = CommandFollowingAnalysis()
    generated_reports: list[tuple[str, str, Path, Optional[Path]]] = []

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        if report:
            patient_panel = style_utils.build_patient_panel(pid)

        for sess in sessions:
            typer.echo(f"[command-following] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, alpha=alpha)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — ERD Results")

                    if report:
                        cf_report = CommandFollowingReport(pipeline, session_id=sess)
                        pdf_out = cf_report.report_file.with_suffix(".pdf")
                        style_utils.stitch_and_save(
                            [patient_panel, cf_report.build_session_html()],
                            output_path=cf_report.report_file,
                            title="Command Following",
                            generator_name="AwakenAI Capstone",
                            extra_css=cf_report._build_css_extensions(),
                            pdf_path=pdf_out,
                        )
                        status = generate_pdf_from_html(cf_report.report_file, pdf_out)
                        generated_reports.append((pid, sess, cf_report.report_file, pdf_out if status else None))
            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)

    if report and generated_reports:
        typer.echo("\nGenerated Reports:")
        for pid, sess, html_path, pdf_path in generated_reports:
            print_report_paths(pid, sess, html_path=html_path, pdf_path=pdf_path)
