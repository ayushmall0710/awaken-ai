"""Command Following pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
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
    html_fragments: list[str] = []
    extra_css: str = ""

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        if report:
            html_fragments.append(style_utils.build_patient_panel(pid))

        for sess in sessions:
            typer.echo(f"[command-following] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, alpha=alpha)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — ERD Results")

                    if report:
                        cf_report = CommandFollowingReport(pipeline, session_id=sess)
                        extra_css = cf_report._build_css_extensions()
                        html_fragments.append(cf_report.build_session_html())
            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)

    if report and html_fragments:
        out = style_utils.stitch_and_save(
            html_fragments,
            output_path=config.REPORTS_DIR / "combined_cf_report.html",
            title="Command Following — Combined Report",
            generator_name="AwakenAI Capstone",
            extra_css=extra_css,
        )
        typer.echo(f"Combined report: {out}")
