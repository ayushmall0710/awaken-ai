"""Claassen SVM Command Following pipeline runner for the awakenai CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.cli.cli_utils import generate_pdf_from_html, print_report_paths, print_table
from src.data_loading import UnifiedDataLoader
from src.pipelines.command_following_claassen import CommandFollowingClaassen
from src.reports import style_utils
from src.reports.command_following_claassen_report import CommandFollowingClaassenReport


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    alpha: float,
    report: bool = False,
    n_perms: int = 1000,
) -> None:
    """Run CommandFollowingClaassen (SVM) for the given patients/sessions."""
    pipeline = CommandFollowingClaassen(loader=loader, n_permutations=n_perms)
    generated_reports: list[tuple[str, str, Path, Optional[Path]]] = []

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        if report:
            patient_panel = style_utils.build_patient_panel(pid)

        for sess in sessions:
            typer.echo(f"[command-following-svm] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, session_id=sess, alpha=alpha)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — SVM Results")

                summary = pipeline.generate_summary()
                typer.echo(f"  Classification: {summary['cmd_status']}")

                for sr in summary.get("side_results", []):
                    typer.echo(
                        f"    {sr['side']}: AUC={sr['auc']:.3f}  "
                        f"Acc={sr['accuracy']:.1%}  "
                        f"p={sr['p_value_perm']:.4f}  "
                        f"{'✓ Significant' if sr['significant'] else '✗ Not significant'}"
                    )

                if report:
                    svm_report = CommandFollowingClaassenReport(pipeline, session_id=sess)
                    pdf_out = svm_report.report_file.with_suffix(".pdf")
                    style_utils.stitch_and_save(
                        [patient_panel, svm_report.build_session_html()],
                        output_path=svm_report.report_file,
                        title="Command Following SVM",
                        generator_name="AwakenAI Capstone",
                        extra_css=svm_report._build_css_extensions(),
                        pdf_path=pdf_out,
                    )
                    status = generate_pdf_from_html(svm_report.report_file, pdf_out)
                    generated_reports.append((pid, sess, svm_report.report_file, pdf_out if status else None))
            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)

    if report and generated_reports:
        typer.echo("\nGenerated Reports:")
        for pid, sess, html_path, pdf_path in generated_reports:
            print_report_paths(pid, sess, html_path=html_path, pdf_path=pdf_path)
