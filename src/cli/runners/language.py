"""Language Tracking pipeline runner for the awakenai CLI."""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
from src.pipelines.language_tracking import LanguageTrackingAnalysis


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    report: bool = False,
) -> None:
    """Run LanguageTrackingAnalysis for the given patients/sessions."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()
        report_fragments: list[str] = []
        extra_css: str = ""

        for sess in sessions:
            typer.echo(f"[language] {pid} / {sess} ...")
            pipeline = LanguageTrackingAnalysis(loader=loader, session_id=sess)
            try:
                df = pipeline.run(pid, session_id=sess)
                if df.empty:
                    continue

                # CLI: Show a subset of columns for better visibility
                cli_cols = [
                    "focus",
                    "itpc_comprehension",
                    "dft_p_comprehension",
                    "morlet_itpc_comprehension",
                    "morlet_p_comprehension",
                ]
                # Filter to available columns (protect against schema changes)
                available_cols = [c for c in cli_cols if c in df.columns]
                print_table(df[available_cols], title=f"{pid} / {sess} — Language Tracking Results")

                out_dir = config.LOCAL_DATA_ROOT / "outputs" / "language" / pid / sess / timestamp
                out_dir.mkdir(parents=True, exist_ok=True)

                # Save CSV
                out_file = out_dir / "features.csv"
                write_header = not out_file.exists()
                df.to_csv(out_file, mode="a", index=False, header=write_header)
                typer.echo(f"  Saved features to: {out_file}")

                # Save intermediate arrays for report/re-analysis
                if pipeline._dft_spectrum_full is not None:
                    npz_file = out_dir / "features.npz"
                    np.savez(
                        npz_file,
                        dft_spectrum_full=pipeline._dft_spectrum_full,
                        dft_freqs=pipeline._dft_freqs,
                        ch_names=np.array(pipeline._dft_ch_names),
                    )
                    typer.echo(f"  Saved arrays to: {npz_file}")

                if report:
                    from src.reports import style_utils
                    from src.reports.language_tracking_report import LanguageTrackingReport

                    rpt = LanguageTrackingReport(pipeline, session_id=sess, output_dir=out_dir)
                    if len(sessions) == 1:
                        path = rpt.generate()
                        typer.echo(f"  Report: {path}")
                    else:
                        if not extra_css:
                            extra_css = ""
                        report_fragments.append(rpt.build_session_html())

            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)

        # Multi-session combined report
        if report and len(report_fragments) >= 1:
            from src.reports import style_utils

            combined_dir = config.LOCAL_DATA_ROOT / "outputs" / "language" / timestamp
            combined_dir.mkdir(parents=True, exist_ok=True)
            out_path = style_utils.stitch_and_save(
                report_fragments,
                output_path=combined_dir / "language_combined_report.html",
                title=f"{pid} — Language Tracking Combined Report",
                generator_name="Language Tracking Pipeline",
                extra_css=extra_css,
            )
            typer.echo(f"  Combined report: {out_path}")
