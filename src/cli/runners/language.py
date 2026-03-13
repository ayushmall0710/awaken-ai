"""Language Tracking pipeline runner for the awakenai CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
from src.pipelines.language_tracking import LanguageTrackingAnalysis
from src.reports.language_tracking_report import LanguageTrackingReport


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    report: bool = False,
) -> None:
    """Run LanguageTrackingAnalysis for the given patients/sessions."""

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
                print_table(
                    df[available_cols],
                    title=f"{pid} / {sess} — Language Tracking Results",
                )

                # Construct standardized output directory for artifacts (CSV, NPZ)
                out_dir = Path(
                    config.REPORT_DIR_TEMPLATE.format(
                        patient_id=pid,
                        session_id=sess,
                        pipeline_name="language_tracking",
                    )
                )
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
                    rpt = LanguageTrackingReport(pipeline, session_id=sess, output_dir=out_dir)
                    fragment = rpt.build_session_html()
                    report_fragments.append(fragment)

                    # Always provide the standalone report in the session dir
                    path = rpt.generate()
                    typer.echo(f"  Report: {path}")

            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)

        # Per-patient combined report (shows all successful sessions for this patient)
        if report and len(report_fragments) > 1:
            from src.reports import style_utils

            # Save in patient's report using the combined template
            patient_report_dir = Path(
                config.COMBINED_REPORT_DIR_TEMPLATE.format(patient_id=pid, pipeline_name="language_tracking")
            )
            patient_report_dir.mkdir(parents=True, exist_ok=True)

            out_path = style_utils.stitch_and_save(
                [style_utils.build_patient_panel(pid)] + report_fragments,
                output_path=patient_report_dir / "report.html",
                title=f"{pid} — Language Tracking Summary",
                generator_name="Language Tracking Pipeline",
                extra_css=extra_css,
            )
            typer.echo(f"  Patient report: {out_path}")
