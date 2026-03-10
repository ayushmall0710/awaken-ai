"""Language Tracking pipeline runner for the awakenai CLI."""

from __future__ import annotations

import datetime
from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader, config
from src.pipelines.language_tracking import LanguageTrackingAnalysis


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
) -> None:
    """Run LanguageTrackingAnalysis for the given patients/sessions."""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()
        for sess in sessions:
            typer.echo(f"[language] {pid} / {sess} ...")
            pipeline = LanguageTrackingAnalysis(loader=loader, session_id=sess)
            try:
                df = pipeline.run(pid)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — Language Tracking Results")

                    # Save results to data/outputs/language/<patient>/<session>/<timestamp>/features.csv
                    out_dir = config.LOCAL_DATA_ROOT / "outputs" / "language" / pid / sess / timestamp
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / "features.csv"

                    write_header = not out_file.exists()
                    df.to_csv(out_file, mode="a", index=False, header=write_header)
                    typer.echo(f"  Saved features to: {out_file}")
            except Exception as e:
                typer.echo(f"  ✗ Failed: {e}", err=True)
