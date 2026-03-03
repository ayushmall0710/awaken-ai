"""Language Tracking pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader
from src.pipelines.language_tracking import LanguageTrackingAnalysis


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    focus: str,
) -> None:
    """Run LanguageTrackingAnalysis for the given patients/sessions."""

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_sessions()
        for sess in sessions:
            typer.echo(f"[language] {pid} / {sess} (Focus: {focus}) ...")
            pipeline = LanguageTrackingAnalysis(loader=loader, focus=focus, session_id=sess)
            try:
                df = pipeline.run(pid)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — Language Tracking Results")
            except Exception as e:
                typer.echo(f"  ✗ Failed: {e}", err=True)
