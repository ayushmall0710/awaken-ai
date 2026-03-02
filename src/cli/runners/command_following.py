"""Command Following pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader
from src.pipelines.command_following import CommandFollowingAnalysis


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    alpha: float,
) -> None:
    """Run CommandFollowingAnalysis for the given patients/sessions."""
    pipeline = CommandFollowingAnalysis()

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_sessions()

        for sess in sessions:
            typer.echo(f"[command-following] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, alpha=alpha)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — ERD Results")
            except Exception as e:
                typer.echo(f"  ✗ Failed: {e}", err=True)
