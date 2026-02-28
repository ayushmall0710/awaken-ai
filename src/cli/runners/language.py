"""Language Tracking pipeline runner — stub, coming soon."""

from __future__ import annotations

from typing import Optional

import typer

from src.data_loading import UnifiedDataLoader


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    focus: str,
) -> None:
    """Run LanguageTrackingAnalysis — stub until PR #44 merges."""
    typer.echo("Language pipeline not yet available — waiting for PR #44 to merge.", err=True)
    raise typer.Exit(1)
