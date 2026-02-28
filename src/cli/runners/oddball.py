"""Oddball pipeline runner — stub, coming soon."""

from __future__ import annotations

from typing import Optional

import typer

from src.data_loading import UnifiedDataLoader


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    electrodes: Optional[list[str]],
) -> None:
    """Run OddballPipeline — stub until ENG-02b merges."""
    typer.echo("Oddball pipeline not yet available — waiting for PR #37 to merge.", err=True)
    raise typer.Exit(1)
