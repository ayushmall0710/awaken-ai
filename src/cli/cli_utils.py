"""Shared CLI utilities."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import typer

from src.data_loading import UnifiedDataLoader
from src.data_loading.unified_data_loader import UnifiedDataLoadingError


def get_loader() -> UnifiedDataLoader:
    """Instantiate the data loader or fail gracefully with setup instructions."""
    try:
        return UnifiedDataLoader()
    except UnifiedDataLoadingError as e:
        typer.echo(f"\n[Error] {e}", err=True)
        typer.echo("-> Run 'awakenai unify-data' to generate the required dataset first.\n", err=True)
        raise typer.Exit(1)


def resolve_patients(
    patients: Optional[list[str]],
    all_patients: bool,
    loader: UnifiedDataLoader,
) -> list[str]:
    """Return final list of patient IDs based on CLI args."""
    if all_patients and patients:
        typer.echo("Error: provide either patient IDs or --all, not both.", err=True)
        raise typer.Exit(1)
    if all_patients:
        return loader.get_patient_ids()
    if not patients:
        typer.echo("Error: provide at least one patient ID or --all.", err=True)
        raise typer.Exit(1)
    return list(patients)


def print_table(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """Print a DataFrame as a compact table to stdout."""
    if title:
        typer.echo(f"\n{title}")
        typer.echo("─" * len(title))
    typer.echo(df.to_string(index=False))
    typer.echo()
