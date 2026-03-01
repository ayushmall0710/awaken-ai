"""Oddball (ENG-02b) pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import typer

from src.data_loading import UnifiedDataLoader
from src.data_processing.erp_pipeline import OddballERPPipeline


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    electrodes: Optional[list[str]],
) -> None:
    """Run OddballERPPipeline for the given patients/sessions."""
    pipeline = OddballERPPipeline(loader=loader)
    n_success = 0

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient_sessions(pid)
        if not sessions:
            typer.echo(f"[oddball] {pid} ... no sessions found", err=True)
            continue

        for sess in sessions:
            typer.echo(f"[oddball] {pid} / {sess} ...")
            try:
                result = pipeline.process_patient(
                    pid,
                    date=sess,
                    custom_electrodes=electrodes,
                )
            except Exception as e:
                typer.echo(f"  ✗ Failed: {e}", err=True)
                continue

            status = result.get("status")
            if status == "success":
                n_success += 1
                _print_success(result)
            else:
                typer.echo(f"  - Skipped ({status})", err=True)

    if n_success == 0:
        raise typer.Exit(1)


def _print_success(result: dict) -> None:
    """Print concise oddball success summary."""
    features = result.get("features")
    if isinstance(features, pd.DataFrame) and not features.empty:
        row = features.iloc[0]
        n_epochs = int(row.get("n_epochs", 0))
        amp = row.get("p300_amplitude_uV")
        lat = row.get("p300_latency_ms")
        if pd.notna(amp) and pd.notna(lat):
            typer.echo(f"  ✓ Success: epochs={n_epochs}, p300={float(amp):.2f}uV @ {float(lat):.1f}ms")
        else:
            typer.echo(f"  ✓ Success: epochs={n_epochs}")
        return

    typer.echo("  ✓ Success")
