"""Oddball (ENG-02b) pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import typer

from src.cli.cli_utils import print_table
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
        sessions = [session] if session else loader.get_patient(pid).list_sessions()

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
                features = result.get("features")
                if isinstance(features, pd.DataFrame) and not features.empty:
                    print_table(features, title=f"{pid} / {sess} — P300 Results")
                else:
                    typer.echo("  ✓ Success")
            else:
                typer.echo(f"  - Skipped ({status})", err=True)

    if n_success == 0:
        raise typer.Exit(1)
