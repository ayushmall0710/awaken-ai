"""Oddball pipeline runner — stub, coming soon."""

from __future__ import annotations

from typing import Optional

import typer

from src.data_loading import UnifiedDataLoader
from src.pipelines.p300_oddball import P300OddballPipeline


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    electrodes: Optional[list[str]],
) -> None:
    """Run P300OddballPipeline for the given patients (all sessions or single --session date)."""
    pipeline = P300OddballPipeline(loader=loader)

    for pid in patient_ids:
        typer.echo(f"[oddball] {pid}" + (f" (session {session})" if session else " (all sessions)") + " ...")
        try:
            df = pipeline.run(pid, session=session, custom_electrodes=electrodes)
            summary = pipeline.generate_summary()

            if df is None or df.empty:
                typer.echo("  No oddball data or features.")
                continue

            typer.echo(
                f"  Status: {summary.get('status', 'UNKNOWN')} | "
                f"sessions={summary.get('n_sessions', 0)} | "
                f"mean_amp={summary.get('mean_amplitude_uV', float('nan')):.2f}µV | "
                f"mean_lat={summary.get('mean_latency_ms', float('nan')):.1f}ms"
            )
        except Exception as e:  # pragma: no cover - defensive
            typer.echo(f"  ✗ Failed: {e}", err=True)
