"""Claassen SVM Command Following pipeline runner for the awakenai CLI."""

from __future__ import annotations

from typing import Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader
from src.pipelines.command_following_claassen import CommandFollowingClaassen


def run(
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    alpha: float,
) -> None:
    """Run CommandFollowingClaassen (SVM) for the given patients/sessions."""
    pipeline = CommandFollowingClaassen(loader=loader)

    for pid in patient_ids:
        sessions = [session] if session else loader.get_patient(pid).list_session_ids()

        for sess in sessions:
            typer.echo(f"[command-following-svm] {pid} / {sess} ...")
            try:
                df = pipeline.run(pid, session_id=sess, alpha=alpha)
                if not df.empty:
                    print_table(df, title=f"{pid} / {sess} — SVM Results")

                summary = pipeline.generate_summary()
                typer.echo(f"  Classification: {summary['cmd_status']}")

                for sr in summary.get("side_results", []):
                    typer.echo(
                        f"    {sr['side']}: AUC={sr['auc']:.3f}  "
                        f"Acc={sr['accuracy']:.1%}  "
                        f"p={sr['p_value_perm']:.4f}  "
                        f"{'✓ Significant' if sr['significant'] else '✗ Not significant'}"
                    )
            except Exception as e:
                typer.echo(f"  Failed: {e}", err=True)
