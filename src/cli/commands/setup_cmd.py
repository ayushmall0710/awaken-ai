"""`awakenai setup` — guided setup wizard for patient data prerequisites."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import typer

from src.data_loading import UnifiedDataLoader, config

setup_app = typer.Typer(
    help="Set up prerequisites for a patient, then optionally run analysis.", invoke_without_command=True
)


def _events_done(patient_id: str) -> bool:
    return (config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet").exists()


def _epochs_done(patient_id: str, loader: UnifiedDataLoader) -> bool:
    sessions = loader.get_patient(patient_id).list_sessions()
    return any((config.EPOCHS_DIR / patient_id / date).exists() for date in sessions)


def _run_update_patient_records(verbose: bool) -> bool:
    from src.data_loading.digitize_patient_records import main as digitize_main

    digitize_main()
    return True


def _run_timestamp_alignment(patient_id: str, verbose: bool) -> bool:
    from src.data_processing.timestamp_aligner import TimestampAligner

    aligner = TimestampAligner(patient_id=patient_id, verbose=verbose)
    aligner.align(save=True)
    return True


def _run_artifact_rejection(patient_id: str, verbose: bool, loader: UnifiedDataLoader) -> bool:
    from src.data_processing.artifact_rejection import ArtifactRejector

    rejector = ArtifactRejector(loader=loader, verbose=verbose)
    rejector.run([patient_id], save=True)
    return True


def _confirm_step(name: str, done: bool, force: bool) -> bool:
    """Show step status and ask user to confirm. Returns True if should run."""
    if force:
        status = "Complete" if done else "Not complete"
        typer.echo(f"      Status: {status} (force)")
        return True
    if done:
        typer.echo("      Status: Complete")
        return typer.confirm("      Run again?", default=False)
    else:
        typer.echo("      Status: Not complete")
        return typer.confirm("      Run?", default=True)


_PIPELINE_BY_TRIAL_TYPE = {
    "left_command": "command-following",
    "right_command": "command-following",
    "language": "language",
    "oddball": "oddball",
}


def _print_patient_summary(patient_id: str, loader: UnifiedDataLoader) -> None:
    patient = loader.get_patient(patient_id)
    sessions = patient.list_sessions()
    pipelines = sorted(
        {
            _PIPELINE_BY_TRIAL_TYPE[tt]
            for tt in patient.trials_df["trial_type"].unique()
            if tt in _PIPELINE_BY_TRIAL_TYPE
        }
    )
    typer.echo(f"  Sessions ({len(sessions)}): {', '.join(sessions)}")
    typer.echo(f"  Available pipelines: {', '.join(pipelines) or 'none'}")


def _do_setup(patient_id: str, verbose: bool, force: bool, loader: UnifiedDataLoader) -> None:
    """Run all setup steps for a single patient."""
    if not verbose:
        logging.disable(logging.WARNING)

    typer.echo(f"\nSetting up {patient_id}")
    typer.echo("─" * 40)
    _print_patient_summary(patient_id, loader)

    steps = [
        (
            "Update patient records",
            "Syncs clinical visit history and notes from source spreadsheets.\n",
            False,
            lambda: _run_update_patient_records(verbose),
        ),
        (
            "Timestamp alignment",
            "Matches EEG signal timestamps to exact audio trigger events.\n"
            "      Without this, pipelines cannot locate trials within the EEG recording.",
            _events_done(patient_id),
            lambda: _run_timestamp_alignment(patient_id, verbose),
        ),
        (
            "Artifact rejection",
            "Runs ICA to remove eye/muscle noise, detects bad channels, rejects\n"
            "      high-amplitude epochs, and saves clean epoch files for analysis.",
            _epochs_done(patient_id, loader),
            lambda: _run_artifact_rejection(patient_id, verbose, loader),
        ),
    ]

    skipped = []
    for i, (name, description, is_done, run_fn) in enumerate(steps, 1):
        typer.echo(f"\n[{i}/{len(steps)}] {name}")
        typer.echo(f"      {description}")
        if not _confirm_step(name, is_done, force):
            skipped.append(name)
            typer.echo(f"      Skipped. Some analysis steps may not work without {name.lower()}.")
            continue
        run_fn()
        typer.echo("      Done.")

    if not verbose:
        logging.disable(logging.NOTSET)

    if skipped:
        typer.echo(f"\nSetup partially complete for {patient_id}. Skipped: {', '.join(skipped)}.")
    else:
        typer.echo(f"\nSetup complete for {patient_id}.")


@setup_app.callback()
def setup_cmd(
    ctx: typer.Context,
    patients: Annotated[
        Optional[list[str]],
        typer.Argument(help="Patient IDs to set up (e.g. CON008 CON009). Omit if using --all."),
    ] = None,
    all_patients: Annotated[bool, typer.Option("--all", "-a", help="Do setup for all available patients")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output for each step")] = True,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip Y/n prompts and run all steps")] = False,
) -> None:
    """Set up prerequisites (patient records, alignment, artifact rejection) for one or more patients."""
    if ctx.invoked_subcommand is not None:
        return

    from src.cli.cli_utils import resolve_patients

    loader = UnifiedDataLoader()
    patient_ids = resolve_patients(patients, all_patients, loader)

    if all_patients and len(patient_ids) > 1 and not force:
        typer.confirm(f"Run setup for all {len(patient_ids)} patients?", default=True, abort=True)

    for pid in patient_ids:
        _do_setup(pid, verbose, force, loader)

    typer.echo("\nAll done.")


@setup_app.command("run")
def setup_and_run_cmd(
    patients: Annotated[
        Optional[list[str]],
        typer.Argument(help="Patient IDs (e.g. CON008 CON009). Omit if using --all."),
    ] = None,
    all_patients: Annotated[bool, typer.Option("--all", "-a", help="Run for all available patients")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip Y/n prompts and run all setup steps")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed output for each step")] = False,
    pipeline: Annotated[
        Optional[str], typer.Option("--pipeline", "-p", help="Which pipeline to run after setup")
    ] = None,
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Restrict to a specific session")] = None,
) -> None:
    """Set up prerequisites and then run analysis pipelines.

    Runs setup steps first (with Y/n per step unless --force), then dispatches pipelines.
    """
    from src.cli.cli_utils import resolve_patients

    loader = UnifiedDataLoader()
    patient_ids = resolve_patients(patients, all_patients, loader)

    if all_patients and len(patient_ids) > 1 and not force:
        typer.confirm(f"Run setup for all {len(patient_ids)} patients?", default=True, abort=True)

    for pid in patient_ids:
        _do_setup(pid, verbose, force, loader)

    typer.echo("\nSetup done. Starting analysis...\n")

    # Reuse main.py's run dispatch — import lazily to avoid circular refs
    from src.cli.main import Pipeline, _detect_pipelines, _dispatch_pipelines, _guard_setup
    from src.cli.runners import command_following as cf_runner
    from src.cli.runners import language as lang_runner
    from src.cli.runners import oddball as ob_runner

    _guard_setup(patient_ids, loader)
    pl = Pipeline(pipeline) if pipeline else None
    pipelines_to_run = {pl} if pl else _detect_pipelines(patient_ids, session, loader)

    if not pipelines_to_run:
        typer.echo("No applicable pipelines found for the given patients/session.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Running pipelines: {', '.join(p.value for p in sorted(pipelines_to_run, key=lambda x: x.value))}")
    _dispatch_pipelines(
        pipelines_to_run, loader, patient_ids, session, 0.05, "LH", None, cf_runner, lang_runner, ob_runner
    )
