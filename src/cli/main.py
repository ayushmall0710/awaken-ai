"""awakenai — CLI entry point.

Usage:
    awakenai setup CON008
    awakenai list patients
    awakenai list sessions CON008
    awakenai list trials CON008 [--session DATE] [--type TYPE]
    awakenai info patient CON008
    awakenai info session CON008 2025-01-10
    awakenai run CON008 [--pipeline PIPELINE] [--session DATE] [--all] [OPTIONS...]
"""

from __future__ import annotations

import types
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.cli.cli_utils import resolve_patients
from src.cli.commands.inspect_cmd import count_app, info_app, list_app
from src.cli.commands.setup_cmd import setup_app
from src.cli.logging_config import setup_logging
from src.data_loading import UnifiedDataLoader, config

__version__ = "0.1.0"

app = typer.Typer(
    name="awakenai",
    help="EEG analysis pipeline orchestrator for awaken-ai.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

app.add_typer(list_app, name="list")
app.add_typer(info_app, name="info")
app.add_typer(count_app, name="count")
app.add_typer(setup_app, name="setup")


@app.command("qc")
def qc_cmd(
    patients: Annotated[
        Optional[list[str]],
        typer.Option(
            "--patient",
            "-p",
            help="Patient ID to include (repeatable, e.g. -p CON008 -p CON009). Omit to use --all.",
        ),
    ] = None,
    all_patients: Annotated[
        bool, typer.Option("--all", "-a", help="Include all available patients")
    ] = False,
    session: Annotated[
        Optional[str], typer.Option("--session", "-s", help="Restrict to a specific session date (YYYY-MM-DD)")
    ] = None,
    output_dir: Annotated[
        Optional[str], typer.Option("--output", "-o", help="Directory to write qc_report.html and qc_summary.csv")
    ] = None,
) -> None:
    """Generate an HTML Quality Control report.

    Examples:\n
      awakenai qc                              # All patients, all sessions\n
      awakenai qc -p CON008                    # Single patient\n
      awakenai qc -p CON008 -p CON009          # Multiple patients\n
      awakenai qc -s 2025-08-14                # Specific session date\n
      awakenai qc -p CON008 -s 2025-08-14     # Patient + session\n
      awakenai qc -a                           # All patients (explicit)\n
      awakenai qc -p CON008 -o /tmp/reports   # Custom output directory\n
    """
    from src.cli.runners import qc_report as qc_runner

    # --all and explicit -p are mutually exclusive
    if all_patients and patients:
        typer.echo("Error: provide patient IDs via -p or use --all, not both.", err=True)
        raise typer.Exit(1)

    # Resolve patient list (None = all — generate_qc_report handles that internally)
    patient_ids: Optional[list[str]] = None
    if patients:
        patient_ids = list(patients)
    elif not all_patients:
        # No flags given — default to all patients (same as --all)
        patient_ids = None

    out_path = None if output_dir is None else Path(output_dir)

    # Print active filters so the user knows what's being generated
    if patient_ids or session:
        typer.echo("QC filters active:")
        if patient_ids:
            typer.echo(f"  Patients : {', '.join(sorted(patient_ids))}")
        if session:
            typer.echo(f"  Session  : {session}")
    else:
        typer.echo("Generating full QC report (all patients, all sessions)...")

    report_path = qc_runner.run(
        patient_ids=patient_ids,
        session=session,
        output_dir=out_path,
    )

    typer.echo(f"\nSuccess! Report written to: {report_path}")



class Pipeline(str, Enum):
    command_following = "command-following"
    language = "language"
    oddball = "oddball"


# Trial-type → Pipeline mapping used for auto-dispatch
_TRIAL_TYPE_TO_PIPELINE: dict[str, Pipeline] = {
    "left_command": Pipeline.command_following,
    "right_command": Pipeline.command_following,
    "language": Pipeline.language,
    "oddball": Pipeline.oddball,
}


def _check_setup(patient_id: str, session: Optional[str], loader: UnifiedDataLoader) -> list[str]:
    """Return list of incomplete setup step names for patient_id."""
    missing = []
    if not (config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet").exists():
        missing.append("timestamp alignment")
    sessions = [session] if session else loader.get_patient_sessions(patient_id)
    if not any((config.EPOCHS_DIR / patient_id / date).exists() for date in sessions):
        missing.append("artifact rejection")
    return missing


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"awakenai v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit"),
    ] = None,
) -> None:
    setup_logging(verbose)


@app.command("run")
def run_cmd(
    patients: Annotated[
        Optional[list[str]],
        typer.Argument(help="Patient IDs to run (e.g. CON008 CON009). Omit if using --all."),
    ] = None,
    all_patients: Annotated[bool, typer.Option("--all", "-a", help="Run on all available patients")] = False,
    pipeline: Annotated[
        Optional[Pipeline],
        typer.Option("--pipeline", "-p", help="Which pipeline to run. Omit to auto-detect from trial types."),
    ] = None,
    session: Annotated[
        Optional[str], typer.Option("--session", "-s", help="Restrict to a specific session date")
    ] = None,
    # command-following specific
    alpha: Annotated[float, typer.Option("--alpha", help="Significance threshold for ERD test")] = 0.05,
    # language specific
    focus: Annotated[str, typer.Option("--focus", help="Channel focus for language pipeline: LH or Clinical")] = "LH",
    # oddball specific
    electrodes: Annotated[
        Optional[str],
        typer.Option("--electrodes", help="Comma-separated electrodes for oddball pipeline (e.g. --electrodes Pz,Cz)"),
    ] = None,
) -> None:
    """Run analysis pipelines for one or more patients.

    Without --pipeline, auto-detects which pipelines apply based on available trial types.
    """
    from src.cli.runners import command_following as cf_runner
    from src.cli.runners import language as lang_runner
    from src.cli.runners import oddball as ob_runner

    loader = UnifiedDataLoader()
    patient_ids = resolve_patients(patients, all_patients, loader)
    _guard_setup(patient_ids, session, loader)

    pipelines_to_run = {pipeline} if pipeline else _detect_pipelines(patient_ids, session, loader)
    if not pipelines_to_run:
        typer.echo("No applicable pipelines found for the given patients/session.", err=True)
        raise typer.Exit(1)

    parsed_electrodes = [e.strip() for e in electrodes.split(",")] if electrodes else None

    typer.echo(f"Running pipelines: {', '.join(p.value for p in sorted(pipelines_to_run, key=lambda x: x.value))}")
    _dispatch_pipelines(
        pipelines_to_run,
        loader,
        patient_ids,
        session,
        alpha,
        focus,
        parsed_electrodes,
        cf_runner,
        lang_runner,
        ob_runner,
    )


def _guard_setup(patient_ids: list[str], session: Optional[str], loader: UnifiedDataLoader) -> None:
    """Check all patients have completed setup. Raises Exit(1) with a Rich panel if not."""
    not_ready = {pid: missing for pid in patient_ids if (missing := _check_setup(pid, session, loader))}
    if not not_ready:
        return

    from rich.console import Console
    from rich.panel import Panel

    console = Console(stderr=True)
    details = "\n".join(f"  {pid}: missing {', '.join(m)}" for pid, m in not_ready.items())
    ids = " ".join(not_ready)
    console.print(
        Panel(
            f"[bold]Setup incomplete[/bold]\n\n{details}\n\n"
            f"Run [cyan bold]awakenai setup {ids}[/cyan bold] to continue.",
            title="[red]Cannot run pipeline[/red]",
            border_style="red",
            padding=(1, 2),
        )
    )
    raise typer.Exit(1)


def _detect_pipelines(patient_ids: list[str], session: Optional[str], loader: UnifiedDataLoader) -> set[Pipeline]:
    """Auto-detect applicable pipelines from trial types across all patients."""
    df = loader.trials_df[loader.trials_df["patient_id"].isin(patient_ids)]
    if session:
        df = df[df["date"] == session]
    return {matched for trial_type in df["trial_type"].unique() if (matched := _TRIAL_TYPE_TO_PIPELINE.get(trial_type))}


def _dispatch_pipelines(
    pipelines: set[Pipeline],
    loader: UnifiedDataLoader,
    patient_ids: list[str],
    session: Optional[str],
    alpha: float,
    focus: str,
    electrodes: Optional[list[str]],
    cf_runner: types.ModuleType,
    lang_runner: types.ModuleType,
    ob_runner: types.ModuleType,
) -> None:
    """Dispatch to each applicable pipeline runner."""
    if Pipeline.command_following in pipelines:
        cf_runner.run(loader, patient_ids, session, alpha)
    if Pipeline.language in pipelines:
        lang_runner.run(loader, patient_ids, session, focus)
    if Pipeline.oddball in pipelines:
        ob_runner.run(loader, patient_ids, session, electrodes)
