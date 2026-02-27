"""`awakenai list`, `awakenai info`, and `awakenai count` commands.

Pure display layer. All data logic lives in PatientData — these commands
just call PatientData methods and print.
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from src.cli.cli_utils import print_table
from src.data_loading import UnifiedDataLoader

list_app = typer.Typer(help="List patients, sessions, or trials.")
info_app = typer.Typer(help="Show detailed info about a patient or session.")
count_app = typer.Typer(help="Show session or trial counts by type.")

_TRIAL_DISPLAY_COLS = ["trial_id", "date", "trial_type"]
_TRIAL_DETAILED_COLS = ["trial_id", "date", "trial_type", "start_time", "end_time", "duration", "num_events"]
_SESSION_DISPLAY_COLS = ["trial_type", "start_time", "end_time", "duration"]


def _get_loader() -> UnifiedDataLoader:
    return UnifiedDataLoader()


# ─── list commands ─────────────────────────────────────────────────────────────


@list_app.command("patients")
def list_patients() -> None:
    """List all available patient IDs."""
    ids = _get_loader().get_patient_ids()
    typer.echo(f"{len(ids)} patient(s)\n")
    for pid in ids:
        typer.echo(pid)


@list_app.command("sessions")
def list_sessions(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
) -> None:
    """List recording sessions (dates) for a patient."""
    sessions = _get_loader().get_patient(patient_id).list_sessions()
    typer.echo(f"{len(sessions)} session(s)\n")
    for s in sessions:
        typer.echo(s)


@list_app.command("trials")
def list_trials(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Filter by session date")] = None,
    trial_type: Annotated[Optional[str], typer.Option("--type", "-t", help="Filter by trial type")] = None,
    detailed: Annotated[
        bool, typer.Option("--detailed", "-d", help="Show timing columns (start, end, duration)")
    ] = False,
) -> None:
    """List trials for a patient, optionally filtered by session or type."""
    patient = _get_loader().get_patient(patient_id, session=session, trial_type=trial_type)
    if patient.trials_df.empty:
        typer.echo("No trials found for the given filters.")
        raise typer.Exit(0)
    df = patient.trials_df.copy()
    df.insert(0, "trial_id", "")
    df["num_events"] = df["sentences"].str.len().fillna(0).astype(int)

    cols = _TRIAL_DETAILED_COLS if detailed else _TRIAL_DISPLAY_COLS
    typer.echo(f"{len(df)} trial(s)\n")
    print_table(df[df.columns.intersection(cols)], title=f"Trials — {patient_id}")


# ─── count commands ────────────────────────────────────────────────────────────


@count_app.command("trials")
def count_trials(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
    session: Annotated[Optional[str], typer.Option("--session", "-s", help="Filter by session date")] = None,
) -> None:
    """Show trial counts grouped by type for a patient."""
    patient = _get_loader().get_patient(patient_id, session=session)
    counts = patient.trials_df["trial_type"].value_counts()

    typer.echo(f"\nTrials — {patient_id}" + (f" / {session}" if session else ""))
    typer.echo(f"  Total: {counts.sum()}\n")
    for trial_type, n in counts.items():
        typer.echo(f"  {trial_type:<20} {n}")
    typer.echo()


# ─── info commands ─────────────────────────────────────────────────────────────


@info_app.command("patient")
def info_patient(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
) -> None:
    """Show patient summary: sessions, trial counts by type, and clinical metadata."""
    data = _get_loader().get_patient(patient_id).info()

    sessions = data["sessions"]
    last_5 = ", ".join(sessions[-5:])
    sessions_str = f"{len(sessions)} total (last 5: {last_5})" if len(sessions) > 5 else ", ".join(sessions)

    typer.echo(f"\nPatient: {data['patient_id']}")
    typer.echo(f"  Sessions: {sessions_str}")
    typer.echo(f"  Total trials: {data['total_trials']}")

    typer.echo("\n  Trials by type:")
    for trial_type, count in data["trial_counts"].items():
        typer.echo(f"    {trial_type}: {count}")

    if data["first_visit"]:
        total_visits = len(data.get("visit_history", []))
        typer.echo(f"\n  First visit:   {data['first_visit']}")
        typer.echo(f"  Last visit:    {data['last_visit']}")
        if total_visits:
            typer.echo(f"  Total visits:  {total_visits}")

    if data["notes"]:
        typer.echo("\n  Notes:")
        for note in data["notes"]:
            typer.echo(f"    [{note['date']}] {note['notes']}")
    typer.echo()


@info_app.command("session")
def info_session(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
    date: Annotated[str, typer.Argument(help="Session date (e.g. 2025-01-10)")],
) -> None:
    """Show trial types and counts for a specific session."""
    patient = _get_loader().get_patient(patient_id, session=date)
    if patient.trials_df.empty:
        typer.echo(f"No trials found for {patient_id} on {date}.")
        raise typer.Exit(1)

    typer.echo(f"\nSession: {patient_id} / {date}")
    typer.echo(f"  Trial types: {', '.join(patient.get_trial_types())}")
    typer.echo(f"  Total trials: {len(patient.trials_df)}\n")

    cols = patient.trials_df.columns.intersection(_SESSION_DISPLAY_COLS)
    print_table(patient.trials_df[cols], title="Trials")


@info_app.command("trial")
def info_trial(
    patient_id: Annotated[str, typer.Argument(help="Patient ID (e.g. CON008)")],
    trial_idx: Annotated[int, typer.Argument(help="Trial index (0-based)")],
) -> None:
    """Show details for a specific trial: type, timing, and sentences."""
    trial = _get_loader().get_patient(patient_id).get_trial(trial_idx)

    typer.echo(f"\nTrial {trial_idx} — {patient_id}")
    typer.echo(f"  Type:       {trial.get('trial_type', 'N/A')}")
    typer.echo(f"  Date:       {trial.get('date', 'N/A')}")
    typer.echo(f"  Start:      {trial.get('start_time'):.3f}s" if trial.get("start_time") else "  Start:      N/A")
    typer.echo(f"  End:        {trial.get('end_time'):.3f}s" if trial.get("end_time") else "  End:        N/A")
    typer.echo(f"  Duration:   {trial.get('duration'):.3f}s" if trial.get("duration") else "  Duration:   N/A")

    sentences = trial.get("sentences", [])
    if len(sentences) > 0:
        typer.echo(f"\n  Sentences ({len(sentences)}):")
        for s in sentences:
            event = s.get("event", "") if isinstance(s, dict) else s
            typer.echo(f"    {event}")
    typer.echo()


_PIPELINE_DOCS = [
    ("command-following", "Motor imagery (imagine moving left/right hand)."),
    ("language", "Spoken sentence comprehension (listen & respond)."),
    ("oddball", "Auditory attention (standard vs. deviant sequences)."),
]


@info_app.command("trial-types")
def info_trial_types() -> None:
    """Describe the 3 main analysis pipelines and what they do."""
    typer.echo("\nAnalysis Pipelines\n")
    typer.echo(f"  {'Pipeline':<20} Description")
    typer.echo("  " + "─" * 75)
    for pipeline, desc in _PIPELINE_DOCS:
        typer.echo(f"  {pipeline:<20} {desc}")
    typer.echo()
