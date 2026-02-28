"""Tests for the awakenai CLI commands."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.get_patient_ids.return_value = ["CON008", "CON009"]

    patient = MagicMock()
    patient.list_sessions.return_value = ["2025-01-10"]
    patient.list_session_ids.return_value = ["s_CON008_20250110"]
    patient.get_trial_types.return_value = ["left_command", "right_command"]

    loader.get_patient.return_value = patient

    loader.get_patient_summary.return_value = pd.DataFrame(
        {"patient_id": ["CON008"], "trial_type": ["left_command"], "count": [10]}
    )
    patient.trials_df = pd.DataFrame(
        {
            "date": ["2025-01-10", "2025-01-10"],
            "trial_type": ["left_command", "right_command"],
            "start_time": [0.0, 10.0],
            "end_time": [5.0, 15.0],
            "duration": [5.0, 5.0],
            "sentences": [["sent1"], ["sent2"]],
        }
    )
    patient.validate.return_value = {"has_trials": True, "edf_loadable": True}
    patient.info.return_value = {
        "patient_id": "CON008",
        "sessions": ["2025-01-10"],
        "total_trials": 2,
        "trial_counts": {"left_command": 1, "right_command": 1},
        "first_visit": "2024-01-01",
        "last_visit": "2025-01-10",
        "notes": [],
    }
    loader.get_patient.return_value = patient
    loader.trials_df = pd.DataFrame(
        {
            "patient_id": ["CON008", "CON008"],
            "date": ["2025-01-10", "2025-01-10"],
            "trial_type": ["left_command", "right_command"],
            "sentences": [["sent1"], ["sent2"]],
        }
    )
    return loader


# ─── list patients ────────────────────────────────────────────────────────────


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_list_patients(MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["list", "patients"])
    assert result.exit_code == 0
    assert "CON008" in result.output
    assert "CON009" in result.output


# ─── list sessions ────────────────────────────────────────────────────────────


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_list_sessions(MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["list", "sessions", "CON008"])
    assert result.exit_code == 0
    assert "s_CON008_20250110" in result.output


# ─── list trials ─────────────────────────────────────────────────────────────


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_list_trials(MockLoader, mock_loader):
    mock_loader.trials_df = pd.DataFrame(
        {
            "patient_id": ["CON008", "CON008"],
            "date": ["2025-01-10", "2025-01-11"],
            "trial_type": ["left_command", "language"],
            "sentences": [["sent1"], ["sent2"]],
        }
    )
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["list", "trials", "CON008"])
    assert result.exit_code == 0
    assert "left_command" in result.output


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_list_trials_filtered_by_type(MockLoader, mock_loader):
    mock_loader.trials_df = pd.DataFrame(
        {
            "patient_id": ["CON008", "CON008"],
            "date": ["2025-01-10", "2025-01-11"],
            "trial_type": ["left_command", "language"],
            "sentences": [["sent1"], ["sent2"]],
        }
    )
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["list", "trials", "CON008", "--type", "left_command"])
    assert result.exit_code == 0
    assert "left_command" in result.output


# ─── info patient ─────────────────────────────────────────────────────────────


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_info_patient(MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["info", "patient", "CON008"])
    assert result.exit_code == 0
    assert "CON008" in result.output
    assert "Sessions" in result.output or "session" in result.output.lower()


# ─── info session ─────────────────────────────────────────────────────────────


@patch("src.cli.commands.inspect_cmd.UnifiedDataLoader")
def test_info_session(MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    result = runner.invoke(app, ["info", "session", "CON008", "s_CON008_20250110"])
    assert result.exit_code == 0
    assert "s_CON008_20250110" in result.output


# ─── run auto-dispatch ────────────────────────────────────────────────────────


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
@patch("src.cli.runners.command_following.run")
def test_run_auto_dispatches_command_following(MockCF_run, mock_config, MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    mock_config.ALIGNED_EVENTS_DIR.__truediv__.return_value.exists.return_value = True
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

    result = runner.invoke(app, ["run", "CON008"])
    assert result.exit_code == 0
    assert "command-following" in result.output


# ─── version ──────────────────────────────────────────────────────────────────


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "awakenai" in result.output


# ─── setup guard session tests ──────────────────────────────────────────────────


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
def test_run_fails_if_session_setup_missing(mock_config, MockLoader, mock_loader):
    """Test that specifying a session strictly checks that session's epochs."""
    MockLoader.return_value = mock_loader
    mock_config.ALIGNED_EVENTS_DIR.return_value.exists.return_value = True

    # Mock EPOCHS_DIR to say "Session 2 doesn't exist"
    mock_session_path = MagicMock()
    mock_session_path.exists.return_value = False
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value = mock_session_path

    result = runner.invoke(app, ["run", "CON008", "--session", "2025-01-11"])
    # It should fail with Exit(1) because epoch dir checks fail
    assert result.exit_code == 1
    assert "Setup incomplete" in result.output
    assert "artifact rejection" in result.output


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
@patch("src.cli.runners.command_following.CommandFollowingAnalysis")
def test_run_passes_if_session_setup_exists(MockCFA, mock_config, MockLoader, mock_loader):
    MockLoader.return_value = mock_loader
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = pd.DataFrame({"erd_dB": [-3.0], "side": ["left"], "band": ["alpha"]})
    MockCFA.return_value = mock_pipeline

    # Mock events and epochs to exist
    mock_config.ALIGNED_EVENTS_DIR.__truediv__.return_value.exists.return_value = True
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

    result = runner.invoke(app, ["run", "CON008", "--session", "2025-01-10", "-p", "command-following"])
    assert result.exit_code == 0
    assert "command-following" in result.output


# ─── run dispatcher permutations tests ───────────────────────────────────────────────────


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
@patch("src.cli.runners.command_following.run")
@patch("src.cli.runners.language.run")
def test_run_multiple_pipelines_auto_detected(mock_lang_run, mock_cf_run, mock_config, MockLoader, mock_loader):
    """Test that multiple trial types trigger multiple pipelines."""
    # Redefine the mock trial types to have both command following and language
    mock_loader.trials_df = pd.DataFrame(
        {
            "patient_id": ["CON008", "CON008"],
            "date": ["2025-01-10", "2025-01-11"],
            "trial_type": ["left_command", "language"],
            "sentences": [["sent1", "sent2"], ["sent3"]],
        }
    )
    MockLoader.return_value = mock_loader
    mock_config.ALIGNED_EVENTS_DIR.__truediv__.return_value.exists.return_value = True
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

    result = runner.invoke(app, ["run", "CON008"])
    assert result.exit_code == 0

    # Verify that BOTH runners were called
    mock_cf_run.assert_called_once()
    mock_lang_run.assert_called_once()


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
@patch("src.cli.runners.oddball.run")
def test_run_explicit_pipeline_args(mock_ob_run, mock_config, MockLoader, mock_loader):
    """Test running generic pipeline specific arguments."""
    MockLoader.return_value = mock_loader
    mock_config.ALIGNED_EVENTS_DIR.__truediv__.return_value.exists.return_value = True
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

    result = runner.invoke(app, ["run", "CON008", "-p", "oddball", "--electrodes", "Pz,Cz"])
    assert result.exit_code == 0

    # Validate electrodes argument was parsed and passed
    kwargs = mock_ob_run.call_args[0]
    assert "Pz" in kwargs[3]
    assert "Cz" in kwargs[3]


@patch("src.cli.main.UnifiedDataLoader")
@patch("src.cli.main.config")
@patch("src.cli.runners.command_following.run")
def test_run_all_patients(mock_cf_run, mock_config, MockLoader, mock_loader):
    """Test running with --all flag uses all patients in DB."""
    mock_loader.trials_df = pd.DataFrame(
        {
            "patient_id": ["CON008", "CON009"],
            "date": ["2025-01-10", "2025-01-10"],
            "trial_type": ["left_command", "left_command"],
            "sentences": [["sent1"], ["sent2"]],
        }
    )
    MockLoader.return_value = mock_loader
    mock_config.ALIGNED_EVENTS_DIR.__truediv__.return_value.exists.return_value = True
    mock_config.EPOCHS_DIR.__truediv__.return_value.__truediv__.return_value.exists.return_value = True

    result = runner.invoke(app, ["run", "--all"])
    assert result.exit_code == 0

    # Assert dispatch received both patients
    passed_patients = mock_cf_run.call_args[0][1]
    assert "CON008" in passed_patients
    assert "CON009" in passed_patients


# ─── --help ───────────────────────────────────────────────────────────────────


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "info" in result.output
    assert "run" in result.output
