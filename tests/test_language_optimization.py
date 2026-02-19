"""
Unit tests for ENG-05 Language Optimization.
"""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from src.data_processing.language_optimization import LanguageProcessor


@pytest.fixture
def mock_language_epochs():
    """Create a mock Epochs object with language-relevant channels."""
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T7",
        "T8",
        "P7",
        "P8",
        "Fz",
        "Cz",
        "Pz",
        "ECG",
        "EOG",
    ]
    sfreq = 1000.0
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    # Create 3 epochs, 2 seconds each
    n_epochs = 3
    n_samples = int(2.0 * sfreq)
    data = np.random.randn(n_epochs, len(ch_names), n_samples)

    events = np.array([[0, 0, 1], [1000, 0, 1], [2000, 0, 1]])
    event_id = {"language": 1}

    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0)
    return epochs


@pytest.fixture
def mock_loader(mock_language_epochs):
    """Mock UnifiedDataLoader."""
    with patch("src.data_processing.language_optimization.UnifiedDataLoader") as MockLoader:
        loader_instance = MockLoader.return_value

        # Mock get_patient_sessions
        loader_instance.get_patient_sessions.return_value = ["2024-01-01"]

        # Mock load_clean_epochs
        loader_instance.load_clean_epochs.return_value = mock_language_epochs

        yield loader_instance


def test_initialization(mock_loader):
    """Test standard initialization."""
    processor = LanguageProcessor(loader=mock_loader)
    assert processor.loader is not None
    assert "F7" in processor.LH_FOCUS_CHANNELS


def test_process_patient_success(mock_loader):
    """Test process_patient loads clean epochs and returns them."""
    processor = LanguageProcessor(loader=mock_loader)

    epochs = processor.process_patient("TEST", focus="LH")

    assert epochs is not None
    # mne.EpochsArray does not inherit from mne.Epochs, but both share BaseEpochs (which is not always easy to import)
    # So we just check if it looks like epochs
    assert len(epochs) == 3
    mock_loader.load_clean_epochs.assert_called_with("TEST", "2024-01-01", trial_type="language")


def test_select_optimal_channels_lh(mock_language_epochs):
    """Test LH channel selection priority on Epochs."""
    processor = LanguageProcessor(MagicMock())

    # Run selection
    processed_epochs = processor.select_optimal_channels(mock_language_epochs, focus="LH")

    # Verify LH channels are present (F7, T7, P7)
    current_chs = processed_epochs.ch_names
    assert "F7" in current_chs
    assert "T7" in current_chs
    assert "P7" in current_chs

    # Verify we didn't lose too many
    assert len(current_chs) >= 6


def test_select_optimal_channels_missing():
    """Test behavior when target channels are missing (should warn, not crash)."""
    processor = LanguageProcessor(MagicMock())

    # Create epochs with only generic names
    info = mne.create_info(["CH1", "CH2"], 1000.0, ch_types="eeg")
    data = np.zeros((1, 2, 1000))
    bad_epochs = mne.EpochsArray(data, info)

    # Should log warning but return original epochs if NO matches found
    processed_epochs = processor.select_optimal_channels(bad_epochs, focus="LH")
    assert processed_epochs.ch_names == ["CH1", "CH2"]


def test_preprocess_signal(mock_language_epochs):
    """Test filtering parameters on Epochs."""
    processor = LanguageProcessor(MagicMock())

    # Run processing
    filtered = processor.preprocess_signal(mock_language_epochs)

    # Check filter info
    assert filtered.info["highpass"] == 0.5
    assert filtered.info["lowpass"] == 30.0
