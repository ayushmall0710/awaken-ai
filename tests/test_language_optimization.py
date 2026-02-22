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
def itpc_epochs():
    """
    Epochs suitable for ITPC computation: 256 Hz, 16-second duration.

    ITPC analysis targets 0.05-2.0 Hz. This requires long epochs to resolve
    very low frequencies (resolution = 1/16 = 0.0625 Hz).
    """
    ch_names = ["F7", "T7", "P7", "F3", "C3", "P3", "Fz"]
    sfreq = 256.0
    n_samples = int(16.0 * sfreq)  # 4096 samples
    n_epochs = 5

    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    data = np.random.randn(n_epochs, len(ch_names), n_samples) * 1e-5  # V scale

    # Evenly spaced events
    event_samps = (np.arange(n_epochs) * n_samples).astype(int)
    events = np.column_stack([event_samps, np.zeros(n_epochs, int), np.ones(n_epochs, int)])
    event_id = {"language": 1}

    return mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0)


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
    """Test filtering and downsampling in preprocess_signal."""
    processor = LanguageProcessor(MagicMock())

    filtered = processor.preprocess_signal(mock_language_epochs)

    assert filtered.info["highpass"] == 0.5
    assert filtered.info["lowpass"] == 30.0
    # Input is 1000 Hz -- should be downsampled to TARGET_SFREQ
    assert filtered.info["sfreq"] == processor.TARGET_SFREQ


def test_preprocess_signal_no_downsample(itpc_epochs):
    """preprocess_signal should skip resampling when already at target or below."""
    processor = LanguageProcessor(MagicMock())
    # itpc_epochs is already at 256 Hz (TARGET_SFREQ), so no resampling should occur.
    filtered = processor.preprocess_signal(itpc_epochs)
    assert filtered.info["sfreq"] == 256.0


# --- ITPC: Morlet ---


def test_compute_itpc_returns_data_and_itc(itpc_epochs):
    """compute_itpc returns ndarray and AverageTFR with expected shape."""
    processor = LanguageProcessor(MagicMock())
    itpc_data, itc_obj = processor.compute_itpc(itpc_epochs)

    n_channels = len(itpc_epochs.ch_names)
    n_freqs = len(processor.ITPC_FREQS)

    assert isinstance(itpc_data, np.ndarray)
    # Shape must be (n_channels, n_freqs, n_times)
    assert itpc_data.shape[0] == n_channels
    assert itpc_data.shape[1] == n_freqs
    # Values must be in [0, 1] (ITPC is a phase coherence measure)
    assert np.all(itpc_data >= 0)
    assert np.all(itpc_data <= 1)


def test_compute_itpc_custom_freqs(itpc_epochs):
    """Custom freq/cycle arrays passed to compute_itpc override class defaults."""
    processor = LanguageProcessor(MagicMock())
    custom_freqs = np.array([0.1, 0.5, 1.0])
    custom_cycles = np.array([1.0, 1.0, 1.0])

    itpc_data, _ = processor.compute_itpc(itpc_epochs, freqs=custom_freqs, n_cycles=custom_cycles)

    assert itpc_data.shape[1] == len(custom_freqs)


# --- ITPC: DFT ---


def test_compute_itpc_dft_returns_spectrum(itpc_epochs):
    """compute_itpc_dft returns spectrum (n_channels, n_freqs) and freq axis."""
    processor = LanguageProcessor(MagicMock())
    itpc_spectrum, freqs = processor.compute_itpc_dft(itpc_epochs)

    n_channels = len(itpc_epochs.ch_names)
    assert itpc_spectrum.shape[0] == n_channels
    assert itpc_spectrum.shape[1] == len(freqs)
    assert np.all(itpc_spectrum >= 0)
    assert np.all(itpc_spectrum <= 1)
    # Freq resolution ~0.0625 Hz for 16s epochs at 256 Hz
    assert abs(freqs[1] - freqs[0] - 1.0 / 16.0) < 1e-6


# --- Metrics ---


def test_extract_itpc_metrics_structure(itpc_epochs):
    """extract_itpc_metrics returns all expected keys."""
    processor = LanguageProcessor(MagicMock())
    itpc_data, _ = processor.compute_itpc(itpc_epochs)
    metrics = processor.extract_itpc_metrics(itpc_data)

    expected_keys = {
        "itpc_sentence",
        "itpc_word",
        "ratio_sent_word",
        "freq_sentence_hz",
        "freq_word_hz",
        "idx_sentence",
    }

    assert expected_keys.issubset(metrics.keys())


def test_extract_itpc_metrics_zero_word():
    """Ratio is 0.0 when word ITPC is zero (division safety check)."""
    processor = LanguageProcessor(MagicMock())
    freqs = np.logspace(np.log10(0.05), np.log10(2.0), num=40)
    n_channels = 7
    # All zeros -- worst case
    itpc_data = np.zeros((n_channels, len(freqs), 10))
    metrics = processor.extract_itpc_metrics(itpc_data, freqs=freqs)
    assert metrics["ratio_sent_word"] == 0.0


# --- Channel selection ---


def test_select_optimal_channels_clinical(mock_language_epochs):
    """Clinical channel focus returns a valid subset."""
    processor = LanguageProcessor(MagicMock())
    processed = processor.select_optimal_channels(mock_language_epochs, focus="Clinical")
    assert len(processed.ch_names) >= 1
    # Clinical selection must not add channels that were not in the original
    assert set(processed.ch_names).issubset(set(mock_language_epochs.ch_names))


# --- process_patient edge cases ---


def test_process_patient_no_data():
    """process_patient returns None when all sessions raise FileNotFoundError."""
    processor = LanguageProcessor(MagicMock())
    processor.loader.get_patient_sessions.return_value = ["2024-01-01", "2024-01-02"]
    processor.loader.load_clean_epochs.side_effect = FileNotFoundError("no epochs")

    result = processor.process_patient("TEST")
    assert result is None
