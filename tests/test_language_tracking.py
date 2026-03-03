"""
Unit tests for ENG-05 Language Optimization.
"""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from src.pipelines.language_tracking import LanguageTrackingAnalysis


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
    with patch("src.pipelines.language_tracking.UnifiedDataLoader") as MockLoader:
        loader_instance = MockLoader.return_value

        # Mock get_patient().list_sessions()
        mock_patient = MagicMock()
        mock_patient.list_sessions.return_value = ["2024-01-01"]
        loader_instance.get_patient.return_value = mock_patient

        # Mock load_clean_epochs
        loader_instance.load_clean_epochs.return_value = mock_language_epochs

        yield loader_instance


def test_initialization(mock_loader):
    """Test standard initialization."""
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    assert processor.loader is not None
    assert "F7" in processor.LH_FOCUS_CHANNELS


def test_load_and_preprocess_success(mock_loader):
    """Test load and preprocess work correctly with clean epochs."""
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    processor.patient_id = "TEST"
    processor.focus = "LH"

    processor.load()
    processor.preprocess()

    assert processor.epochs is not None
    # mne.EpochsArray does not inherit from mne.Epochs, but both share BaseEpochs (which is not always easy to import)
    # So we just check if it looks like epochs
    assert len(processor.epochs) == 3
    mock_loader.load_clean_epochs.assert_called_with("TEST", "2024-01-01", trial_type="language")


def test_select_optimal_channels_lh(mock_language_epochs):
    """Test LH channel selection priority on Epochs."""
    processor = LanguageTrackingAnalysis(MagicMock())

    # Run selection
    processed_epochs = processor.select_optimal_channels(mock_language_epochs, focus="LH")

    # Verify LH channels are present (F7, T7, P7)
    current_chs = processed_epochs.ch_names
    assert "F7" in current_chs
    assert "T7" in current_chs
    assert "P7" in current_chs

    # Verify we didn't lose too many
    assert len(current_chs) >= 3


def test_select_optimal_channels_missing():
    """Test behavior when target channels are missing (should warn, not crash)."""
    processor = LanguageTrackingAnalysis(MagicMock())

    # Create epochs with only generic names
    info = mne.create_info(["CH1", "CH2"], 1000.0, ch_types="eeg")
    data = np.zeros((1, 2, 1000))
    bad_epochs = mne.EpochsArray(data, info)

    # Should log warning but return original epochs if NO matches found
    processed_epochs = processor.select_optimal_channels(bad_epochs, focus="LH")
    assert processed_epochs.ch_names == ["CH1", "CH2"]


def test_preprocess_signal(mock_language_epochs):
    """Test filtering and downsampling in preprocess_signal."""
    processor = LanguageTrackingAnalysis(MagicMock())

    filtered = processor.preprocess_signal(mock_language_epochs)

    assert filtered.info["highpass"] == 0.5
    assert filtered.info["lowpass"] == 30.0
    # Input is 1000 Hz -- should be downsampled to TARGET_SFREQ
    assert filtered.info["sfreq"] == processor.TARGET_SFREQ


def test_preprocess_signal_no_downsample(itpc_epochs):
    """preprocess_signal should skip resampling when already at target or below."""
    processor = LanguageTrackingAnalysis(MagicMock())
    # itpc_epochs is already at 256 Hz (TARGET_SFREQ), so no resampling should occur.
    filtered = processor.preprocess_signal(itpc_epochs)
    assert filtered.info["sfreq"] == 256.0


# --- ITPC: Morlet ---


def test_compute_itpc_returns_data_and_itc(itpc_epochs):
    """compute_itpc returns ndarray and AverageTFR with expected shape."""
    processor = LanguageTrackingAnalysis(MagicMock())
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
    processor = LanguageTrackingAnalysis(MagicMock())
    custom_freqs = np.array([0.1, 0.5, 1.0])
    custom_cycles = np.array([1.0, 1.0, 1.0])

    itpc_data, _ = processor.compute_itpc(itpc_epochs, freqs=custom_freqs, n_cycles=custom_cycles)

    assert itpc_data.shape[1] == len(custom_freqs)


# --- ITPC: DFT ---


def test_compute_itpc_dft_returns_spectrum(itpc_epochs):
    """compute_itpc_dft returns spectrum (n_channels, n_freqs) with zero-padded freq axis."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_spectrum, freqs = processor.compute_itpc_dft(itpc_epochs)

    n_channels = len(itpc_epochs.ch_names)
    assert itpc_spectrum.shape[0] == n_channels
    assert itpc_spectrum.shape[1] == len(freqs)
    assert np.all(itpc_spectrum >= 0)
    assert np.all(itpc_spectrum <= 1)
    # Zero-padding must achieve DFT_FREQ_RESOLUTION (0.001 Hz) or finer.
    assert freqs[1] - freqs[0] <= processor.DFT_FREQ_RESOLUTION + 1e-9
    # Sentence and word rate bins must be within half a bin of their targets.
    i_sent = np.argmin(np.abs(freqs - processor.TARGET_SENTENCE_FREQ))
    i_word = np.argmin(np.abs(freqs - processor.TARGET_WORD_FREQ))
    assert abs(freqs[i_sent] - processor.TARGET_SENTENCE_FREQ) <= processor.DFT_FREQ_RESOLUTION / 2
    assert abs(freqs[i_word] - processor.TARGET_WORD_FREQ) <= processor.DFT_FREQ_RESOLUTION / 2


# --- Metrics ---


def test_extract_itpc_metrics_structure(itpc_epochs):
    """extract_itpc_metrics returns all expected keys without single-bin index."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_data, _ = processor.compute_itpc(itpc_epochs)
    metrics = processor.extract_itpc_metrics(itpc_data)

    expected_keys = {
        "itpc_sentence",
        "itpc_word",
        "ratio_sent_word",
        "freq_sentence_hz",
        "freq_word_hz",
    }
    assert expected_keys.issubset(metrics.keys())
    # idx_sentence no longer returned (band-averaged, no single bin)
    assert "idx_sentence" not in metrics
    assert 0 <= metrics["itpc_sentence"] <= 1
    assert 0 <= metrics["itpc_word"] <= 1


def test_extract_itpc_metrics_zero_word():
    """Ratio is 0.0 when all ITPC is zero (division safety check)."""
    processor = LanguageTrackingAnalysis(MagicMock())
    freqs = np.logspace(np.log10(0.05), np.log10(2.0), num=40)
    n_channels = 7
    itpc_data = np.zeros((n_channels, len(freqs), 10))
    metrics = processor.extract_itpc_metrics(itpc_data, freqs=freqs)
    assert metrics["ratio_sent_word"] == 0.0


def test_band_averaging_uses_multiple_bins(itpc_epochs):
    """extract_itpc_metrics averages across all bins in SENTENCE_BAND, not just one."""
    processor = LanguageTrackingAnalysis(MagicMock())
    freqs = processor.ITPC_FREQS
    sent_mask = (freqs >= processor.SENTENCE_BAND[0]) & (freqs <= processor.SENTENCE_BAND[1])
    word_mask = (freqs >= processor.WORD_BAND[0]) & (freqs <= processor.WORD_BAND[1])

    # At least 2 Morlet bins must fall inside each band for averaging to be meaningful.
    assert sent_mask.sum() >= 2, "Sentence band must span at least 2 Morlet bins"
    assert word_mask.sum() >= 2, "Word band must span at least 2 Morlet bins"

    # Result must be in valid ITPC range.
    itpc_data, _ = processor.compute_itpc(itpc_epochs)
    metrics = processor.extract_itpc_metrics(itpc_data)
    assert 0.0 <= metrics["itpc_sentence"] <= 1.0
    assert 0.0 <= metrics["itpc_word"] <= 1.0
    # Peak frequency must lie within the respective band.
    assert processor.SENTENCE_BAND[0] <= metrics["freq_sentence_hz"] <= processor.SENTENCE_BAND[1]
    assert processor.WORD_BAND[0] <= metrics["freq_word_hz"] <= processor.WORD_BAND[1]


# --- Channel selection ---


def test_select_optimal_channels_clinical(mock_language_epochs):
    """Clinical channel focus returns a valid subset."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processed = processor.select_optimal_channels(mock_language_epochs, focus="Clinical")
    assert len(processed.ch_names) >= 1
    # Clinical selection must not add channels that were not in the original
    assert set(processed.ch_names).issubset(set(mock_language_epochs.ch_names))


def test_load_no_data():
    """load raises ValueError when all sessions raise FileNotFoundError."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor.patient_id = "TEST"
    mock_patient = MagicMock()
    mock_patient.list_sessions.return_value = ["2024-01-01", "2024-01-02"]
    processor.loader.get_patient.return_value = mock_patient
    processor.loader.load_clean_epochs.side_effect = FileNotFoundError("no epochs")

    with pytest.raises(ValueError, match="No clean epochs found for TEST\\. Run 'awakenai preprocess' first\\."):
        processor.load()
