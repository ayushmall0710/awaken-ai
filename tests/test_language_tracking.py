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

    # Create 3 epochs, 16 seconds each
    n_epochs = 3
    n_samples = int(16.0 * sfreq)
    data = np.random.randn(n_epochs, len(ch_names), n_samples)

    events = np.array([[0, 0, 1], [16000, 0, 1], [32000, 0, 1]])
    event_id = {"language": 1}

    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0)
    return epochs


@pytest.fixture
def itpc_epochs():
    """
    Epochs suitable for ITPC computation: 256 Hz, 16-second duration.

    ITPC analysis targets 0.05-2.0 Hz. This requires long epochs to resolve
    very low frequencies (resolution = 1/16 = 0.0625 Hz).

    Includes both LH (F7, T7, P7, F3, C3, P3) and RH (F8, T8, F4, C4) channels
    so that lh/rh focus channel subsets are both non-empty.
    """
    ch_names = ["F7", "T7", "P7", "F3", "C3", "P3", "Fz", "F8", "T8", "F4", "C4"]
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

        # Mock get_patient().list_sessions() and list_session_ids()
        mock_patient = MagicMock()
        mock_patient.list_sessions.return_value = ["2024-01-01"]
        mock_patient.list_session_ids.return_value = ["2024-01-01"]
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

    processor.load()
    processor.preprocess()

    assert processor.epochs is not None
    # mne.EpochsArray does not inherit from mne.Epochs, but both share BaseEpochs (which is not always easy to import)
    # So we just check if it looks like epochs
    assert len(processor.epochs) == 3
    mock_loader.load_clean_epochs.assert_called_with("TEST", "2024-01-01", trial_type="language")


def test_pick_channel_subset_lh(mock_language_epochs):
    """LH channel pick returns correct channels."""
    from src.data_loading import config as cfg

    processor = LanguageTrackingAnalysis(MagicMock())
    picked = processor._pick_channel_subset(mock_language_epochs, cfg.LH_FOCUS_CHANNELS)
    assert "F7" in picked.ch_names
    assert "T7" in picked.ch_names
    assert len(picked.ch_names) >= 3


def test_pick_channel_subset_missing():
    """Returns original epochs when no target channels found."""
    processor = LanguageTrackingAnalysis(MagicMock())
    info = mne.create_info(["CH1", "CH2"], 1000.0, ch_types="eeg")
    data = np.zeros((1, 2, 1000))
    bad_epochs = mne.EpochsArray(data, info)
    result = processor._pick_channel_subset(bad_epochs, ["F7", "T7"])
    assert result.ch_names == ["CH1", "CH2"]


def test_pick_channel_subset_clinical(mock_language_epochs):
    """Clinical channel pick returns valid subset of available channels."""
    from src.data_loading import config as cfg

    processor = LanguageTrackingAnalysis(MagicMock())
    picked = processor._pick_channel_subset(mock_language_epochs, cfg.CLINICAL_20)
    assert len(picked.ch_names) >= 1
    assert set(picked.ch_names).issubset(set(mock_language_epochs.ch_names))


def test_preprocess_signal(mock_language_epochs):
    """Test filtering and downsampling in _preprocess_signal."""
    processor = LanguageTrackingAnalysis(MagicMock())

    filtered = processor._preprocess_signal(mock_language_epochs)

    assert filtered.info["highpass"] == pytest.approx(processor.HIGHPASS_FREQ, abs=0.01)
    assert filtered.info["lowpass"] == pytest.approx(processor.LOWPASS_FREQ, abs=0.1)
    # Input is 1000 Hz -- should be downsampled to TARGET_SFREQ
    assert filtered.info["sfreq"] == processor.TARGET_SFREQ


def test_preprocess_signal_no_downsample(itpc_epochs):
    """_preprocess_signal should skip resampling when already at target or below."""
    processor = LanguageTrackingAnalysis(MagicMock())
    # itpc_epochs is already at 256 Hz (TARGET_SFREQ), so no resampling should occur.
    filtered = processor._preprocess_signal(itpc_epochs)
    assert filtered.info["sfreq"] == 256.0


# --- ITPC: Morlet ---


def test_compute_itpc_returns_data_and_itc(itpc_epochs):
    """_compute_itpc returns ndarray and AverageTFR with expected shape."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_data, itc_obj = processor._compute_itpc(itpc_epochs)

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
    """Custom freq/cycle arrays passed to _compute_itpc override class defaults."""
    processor = LanguageTrackingAnalysis(MagicMock())
    custom_freqs = np.array([0.1, 0.5, 1.0])
    custom_cycles = np.array([1.0, 1.0, 1.0])

    itpc_data, _ = processor._compute_itpc(itpc_epochs, freqs=custom_freqs, n_cycles=custom_cycles)

    assert itpc_data.shape[1] == len(custom_freqs)


# --- ITPC: DFT ---


def test_compute_itpc_dft_returns_spectrum(itpc_epochs):
    """_compute_itpc_dft returns spectrum (n_channels, n_freqs) with zero-padded freq axis."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_spectrum, freqs = processor._compute_itpc_dft(itpc_epochs)

    n_channels = len(itpc_epochs.ch_names)
    assert itpc_spectrum.shape[0] == n_channels
    assert itpc_spectrum.shape[1] == len(freqs)
    assert np.all(itpc_spectrum >= 0)
    assert np.all(itpc_spectrum <= 1)
    # Zero-padding must achieve DFT_FREQ_RESOLUTION (0.01 Hz) or finer.
    assert freqs[1] - freqs[0] <= processor.DFT_FREQ_RESOLUTION + 1e-9
    # Sentence and word rate bins must be within half a bin of their targets.
    i_sent = np.argmin(np.abs(freqs - processor.TARGET_SENTENCE_FREQ))
    i_word = np.argmin(np.abs(freqs - processor.TARGET_WORD_FREQ))
    assert abs(freqs[i_sent] - processor.TARGET_SENTENCE_FREQ) <= processor.DFT_FREQ_RESOLUTION / 2 + 1e-9
    assert abs(freqs[i_word] - processor.TARGET_WORD_FREQ) <= processor.DFT_FREQ_RESOLUTION / 2 + 1e-9


# --- Metrics ---


def test_extract_itpc_metrics_structure(itpc_epochs):
    """_extract_itpc_metrics returns all expected keys without single-bin index."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_data, _ = processor._compute_itpc(itpc_epochs)
    metrics = processor._extract_itpc_metrics(itpc_data)

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
    freqs = np.logspace(np.log10(0.05), np.log10(5.0), num=60)
    n_channels = 7
    itpc_data = np.zeros((n_channels, len(freqs), 10))
    metrics = processor._extract_itpc_metrics(itpc_data, freqs=freqs)
    assert metrics["ratio_sent_word"] == 0.0
    assert metrics["ratio_sent_phrase"] == 0.0


def test_band_averaging_uses_multiple_bins(itpc_epochs):
    """_extract_itpc_metrics averages across all bins in SENTENCE_BAND, not just one."""
    processor = LanguageTrackingAnalysis(MagicMock())
    freqs = processor.ITPC_FREQS
    sent_mask = (freqs >= processor.SENTENCE_BAND[0]) & (freqs <= processor.SENTENCE_BAND[1])
    phrase_mask = (freqs >= processor.PHRASE_BAND[0]) & (freqs <= processor.PHRASE_BAND[1])
    word_mask = (freqs >= processor.WORD_BAND[0]) & (freqs <= processor.WORD_BAND[1])

    # At least some Morlet bins must fall inside each band for averaging to be meaningful.
    assert sent_mask.sum() >= 1, "Sentence band must span at least 1 Morlet bin"
    assert phrase_mask.sum() >= 1, "Phrase band must span at least 1 Morlet bin"
    assert word_mask.sum() >= 1, "Word band must span at least 1 Morlet bin"

    # Result must be in valid ITPC range.
    itpc_data, _ = processor._compute_itpc(itpc_epochs)
    metrics = processor._extract_itpc_metrics(itpc_data)
    assert 0.0 <= metrics["itpc_sentence"] <= 1.0
    assert 0.0 <= metrics["itpc_phrase"] <= 1.0
    assert 0.0 <= metrics["itpc_word"] <= 1.0
    # Peak frequency must lie within the respective band.
    assert processor.SENTENCE_BAND[0] <= metrics["freq_sentence_hz"] <= processor.SENTENCE_BAND[1]
    assert processor.PHRASE_BAND[0] <= metrics["freq_phrase_hz"] <= processor.PHRASE_BAND[1]
    assert processor.WORD_BAND[0] <= metrics["freq_word_hz"] <= processor.WORD_BAND[1]


def test_extract_morlet_observed_itpc_matches_null_math(itpc_epochs):
    """
    _extract_morlet_observed_itpc uses identical math to _compute_surrogate_itpc.

    Specifically: observed = |mean_trials(exp(i * angle(mean_t(complex))))|,
    which is the same quantity the null scrambles. A zero-offset surrogate
    must reproduce the observed value exactly (within floating-point precision).
    """
    processor = LanguageTrackingAnalysis(MagicMock())
    processor._morlet_phases = processor._compute_morlet_target_phases(itpc_epochs)
    observed = processor._extract_morlet_observed_itpc()

    # All three metrics are valid ITPC values in [0, 1]
    for key in ("itpc_word", "itpc_phrase", "itpc_sentence"):
        assert key in observed, f"Missing key: {key}"
        assert 0.0 <= observed[key] <= 1.0, f"{key} = {observed[key]} out of range"

    # Verify that manually applying zero random phase offset reproduces the same value.
    phases = processor._morlet_phases  # (n_trials, n_channels, 3)
    for label, freq_idx in processor._MORLET_FREQ_IDX.items():
        unit_vectors = np.exp(1j * phases[:, :, freq_idx])
        manual_itpc = float(np.mean(np.abs(np.mean(unit_vectors, axis=0))))
        assert abs(observed[f"itpc_{label}"] - manual_itpc) < 1e-10, (
            f"itpc_{label}: _extract_morlet_observed_itpc={observed[f'itpc_{label}']:.6f} vs manual={manual_itpc:.6f}"
        )


def test_extract_morlet_observed_itpc_raises_without_phases():
    """_extract_morlet_observed_itpc raises ValueError when _morlet_phases is None."""
    processor = LanguageTrackingAnalysis(MagicMock())
    assert processor._morlet_phases is None
    with pytest.raises(ValueError, match="_morlet_phases not set"):
        processor._extract_morlet_observed_itpc()


# --- Channel selection ---


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


# --- Permutation test ---


def test_permutation_null_shape(itpc_epochs):
    """Null distribution has correct shape and values in [0, 1]."""
    processor = LanguageTrackingAnalysis(MagicMock())
    null = processor._compute_trial_shuffled_null_itpc(itpc_epochs, n_permutations=50, metric="sentence")
    assert null.shape == (50,)
    assert np.all(null >= 0) and np.all(null <= 1)


def test_permutation_null_near_chance(itpc_epochs):
    """Null ITPC mean should be near theoretical chance 1/sqrt(n_trials * n_channels) or roughly 1/sqrt(n_trials)."""
    processor = LanguageTrackingAnalysis(MagicMock())
    n_trials = len(itpc_epochs)
    chance = 1.0 / np.sqrt(n_trials)
    null = processor._compute_trial_shuffled_null_itpc(itpc_epochs, n_permutations=200, metric="sentence", seed=0)
    # The actual null might be slightly off pure chance depending on true phase clustering,
    # but with random permutation it suppresses false positives effectively.
    # Theoretical chance for purely random phase is ~1/sqrt(N)
    assert abs(np.mean(null) - chance) < 0.5 * chance


def test_permutation_pvalue_computation():
    """p-value equals proportion of null >= observed."""
    observed = 0.14
    null = np.array([0.10, 0.12, 0.13, 0.15, 0.20])
    p = LanguageTrackingAnalysis._compute_permutation_pvalue(observed, null)
    assert p == pytest.approx(2 / 5)


def test_permutation_reproducible(itpc_epochs):
    """Same seed produces identical null distributions."""
    processor = LanguageTrackingAnalysis(MagicMock())
    null_a = processor._compute_trial_shuffled_null_itpc(itpc_epochs, n_permutations=30, seed=99)
    null_b = processor._compute_trial_shuffled_null_itpc(itpc_epochs, n_permutations=30, seed=99)
    np.testing.assert_array_equal(null_a, null_b)


# --- Lateralization ---


def test_lateralization_index_values():
    """LI = (LH - RH) / (LH + RH) for typical values."""
    assert LanguageTrackingAnalysis._compute_lateralization_index(0.15, 0.10) == pytest.approx(0.2)
    assert LanguageTrackingAnalysis._compute_lateralization_index(0.10, 0.10) == pytest.approx(0.0)
    assert LanguageTrackingAnalysis._compute_lateralization_index(0.0, 0.0) == 0.0


def test_preprocess_stores_filtered_epochs(mock_loader):
    """preprocess() stores _epochs_filtered before channel selection."""
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    processor.patient_id = "TEST"
    processor.load()
    processor.preprocess()
    assert processor._epochs_filtered is not None
    assert len(processor._epochs_filtered.ch_names) >= len(processor.epochs.ch_names)


def test_compute_hemisphere_itpc_returns_valid_metrics(mock_loader):
    """_compute_hemisphere_itpc returns valid dict for both hemispheres."""
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    processor.patient_id = "TEST"
    processor.load()
    processor.preprocess()
    lh = processor._compute_hemisphere_itpc("LH")
    rh = processor._compute_hemisphere_itpc("RH")
    for d in (lh, rh):
        assert "itpc_sentence" in d
        assert 0.0 <= d["itpc_sentence"] <= 1.0


# --- Band-width fix ---


def test_dft_uses_peak_within_band():
    """DFT extraction uses peak (not mean) within band."""
    processor = LanguageTrackingAnalysis(MagicMock())
    sfreq = 256.0
    n_fft = int(np.ceil(sfreq / processor.DFT_FREQ_RESOLUTION))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    n_channels = 6
    # Uniform ITPC at 0.12 everywhere
    itpc_spectrum = np.ones((n_channels, len(freqs))) * 0.12
    # Inject peak at word target
    peak_idx = np.argmin(np.abs(freqs - processor.TARGET_WORD_FREQ))
    itpc_spectrum[:, peak_idx] = 0.40
    metrics = processor._extract_itpc_metrics_dft(itpc_spectrum, freqs)
    # Peak extraction should capture the 0.40 peak, not average it away
    assert metrics["itpc_word"] > 0.30


def test_bw_normalized_ratio_present(itpc_epochs):
    """Both metric extractors include ratio_bw_normalized key."""
    processor = LanguageTrackingAnalysis(MagicMock())
    itpc_data, _ = processor._compute_itpc(itpc_epochs)
    m_morlet = processor._extract_itpc_metrics(itpc_data)
    assert "ratio_bw_normalized" in m_morlet

    itpc_spectrum, freqs = processor._compute_itpc_dft(itpc_epochs)
    m_dft = processor._extract_itpc_metrics_dft(itpc_spectrum, freqs)
    assert "ratio_bw_normalized" in m_dft


# --- Per-session trajectory ---


def test_run_per_session_returns_one_row_per_session(mock_loader, itpc_epochs):
    """_run_per_session returns one row per session using the clinical focus."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    mock_loader.get_patient.return_value.list_sessions.return_value = ["2024-01-01", "2024-01-02"]
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    df = processor._run_per_session("TEST")
    assert len(df) == 2
    assert "session_date" in df.columns
    assert "itpc_sentence" in df.columns
    assert "focus" in df.columns
    assert (df["focus"] == "clinical").all()


def test_run_per_session_skips_missing_sessions(mock_loader, itpc_epochs):
    """_run_per_session skips sessions where epochs are missing."""
    mock_loader.get_patient.return_value.list_sessions.return_value = ["2024-01-01", "2024-01-02"]
    mock_loader.load_clean_epochs.side_effect = [itpc_epochs, FileNotFoundError("missing")]
    processor = LanguageTrackingAnalysis(loader=mock_loader)
    df = processor._run_per_session("TEST")
    assert len(df) == 1


def test_initialization_no_focus(mock_loader):
    """Pipeline initializes without focus parameter."""
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    assert not hasattr(pipeline, "focus")


def test_analyze_stores_intermediate_arrays(mock_loader, itpc_epochs):
    """analyze() stores _dft_spectrum_full, _dft_freqs, _morlet_itc as attributes."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    pipeline.analyze(n_permutations=10)

    assert pipeline._dft_spectrum_full is not None
    assert pipeline._dft_freqs is not None
    assert pipeline._morlet_itc is not None
    assert pipeline._dft_spectrum_full.ndim == 2


# --- Morlet calibration regression ---


def test_morlet_pvalue_significant_for_phase_locked_data():
    """
    Morlet p-value < 0.05 when all trials share the same phase (perfect locking).

    This is a calibration regression: after aligning observed ITPC and the null
    on the same quantity (time-averaged complex -> angle -> |mean_trials(exp(i*phase))|),
    the permutation test must reliably detect perfect phase-locking.
    """
    rng = np.random.default_rng(0)
    n_trials, n_channels, n_freqs = 20, 6, 3
    # All trials share a single fixed phase per channel/freq — perfect phase-locking.
    fixed_phase = rng.uniform(0, 2 * np.pi, size=(1, n_channels, n_freqs))
    phases = np.broadcast_to(fixed_phase, (n_trials, n_channels, n_freqs)).copy()

    processor = LanguageTrackingAnalysis(MagicMock())
    processor._morlet_phases = phases

    observed = processor._extract_morlet_observed_itpc()
    for key in ("itpc_word", "itpc_phrase", "itpc_sentence"):
        assert observed[key] > 0.95, f"{key} = {observed[key]:.4f} expected near 1.0 for locked phases"

    for metric, seed in (("sentence", 100), ("phrase", 101), ("word", 102)):
        null = processor._compute_trial_shuffled_null_itpc(
            None, n_permutations=200, metric=metric, seed=seed, method="morlet"
        )
        p = processor._compute_permutation_pvalue(observed[f"itpc_{metric}"], null)
        assert p < 0.05, f"morlet_p_{metric} = {p:.3f} not significant for perfect phase-locking"


def test_morlet_pvalue_not_extreme_for_random_phases():
    """
    Morlet p-value not consistently < 0.01 when phases are uniformly random.

    Checks that the permutation test does not spuriously flag noise as significant.
    With 200 surrogates and uniform random phases, p should not be below the
    Bonferroni-like threshold of 0.01.
    """
    rng = np.random.default_rng(42)
    n_trials, n_channels, n_freqs = 20, 6, 3
    phases = rng.uniform(0, 2 * np.pi, size=(n_trials, n_channels, n_freqs))

    processor = LanguageTrackingAnalysis(MagicMock())
    processor._morlet_phases = phases

    observed = processor._extract_morlet_observed_itpc()
    for metric, seed in (("sentence", 200), ("phrase", 201), ("word", 202)):
        null = processor._compute_trial_shuffled_null_itpc(
            None, n_permutations=200, metric=metric, seed=seed, method="morlet"
        )
        p = processor._compute_permutation_pvalue(observed[f"itpc_{metric}"], null)
        assert p > 0.01, f"morlet_p_{metric} = {p:.3f} suspiciously small for random phases"


def test_compute_per_channel_null_dft_shape(itpc_epochs):
    """_compute_per_channel_null_dft returns per-channel null for each target frequency."""
    processor = LanguageTrackingAnalysis(MagicMock())
    n_permutations = 20
    n_channels = len(itpc_epochs.ch_names)

    null = processor._compute_per_channel_null_dft(itpc_epochs, n_permutations=n_permutations, seed=0)

    for metric in ("sentence", "phrase", "word"):
        assert metric in null
        assert null[metric].shape == (n_permutations, n_channels)
        assert np.all(null[metric] >= 0)
        assert np.all(null[metric] <= 1)


def test_compute_per_channel_null_dft_reproducible(itpc_epochs):
    """Same seed produces identical null distributions."""
    processor = LanguageTrackingAnalysis(MagicMock())
    null_a = processor._compute_per_channel_null_dft(itpc_epochs, n_permutations=15, seed=7)
    null_b = processor._compute_per_channel_null_dft(itpc_epochs, n_permutations=15, seed=7)
    for metric in ("sentence", "phrase", "word"):
        np.testing.assert_array_equal(null_a[metric], null_b[metric])


def test_compute_per_channel_null_morlet_shape(itpc_epochs):
    """_compute_per_channel_null_morlet returns per-channel null for 3 target frequencies."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor._morlet_phases = processor._compute_morlet_target_phases(itpc_epochs)
    n_permutations = 20
    n_channels = len(itpc_epochs.ch_names)

    null = processor._compute_per_channel_null_morlet(n_permutations=n_permutations, seed=0)

    for metric in ("sentence", "phrase", "word"):
        assert metric in null
        assert null[metric].shape == (n_permutations, n_channels)
        assert np.all(null[metric] >= 0)
        assert np.all(null[metric] <= 1)


def test_compute_per_channel_null_morlet_raises_without_phases():
    """Raises ValueError when _morlet_phases is not set."""
    processor = LanguageTrackingAnalysis(MagicMock())
    with pytest.raises(ValueError, match="_morlet_phases not set"):
        processor._compute_per_channel_null_morlet()


def test_compute_per_channel_itpc_morlet_shape(itpc_epochs):
    """_compute_per_channel_itpc_morlet returns (n_channels, 3) array in [0, 1]."""
    processor = LanguageTrackingAnalysis(MagicMock())
    phases = processor._compute_morlet_target_phases(itpc_epochs)
    n_channels = len(itpc_epochs.ch_names)

    per_ch = processor._compute_per_channel_itpc_morlet(phases)

    assert per_ch.shape == (n_channels, 3)
    assert np.all(per_ch >= 0)
    assert np.all(per_ch <= 1)


def test_per_channel_itpc_morlet_consistent_with_observed(itpc_epochs):
    """Per-channel morlet averaged over all channels matches _extract_morlet_observed_itpc."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor._morlet_phases = processor._compute_morlet_target_phases(itpc_epochs)

    per_ch = processor._compute_per_channel_itpc_morlet(processor._morlet_phases)
    per_ch_avg = np.mean(per_ch, axis=0)  # (3,)

    observed = processor._extract_morlet_observed_itpc()

    assert abs(per_ch_avg[processor._MORLET_FREQ_IDX["sentence"]] - observed["itpc_sentence"]) < 1e-10
    assert abs(per_ch_avg[processor._MORLET_FREQ_IDX["word"]] - observed["itpc_word"]) < 1e-10


# --- _resolve_focuses ---


def test_resolve_focuses_keys(mock_language_epochs):
    """_resolve_focuses returns dict with exactly the 4 focus keys."""
    processor = LanguageTrackingAnalysis(MagicMock())
    focuses = processor._resolve_focuses(mock_language_epochs.ch_names, optimal_channels=["F7", "T7"])
    assert set(focuses.keys()) == {"clinical", "lh", "rh", "optimal"}


def test_resolve_focuses_channels_subset_of_available(mock_language_epochs):
    """All channels in every focus are present in available_ch_names."""
    processor = LanguageTrackingAnalysis(MagicMock())
    available = mock_language_epochs.ch_names
    focuses = processor._resolve_focuses(available, optimal_channels=["F7"])
    for name, chs in focuses.items():
        assert all(ch in available for ch in chs), f"Focus {name} contains unavailable channel"


def test_resolve_focuses_optimal_empty_when_no_cluster():
    """Optimal focus is [] when no cluster channels provided."""
    processor = LanguageTrackingAnalysis(MagicMock())
    focuses = processor._resolve_focuses(["F7", "C3"], optimal_channels=[])
    assert focuses["optimal"] == []


# --- _compute_focus_pvalue ---


def test_compute_focus_pvalue_returns_four_keys(itpc_epochs):
    """_compute_focus_pvalue returns p-values for all 4 metrics."""
    processor = LanguageTrackingAnalysis(MagicMock())
    n_ch = len(itpc_epochs.ch_names)
    per_ch_null = {
        "sentence": np.random.rand(50, n_ch),
        "phrase": np.random.rand(50, n_ch),
        "word": np.random.rand(50, n_ch),
    }
    obs = {"sentence": 0.5, "phrase": 0.4, "word": 0.3}
    p = processor._compute_focus_pvalue(per_ch_null, list(range(n_ch)), obs)

    assert set(p.keys()) == {"p_sentence", "p_phrase", "p_word", "p_comprehension"}
    for v in p.values():
        assert 0.0 <= v <= 1.0


def test_compute_focus_pvalue_is_zero_when_observed_exceeds_all_null():
    """P-value is 0.0 when observed is above every surrogate value."""
    processor = LanguageTrackingAnalysis(MagicMock())
    n_ch, n_surr = 4, 100
    per_ch_null = {k: np.full((n_surr, n_ch), 0.1) for k in ("sentence", "phrase", "word")}
    obs = {"sentence": 0.9, "phrase": 0.9, "word": 0.9}
    p = processor._compute_focus_pvalue(per_ch_null, list(range(n_ch)), obs)
    assert p["p_sentence"] == 0.0
    assert p["p_comprehension"] == 0.0


# --- _build_focus_row ---


def _make_focus_row_args(itpc_epochs, processor):
    """Helper to construct valid per-channel arrays for _build_focus_row tests."""
    ch_names = itpc_epochs.ch_names
    n_ch = len(ch_names)
    sfreq = itpc_epochs.info["sfreq"]
    n_times = itpc_epochs.get_data().shape[2]
    n_pad = int(np.ceil(sfreq / processor.DFT_FREQ_RESOLUTION))
    n_fft = max(n_pad, n_times)
    dft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    n_freqs = len(dft_freqs)
    null = {k: np.full((10, n_ch), 0.1) for k in ("sentence", "phrase", "word")}
    return {
        "clinical_ch_names": ch_names,
        "per_ch_itpc_dft": np.full((n_ch, n_freqs), 0.3),
        "dft_freqs": dft_freqs,
        "per_ch_itpc_morlet": np.full((n_ch, 3), 0.25),
        "per_ch_null_dft": null,
        "per_ch_null_morlet": {k: np.full((10, n_ch), 0.1) for k in ("sentence", "phrase", "word")},
    }


def test_build_focus_row_empty_channels_produces_nans(itpc_epochs):
    """Row with empty channels list has NaN for all ITPC and p-value columns."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor.patient_id = "CON008"
    processor._epochs_filtered = itpc_epochs
    args = _make_focus_row_args(itpc_epochs, processor)

    row = processor._build_focus_row(focus="optimal", channels=[], **args)

    assert row["focus"] == "optimal"
    assert row["channels"] == []
    for col in ("itpc_sentence", "itpc_word", "dft_p_word", "morlet_p_comprehension"):
        assert np.isnan(row[col]), f"{col} should be NaN but got {row[col]}"


def test_build_focus_row_with_channels_produces_values(itpc_epochs):
    """Row with channels produces numeric ITPC and p-values in valid range."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor.patient_id = "CON008"
    processor._epochs_filtered = itpc_epochs
    args = _make_focus_row_args(itpc_epochs, processor)

    row = processor._build_focus_row(focus="lh", channels=itpc_epochs.ch_names[:3], **args)

    assert row["focus"] == "lh"
    assert row["channels"] == itpc_epochs.ch_names[:3]
    assert not np.isnan(row["itpc_sentence"])
    assert 0.0 <= row["dft_p_word"] <= 1.0
    assert 0.0 <= row["morlet_p_comprehension"] <= 1.0
    assert "itpc_comprehension" in row


def test_build_focus_row_column_set(itpc_epochs):
    """Row contains exactly the expected column keys."""
    processor = LanguageTrackingAnalysis(MagicMock())
    processor.patient_id = "CON008"
    processor._epochs_filtered = itpc_epochs
    args = _make_focus_row_args(itpc_epochs, processor)

    row = processor._build_focus_row(focus="clinical", channels=itpc_epochs.ch_names, **args)

    expected_keys = {
        "patient_id",
        "n_trials",
        "focus",
        "channels",
        "itpc_word",
        "itpc_phrase",
        "itpc_sentence",
        "itpc_comprehension",
        "morlet_itpc_word",
        "morlet_itpc_phrase",
        "morlet_itpc_sentence",
        "morlet_itpc_comprehension",
        "dft_p_word",
        "dft_p_phrase",
        "dft_p_sentence",
        "dft_p_comprehension",
        "morlet_p_word",
        "morlet_p_phrase",
        "morlet_p_sentence",
        "morlet_p_comprehension",
    }
    assert expected_keys == set(row.keys())


# --- _select_optimal_channels ---


def test_select_optimal_channels_returns_list_of_strings(itpc_epochs):
    """_select_optimal_channels returns a list of channel name strings."""
    processor = LanguageTrackingAnalysis(MagicMock())
    phases = processor._compute_morlet_target_phases(itpc_epochs)

    # Mock DFT results required by the surgical selection logic
    processor._dft_spectrum_full = np.zeros((len(itpc_epochs.ch_names), 100))
    processor._dft_freqs = np.linspace(0, 10, 100)

    info = itpc_epochs.info.copy()
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    result = processor._select_optimal_channels(
        morlet_phases=phases,
        ch_names=itpc_epochs.ch_names,
        info=info,
        n_permutations=50,
    )

    assert isinstance(result, list)
    assert all(isinstance(ch, str) for ch in result)
    assert all(ch in itpc_epochs.ch_names for ch in result)


def test_select_optimal_channels_sparse_for_random_phases(itpc_epochs):
    """Across 10 random seeds, the mean number of selected channels is below half of total."""
    processor = LanguageTrackingAnalysis(MagicMock())
    info = itpc_epochs.info.copy()
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        pass

    n_trials = len(itpc_epochs)
    n_channels = len(itpc_epochs.ch_names)
    counts = []
    for seed in range(10):
        # Mock DFT results required by the surgical selection logic
        processor._dft_spectrum_full = np.zeros((n_channels, 100))
        processor._dft_freqs = np.linspace(0, 10, 100)

        rng = np.random.default_rng(seed)
        random_phases = rng.uniform(0, 2 * np.pi, size=(n_trials, n_channels, 3))
        result = processor._select_optimal_channels(
            morlet_phases=random_phases,
            ch_names=itpc_epochs.ch_names,
            info=info,
            n_permutations=50,
            seed=seed,
        )
        counts.append(len(result))

    assert float(np.mean(counts)) < n_channels / 2, (
        f"Too many channels selected on average ({np.mean(counts):.1f} / {n_channels}); "
        "cluster permutation may not be working correctly."
    )


def test_select_optimal_channels_graceful_on_bad_info(itpc_epochs):
    """Returns [] gracefully when adjacency computation fails (no montage set)."""
    processor = LanguageTrackingAnalysis(MagicMock())
    phases = processor._compute_morlet_target_phases(itpc_epochs)

    # Mock DFT results required by the surgical selection logic
    processor._dft_spectrum_full = np.zeros((len(itpc_epochs.ch_names), 100))
    processor._dft_freqs = np.linspace(0, 10, 100)

    bare_info = mne.create_info(itpc_epochs.ch_names, itpc_epochs.info["sfreq"], ch_types="eeg")

    result = processor._select_optimal_channels(
        morlet_phases=phases,
        ch_names=itpc_epochs.ch_names,
        info=bare_info,
        n_permutations=50,
    )

    assert result == []


# --- New analyze() long-format tests ---


def test_analyze_returns_four_rows(mock_loader, itpc_epochs):
    """analyze() returns exactly 4 rows, one per focus."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    df = pipeline.analyze(n_permutations=10)

    assert len(df) == 4
    assert set(df["focus"].tolist()) == {"clinical", "lh", "rh", "optimal"}


def test_analyze_required_columns(mock_loader, itpc_epochs):
    """analyze() DataFrame contains all required schema columns."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    df = pipeline.analyze(n_permutations=10)

    required = [
        "patient_id",
        "n_trials",
        "focus",
        "channels",
        "itpc_word",
        "itpc_phrase",
        "itpc_sentence",
        "itpc_comprehension",
        "morlet_itpc_word",
        "morlet_itpc_phrase",
        "morlet_itpc_sentence",
        "morlet_itpc_comprehension",
        "dft_p_word",
        "dft_p_phrase",
        "dft_p_sentence",
        "dft_p_comprehension",
        "morlet_p_word",
        "morlet_p_phrase",
        "morlet_p_sentence",
        "morlet_p_comprehension",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    # Old wide-format columns must NOT be present
    for old_col in (
        "lh_itpc_word",
        "rh_itpc_sentence",
        "lateralization_index_word",
        "ratio_cognitive_acoustic",
        "itpc_comprehension_combined",
    ):
        assert old_col not in df.columns, f"Old column still present: {old_col}"


def test_analyze_channels_column_is_list(mock_loader, itpc_epochs):
    """channels column holds lists; clinical/lh/rh are non-empty."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    df = pipeline.analyze(n_permutations=10)

    for focus in ("clinical", "lh", "rh"):
        row = df[df["focus"] == focus].iloc[0]
        assert isinstance(row["channels"], list)
        assert len(row["channels"]) > 0

    optimal_row = df[df["focus"] == "optimal"].iloc[0]
    assert isinstance(optimal_row["channels"], list)


def test_analyze_optimal_nan_when_no_cluster(mock_loader, itpc_epochs):
    """When _select_optimal_channels returns [], optimal row has NaN metrics."""
    from unittest.mock import patch

    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()

    with patch.object(pipeline, "_select_optimal_channels", return_value=[]):
        df = pipeline.analyze(n_permutations=10)

    optimal_row = df[df["focus"] == "optimal"].iloc[0]
    assert optimal_row["channels"] == []
    assert np.isnan(optimal_row["itpc_sentence"])
    assert np.isnan(optimal_row["dft_p_word"])


def test_analyze_stores_intermediate_arrays_new(mock_loader, itpc_epochs):
    """analyze() still stores _dft_spectrum_full, _dft_freqs, _morlet_itc."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    pipeline.analyze(n_permutations=10)

    assert pipeline._dft_spectrum_full is not None
    assert pipeline._dft_freqs is not None
    assert pipeline._morlet_itc is not None
    assert pipeline._dft_spectrum_full.ndim == 2


# --- generate_summary() long-format tests ---


def test_generate_summary_long_format(mock_loader, itpc_epochs):
    """generate_summary() computes lateralization from lh/rh rows."""
    mock_loader.load_clean_epochs.return_value = itpc_epochs
    pipeline = LanguageTrackingAnalysis(loader=mock_loader)
    pipeline.patient_id = "CON008"
    pipeline.load()
    pipeline.preprocess()
    pipeline.analyze(n_permutations=10)

    summary = pipeline.generate_summary()

    assert "lateralization_index_comprehension" in summary
    assert "ratio_cognitive_acoustic" in summary
    assert "patient_id" in summary
    li = summary["lateralization_index_comprehension"]
    if li is not None:
        assert -1.0 <= li <= 1.0


def test_generate_summary_empty_when_no_results():
    """generate_summary() returns {} when results is None."""
    processor = LanguageTrackingAnalysis(MagicMock())
    assert processor.generate_summary() == {}
