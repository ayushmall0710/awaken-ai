"""Tests for P300OddballPipeline (ENG-02b). All data from fixtures; no real data."""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data_loading import config
from src.pipelines.p300_oddball import (
    ERP_CONFIG,
    STANDARD_EVENT_LABELS,
    P300OddballPipeline,
    SessionData,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_loader():
    """Mock UnifiedDataLoader; load_aligned_events and load_clean_epochs return fixture data."""
    loader = MagicMock()
    loader.load_aligned_events = MagicMock(return_value=pd.DataFrame())
    loader.load_clean_epochs = MagicMock()
    return loader


@pytest.fixture
def mock_eng03_epochs():
    """Synthetic 35s ENG-03 oddball epochs with metadata; channels include Pz, Cz, Fz."""
    sfreq = 512.0
    window_sec = 35.0
    n_channels = 9
    n_samples = int(sfreq * window_sec) + 1
    n_epochs = 3

    ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "Pz", "O1", "O2"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    data = np.random.randn(n_epochs, n_channels, n_samples) * 1e-5

    trial_start_unix = 1704110400.0
    metadata = pd.DataFrame(
        {
            "start_time_unix": [
                trial_start_unix,
                trial_start_unix + 40.0,
                trial_start_unix + 80.0,
            ],
            "end_time_unix": [
                trial_start_unix + window_sec,
                trial_start_unix + 40.0 + window_sec,
                trial_start_unix + 80.0 + window_sec,
            ],
        }
    )

    epochs = mne.EpochsArray(data, info=info, tmin=0.0, baseline=None, verbose=False)
    epochs.metadata = metadata
    return epochs


@pytest.fixture
def aligned_oddball_df(mock_eng03_epochs):
    """Aligned oddball trials; rare event_start timestamps inside mock_eng03_epochs trial windows."""
    trial_start = float(mock_eng03_epochs.metadata["start_time_unix"].iloc[0])
    session_id = "s_P01_20240101"
    return pd.DataFrame(
        [
            {
                "patient_id": "P01",
                "date": "2024-01-01",
                "session_id": session_id,
                "trial_type": "oddball",
                "start_time": trial_start,
                "end_time": trial_start + 35.0,
                "sentences": [
                    {"event": "standard", "event_start": trial_start + 2.0},
                    {"event": "rare", "event_start": trial_start + 5.0, "correlation_score": 0.9},
                    {"event": "standard", "event_start": trial_start + 8.0},
                    {"event": "rare", "event_start": trial_start + 12.0, "correlation_score": 0.85},
                    {"event": "frequent", "event_start": trial_start + 15.0},
                ],
            }
        ]
    )


@pytest.fixture
def mock_evoked():
    """Minimal Evoked with Pz/Cz/Fz and positive deflection in P300 window (300–500 ms)."""
    sfreq = 512.0
    tmin = -0.2
    tmax = 0.7
    n_samples = int((tmax - tmin) * sfreq) + 1
    times = np.linspace(tmin, tmax, n_samples)

    ch_names = ["Fz", "Cz", "Pz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    # Small positive peak in 300–500 ms for P300
    data = np.zeros((3, n_samples))
    window_start, window_end = ERP_CONFIG["p300_window"]
    peak_idx = int((0.4 - tmin) / (tmax - tmin) * (n_samples - 1))
    for ch in range(3):
        data[ch, :] = 1e-6 * np.sin(2 * np.pi * 2 * times)
        data[ch, peak_idx] += 5e-6  # positive deflection

    evoked = mne.EvokedArray(data * 1e6, info=info, tmin=tmin, verbose=False)
    evoked.data = data  # volts; times are derived from tmin/sfreq, do not set .times
    return evoked


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for pipeline outputs; no dependency on repo data/."""
    return tmp_path


@pytest.fixture
def pipeline(mock_loader, temp_output_dir):
    """P300OddballPipeline with mocked loader and temp output dir."""
    return P300OddballPipeline(
        loader=mock_loader,
        output_dir=temp_output_dir,
        verbose=False,
    )


def _make_small_epochs(n_epochs: int = 2, rare_scale: float = 1.0) -> mne.Epochs:
    """Create a small oddball-like epochs object for analyze-path tests."""
    info = mne.create_info(["Fz", "Cz", "Pz"], 512.0, "eeg")
    data = np.zeros((n_epochs, 3, 461))
    p300_idx = 320
    mmn_idx = 180
    data[:, 0, mmn_idx] = -1.5e-6 * rare_scale
    data[:, 1, p300_idx] = 2.0e-6 * rare_scale
    data[:, 2, p300_idx] = 3.0e-6 * rare_scale
    return mne.EpochsArray(data, info=info, tmin=-0.2, verbose=False)


def _make_success_session(session_id: str = "s1") -> SessionData:
    """Create a fully populated success session for analyze-path tests."""
    rare_epochs = _make_small_epochs(n_epochs=2, rare_scale=1.0)
    standard_epochs = _make_small_epochs(n_epochs=2, rare_scale=0.5)
    rare_erp = rare_epochs.average()
    rare_sem = rare_epochs.standard_error()
    standard_erp = standard_epochs.average()
    standard_sem = standard_epochs.standard_error()
    diff_erp = mne.EvokedArray(
        rare_erp.data - standard_erp.data,
        info=rare_erp.info.copy(),
        tmin=rare_erp.tmin,
        verbose=False,
    )
    return SessionData(
        session_id=session_id,
        date="2024-01-01",
        epochs35=MagicMock(),
        status="success",
        epochs=rare_epochs,
        standard_epochs=standard_epochs,
        rare_erp=rare_erp,
        rare_sem=rare_sem,
        standard_erp=standard_erp,
        standard_sem=standard_sem,
        diff_erp=diff_erp,
        n_standard_epochs=len(standard_epochs),
        n_standard_events_candidate=len(standard_epochs),
        mapping_diag={"n_mapped": 2},
    )


def _patch_required_plotters(monkeypatch, pipeline):
    """Replace expensive plot generation with lightweight figures and a tiny GIF frame."""
    monkeypatch.setattr(pipeline.viz, "plot_p300_focus", lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(pipeline.viz, "plot_mmn_focus", lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(pipeline.viz, "plot_erp_figure", lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(pipeline.viz, "plot_erp_image", lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(pipeline.viz, "plot_topomap", lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(pipeline.viz, "animate_topomap", lambda *args, **kwargs: [Image.new("RGB", (8, 8), "white")])


# ── Load ─────────────────────────────────────────────────────────────────────


class TestLoad:
    def test_load_success(self, pipeline, mock_loader, aligned_oddball_df, mock_eng03_epochs):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        mock_loader.load_aligned_events.return_value = aligned_oddball_df
        mock_loader.load_clean_epochs.return_value = mock_eng03_epochs

        pipeline.load()

        assert len(pipeline._session_data) >= 1
        for sid, sess in pipeline._session_data.items():
            assert sess.epochs35 is not None
            assert sess.session_id == sid
            assert sess.date is not None

    def test_load_no_aligned_events(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = None

        with pytest.raises(ValueError, match="No aligned events"):
            pipeline.load()

    def test_load_no_aligned_events_empty_df(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = pd.DataFrame()

        with pytest.raises(ValueError, match="No aligned events"):
            pipeline.load()

    def test_load_no_oddball_trials(self, pipeline, mock_loader):
        pipeline.patient_id = "P01"
        df = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "date": "2024-01-01",
                    "session_id": "s1",
                    "trial_type": "language",
                    "start_time": 0,
                    "end_time": 10,
                    "sentences": [],
                },
            ]
        )
        pipeline.aligned_events = df
        mock_loader.load_aligned_events.return_value = df

        with pytest.raises(ValueError, match="No oddball trials"):
            pipeline.load()

    def test_load_eng03_missing_all_sessions(self, pipeline, mock_loader, aligned_oddball_df):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        mock_loader.load_clean_epochs.side_effect = FileNotFoundError("no epochs")

        with pytest.raises(ValueError, match="Could not load ENG-03 oddball epochs for any session"):
            pipeline.load()

    def test_load_eng03_other_exception_all_sessions(self, pipeline, mock_loader, aligned_oddball_df):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        mock_loader.load_clean_epochs.side_effect = RuntimeError("load failed")

        with pytest.raises(ValueError, match="Could not load ENG-03 oddball epochs for any session"):
            pipeline.load()


# ── Preprocess ───────────────────────────────────────────────────────────────


class TestPreprocess:
    def test_preprocess_without_load_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="load\\(\\) must be called before preprocess"):
            pipeline.preprocess()

    def test_preprocess_insufficient_rare_events(
        self, pipeline, mock_loader, mock_eng03_epochs, aligned_oddball_df, temp_output_dir
    ):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        mock_loader.load_clean_epochs.return_value = mock_eng03_epochs
        pipeline.load()

        # Replace sentences with only one rare event
        pipeline._oddball_trials = pipeline._oddball_trials.copy()
        sid = pipeline._oddball_trials["session_id"].iloc[0]
        pipeline._oddball_trials.loc[pipeline._oddball_trials["session_id"] == sid, "sentences"] = [
            [{"event": "rare", "event_start": 1704110405.0}],
        ]

        pipeline.preprocess()

        sess = list(pipeline._session_data.values())[0]
        assert sess.status == "insufficient_rare_events"

    def test_preprocess_insufficient_epochs_after_mapping(
        self, pipeline, mock_loader, aligned_oddball_df, mock_eng03_epochs
    ):
        """Rare events exist but all unmapped (timestamps outside trial windows) -> insufficient_epochs."""
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        # Epochs with trial windows far in the future so aligned event_starts (trial_start+5, +12) match no window
        sfreq = 512.0
        window_sec = 35.0
        n_samples = int(sfreq * window_sec) + 1
        ch_names = ["Fp1", "Fp2", "F3", "F4", "Fz", "Cz", "Pz", "O1", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = np.random.randn(2, 9, n_samples) * 1e-5
        future_start = 9999999.0
        metadata = pd.DataFrame(
            {
                "start_time_unix": [future_start, future_start + 40.0],
                "end_time_unix": [future_start + window_sec, future_start + 40.0 + window_sec],
            }
        )
        epochs_future = mne.EpochsArray(data, info=info, tmin=0.0, baseline=None, verbose=False)
        epochs_future.metadata = metadata
        mock_loader.load_clean_epochs.return_value = epochs_future
        pipeline.load()
        pipeline.preprocess()

        sess = list(pipeline._session_data.values())[0]
        assert sess.status == "insufficient_epochs"

    def test_preprocess_success_one_session(self, pipeline, mock_loader, aligned_oddball_df, mock_eng03_epochs):
        pipeline.patient_id = "P01"
        pipeline.aligned_events = aligned_oddball_df
        mock_loader.load_clean_epochs.return_value = mock_eng03_epochs
        pipeline.load()
        pipeline.preprocess()

        sess = list(pipeline._session_data.values())[0]
        assert sess.status == "success"
        assert sess.rare_erp is not None
        assert sess.epochs is not None
        assert len(sess.epochs) >= ERP_CONFIG["min_epochs"]

    def test_apply_official_filters_uses_lowpass_and_baseline(self, pipeline, mock_eng03_epochs, monkeypatch):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]
        events = [{"timestamp_unix": trial_start + 10.0, "date": "2024-01-01", "trial_idx": 0}]
        mapped_df, _ = pipeline._map_events_to_trials(events, tw, sfreq=float(mock_eng03_epochs.info["sfreq"]))
        sub = pipeline._extract_subepochs(mock_eng03_epochs, mapped_df)

        calls = {}

        def fake_notch(data, Fs, freqs, method, verbose):
            calls["notch"] = {"Fs": Fs, "freqs": tuple(freqs), "method": method, "shape": data.shape}
            return data + 1.0

        def fake_lowpass(data, sfreq, l_freq, h_freq, method, verbose):
            calls["lowpass"] = {
                "sfreq": sfreq,
                "l_freq": l_freq,
                "h_freq": h_freq,
                "method": method,
                "shape": data.shape,
            }
            return data * 2.0

        monkeypatch.setattr(mne.filter, "notch_filter", fake_notch)
        monkeypatch.setattr(mne.filter, "filter_data", fake_lowpass)

        filtered = pipeline._apply_official_filters(sub)

        assert "notch" not in calls
        assert calls["lowpass"]["h_freq"] == 30.0
        assert calls["lowpass"]["method"] == "iir"
        baseline_mask = (filtered.times >= ERP_CONFIG["tmin"]) & (filtered.times <= 0.0)
        baseline_means = filtered.get_data()[:, :, baseline_mask].mean(axis=-1)
        assert np.allclose(baseline_means, 0.0, atol=1e-12)


# ── Event extraction and mapping ─────────────────────────────────────────────


class TestEventExtractionAndMapping:
    def test_extract_rare_events(self, pipeline, aligned_oddball_df):
        rare = pipeline._extract_rare_events(aligned_oddball_df)
        assert len(rare) == 2
        assert all(e["timestamp_unix"] for e in rare)

    def test_extract_rare_events_missing_sentences(self, pipeline):
        df = pd.DataFrame([{"date": "2024-01-01"}])  # no sentences
        rare = pipeline._extract_rare_events(df)
        assert len(rare) == 0

    def test_extract_rare_events_sentences_not_list(self, pipeline):
        df = pd.DataFrame([{"date": "2024-01-01", "sentences": "not-a-list"}])
        rare = pipeline._extract_rare_events(df)
        assert len(rare) == 0

    def test_extract_standard_events(self, pipeline, aligned_oddball_df):
        standard = pipeline._extract_standard_events(aligned_oddball_df)
        assert len(standard) >= 2  # standard + frequent

    def test_build_trial_windows(self, pipeline, mock_eng03_epochs):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        assert set(tw.columns) == {"eng03_epoch_idx", "start_time_unix", "end_time_unix", "window_sec"}
        assert len(tw) == 3
        assert tw["window_sec"].iloc[0] == 35.0

    def test_build_trial_windows_no_metadata_raises(self, pipeline):
        info = mne.create_info(["Fz"], 256.0, "eeg")
        epochs = mne.EpochsArray(np.zeros((1, 1, 100)), info=info, verbose=False)
        with pytest.raises(ValueError, match="no metadata"):
            pipeline._build_trial_windows(epochs)

    def test_build_trial_windows_missing_start_time_unix_raises(self, pipeline, mock_eng03_epochs):
        mock_eng03_epochs.metadata = mock_eng03_epochs.metadata.drop(columns=["start_time_unix"])
        with pytest.raises(ValueError, match="start_time_unix"):
            pipeline._build_trial_windows(mock_eng03_epochs)

    def test_map_events_to_trials_all_mapped(self, pipeline, mock_eng03_epochs):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]
        events = [
            {"timestamp_unix": trial_start + 5.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 15.0, "date": "2024-01-01", "trial_idx": 0},
        ]
        mapped_df, diag = pipeline._map_events_to_trials(events, tw, sfreq=float(mock_eng03_epochs.info["sfreq"]))
        assert diag["n_rare_events"] == 2
        assert diag["n_mapped"] == 2
        assert diag["n_unmapped"] == 0
        assert len(mapped_df) == 2

    def test_map_events_to_trials_some_unmapped(self, pipeline, mock_eng03_epochs):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]
        events = [
            {"timestamp_unix": trial_start + 5.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 9999.0, "date": "2024-01-01", "trial_idx": 0},
        ]
        _, diag = pipeline._map_events_to_trials(events, tw, sfreq=float(mock_eng03_epochs.info["sfreq"]))
        assert diag["n_mapped"] == 1
        assert diag["n_unmapped"] == 1

    def test_map_events_to_trials_boundary_clip(self, pipeline, mock_eng03_epochs):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]
        events = [{"timestamp_unix": trial_start + 0.05, "date": "2024-01-01", "trial_idx": 0}]
        _, diag = pipeline._map_events_to_trials(events, tw, sfreq=float(mock_eng03_epochs.info["sfreq"]))
        assert diag["n_boundary_clipped"] == 1
        assert diag["n_mapped"] == 0

    def test_extract_subepochs_non_empty(self, pipeline, mock_eng03_epochs):
        tw = pipeline._build_trial_windows(mock_eng03_epochs)
        trial_start = tw["start_time_unix"].iloc[0]
        events = [
            {"timestamp_unix": trial_start + 10.0, "date": "2024-01-01", "trial_idx": 0},
            {"timestamp_unix": trial_start + 20.0, "date": "2024-01-01", "trial_idx": 0},
        ]
        mapped_df, _ = pipeline._map_events_to_trials(events, tw, sfreq=float(mock_eng03_epochs.info["sfreq"]))
        sub = pipeline._extract_subepochs(mock_eng03_epochs, mapped_df)
        assert len(sub) == 2
        assert sub.get_data().shape[1] == len(mock_eng03_epochs.ch_names)
        assert np.isclose(sub.tmin, ERP_CONFIG["tmin"], atol=1.0 / mock_eng03_epochs.info["sfreq"])

    def test_extract_subepochs_empty(self, pipeline, mock_eng03_epochs):
        empty_mapped = pd.DataFrame(
            columns=["timestamp_unix", "eng03_epoch_idx", "offset_sec", "start_sample", "end_sample"]
        )
        sub = pipeline._extract_subepochs(mock_eng03_epochs, empty_mapped)
        assert len(sub) == 0
        assert hasattr(sub, "info")


# ── P300 quantification and validation ─────────────────────────────────────────


class TestP300QuantificationAndValidation:
    def test_detect_p300_peak_valid(self, pipeline, mock_evoked):
        out = pipeline._detect_p300_peak(mock_evoked, "Pz")
        assert "amplitude" in out and "latency" in out
        assert not np.isnan(out["amplitude"])
        assert not np.isnan(out["latency"])

    def test_detect_p300_peak_electrode_missing(self, pipeline, mock_evoked):
        out = pipeline._detect_p300_peak(mock_evoked, "Oz")
        assert np.isnan(out["amplitude"]) and np.isnan(out["latency"])

    def test_validate_p300_electrode_valid(self, pipeline):
        v = pipeline._validate_p300_electrode("Pz", 4.0, 400.0, "P01")
        assert v["is_valid"] is True
        assert v["issues"] == []

    def test_validate_p300_electrode_nan(self, pipeline):
        v = pipeline._validate_p300_electrode("Pz", float("nan"), 400.0, "P01")
        assert v["is_valid"] is False
        assert "missing_data" in v["issues"]

    def test_validate_p300_electrode_negative_amplitude(self, pipeline):
        v = pipeline._validate_p300_electrode("Pz", -1.0, 400.0, "P01")
        assert v["is_valid"] is False
        assert "negative_or_zero_amplitude" in v["issues"]

    def test_validate_p300_electrode_latency_out_of_range(self, pipeline):
        v = pipeline._validate_p300_electrode("Pz", 4.0, 100.0, "P01")
        assert v["is_valid"] is False
        assert "latency_out_of_range" in v["issues"]

    def test_validate_p300_electrode_latency_atypical(self, pipeline):
        v = pipeline._validate_p300_electrode("Pz", 4.0, 550.0, "P01")
        assert v["is_expected_latency"] is False
        assert "latency_atypical" in v["issues"]

    def test_compute_composite_p300_all_valid(self, pipeline, mock_evoked):
        composite = pipeline._compute_composite_p300(mock_evoked, "P01")
        assert composite["n_valid_electrodes"] >= 1
        assert not np.isnan(composite["composite_amplitude"])
        assert composite["best_electrode"] is not None
        assert composite["p300_subtype"] in ("P3a", "P3b", "mixed", "absent")

    def test_quantify_p300_default(self, pipeline, mock_evoked):
        pipeline._last_epoch_diagnostics = {"n_mapped": 5, "n_rare_events": 6, "mapping_rate": 0.83}
        features = pipeline._quantify_p300(
            mock_evoked,
            patient_id="P01",
            session_id="s1",
            date="2024-01-01",
            n_epochs=5,
            custom_electrodes=None,
        )
        assert "p300_amplitude_uV" in features
        assert "p300_latency_ms" in features
        assert "qc_notes" in features
        assert "p300_composite_amplitude_uV" in features
        assert features.get("n_mapped") == 5

    def test_quantify_p300_custom_electrodes(self, pipeline, mock_evoked):
        features = pipeline._quantify_p300(
            mock_evoked,
            patient_id="P01",
            session_id="s1",
            date="2024-01-01",
            n_epochs=5,
            custom_electrodes=["Pz", "Cz"],
        )
        assert "p300_amplitude_Pz_uV" in features
        assert "p300_amplitude_Cz_uV" in features
        assert "Custom electrode analysis" in features.get("qc_notes", "")


# ── Analyze and run ───────────────────────────────────────────────────────────


class TestAnalyzeAndRun:
    def test_analyze_skips_non_success_sessions(self, pipeline, monkeypatch):
        pipeline.patient_id = "P01"
        pipeline._oddball_trials = pd.DataFrame([{"session_id": "s1", "date": "2024-01-01"}])
        sess_success = _make_success_session("s1")
        sess_fail = SessionData(
            session_id="s2",
            date="2024-01-02",
            epochs35=MagicMock(),
            status="insufficient_epochs",
        )
        pipeline._session_data = {"s1": sess_success, "s2": sess_fail}
        _patch_required_plotters(monkeypatch, pipeline)
        monkeypatch.setattr(pipeline, "_save_outputs", lambda **kwargs: None)

        df = pipeline.analyze()

        assert len(df) == 1
        assert df.iloc[0]["session_id"] == "s1"

    def test_analyze_uses_session_objects_for_stats_features_and_saved_outputs(self, pipeline, monkeypatch):
        pipeline.patient_id = "P01"
        session = _make_success_session("s1")
        pipeline._session_data = {"s1": session}

        expected_features = {
            "patient_id": "P01",
            "session_id": "s1",
            "date": "2024-01-01",
            "n_epochs": len(session.epochs),
            "p300_amplitude_uV": 3.0,
            "p300_latency_ms": 400.0,
            "p300_amplitude_Pz_uV": 3.0,
            "p300_latency_Pz_ms": 400.0,
            "diff_amplitude_Pz_uV": 2.5,
            "diff_latency_Pz_ms": 450.0,
            "diff_mmn_amplitude_Fz_uV": -1.0,
            "diff_mmn_latency_Fz_ms": 150.0,
            "p300_n_valid_electrodes": 3,
            "p300_subtype": "P3b",
            "qc_notes": "ok",
        }

        def fake_sig(rare_epochs, standard_epochs):
            assert rare_epochs is session.epochs
            assert standard_epochs is session.standard_epochs
            return {
                "p300_p_value": 0.04,
                "p300_t_stat": 2.1,
                "p300_n_rare": len(rare_epochs),
                "p300_n_standard": len(standard_epochs),
            }

        def fake_quantify(
            erp,
            patient_id,
            session_id,
            date,
            n_epochs,
            custom_electrodes=None,
            diff_erp=None,
            n_standard_epochs=None,
            n_standard_events_candidate=None,
        ):
            assert erp is session.rare_erp
            assert diff_erp is session.diff_erp
            assert patient_id == "P01"
            assert session_id == "s1"
            assert date == "2024-01-01"
            assert n_epochs == len(session.epochs)
            assert n_standard_epochs == len(session.standard_epochs)
            assert n_standard_events_candidate == len(session.standard_epochs)
            return expected_features.copy()

        def fake_save_outputs(patient_id, session_id, epochs, erp, features, standard_erp=None, diff_erp=None):
            assert patient_id == "P01"
            assert session_id == "s1"
            assert epochs is session.epochs
            assert erp is session.rare_erp
            assert standard_erp is session.standard_erp
            assert diff_erp is session.diff_erp
            assert features["p300_p_value"] == 0.04

        monkeypatch.setattr(pipeline, "_compute_p300_significance", fake_sig)
        monkeypatch.setattr(pipeline, "_quantify_p300", fake_quantify)
        monkeypatch.setattr(pipeline, "_save_outputs", fake_save_outputs)
        _patch_required_plotters(monkeypatch, pipeline)

        df = pipeline.analyze()

        assert len(df) == 1
        assert df.iloc[0]["session_id"] == "s1"
        assert df.iloc[0]["p300_p_value"] == 0.04

    def test_analyze_writes_required_plots_including_erp_image(self, pipeline, monkeypatch):
        pipeline.patient_id = "P01"
        session = _make_success_session("s1")
        pipeline._session_data = {"s1": session}

        stale_paths = pipeline._session_plot_paths("P01", "s1")
        for path in stale_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stale")

        monkeypatch.setattr(
            pipeline,
            "_compute_p300_significance",
            lambda rare_epochs, standard_epochs: {
                "p300_p_value": 0.04,
                "p300_t_stat": 2.0,
                "p300_n_rare": len(rare_epochs),
                "p300_n_standard": len(standard_epochs),
            },
        )
        monkeypatch.setattr(
            pipeline,
            "_quantify_p300",
            lambda *args, **kwargs: {
                "patient_id": "P01",
                "session_id": "s1",
                "date": "2024-01-01",
                "n_epochs": len(session.epochs),
                "p300_amplitude_uV": 3.0,
                "p300_latency_ms": 400.0,
                "p300_amplitude_Pz_uV": 3.0,
                "p300_latency_Pz_ms": 400.0,
                "diff_amplitude_Pz_uV": 2.5,
                "diff_latency_Pz_ms": 450.0,
                "diff_mmn_amplitude_Fz_uV": -1.0,
                "diff_mmn_latency_Fz_ms": 150.0,
                "p300_n_valid_electrodes": 3,
                "p300_subtype": "P3b",
                "qc_notes": "ok",
                "n_standard_epochs": len(session.standard_epochs),
            },
        )
        monkeypatch.setattr(pipeline, "_save_outputs", lambda **kwargs: None)
        _patch_required_plotters(monkeypatch, pipeline)

        pipeline.analyze()

        assert stale_paths["p300"].exists()
        assert stale_paths["mmn"].exists()
        assert stale_paths["erp"].exists()
        assert stale_paths["erp_image"].exists()
        assert stale_paths["topomap"].exists()
        assert stale_paths["topomap_gif"].exists()

    def test_analyze_deletes_stale_plots_and_fails_hard_on_plot_error(self, pipeline, monkeypatch):
        pipeline.patient_id = "P01"
        session = _make_success_session("s1")
        pipeline._session_data = {"s1": session}

        stale_paths = pipeline._session_plot_paths("P01", "s1")
        for path in stale_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stale")

        monkeypatch.setattr(
            pipeline,
            "_compute_p300_significance",
            lambda rare_epochs, standard_epochs: {
                "p300_p_value": 0.04,
                "p300_t_stat": 2.0,
                "p300_n_rare": len(rare_epochs),
                "p300_n_standard": len(standard_epochs),
            },
        )
        monkeypatch.setattr(
            pipeline,
            "_quantify_p300",
            lambda *args, **kwargs: {
                "patient_id": "P01",
                "session_id": "s1",
                "date": "2024-01-01",
                "n_epochs": len(session.epochs),
                "p300_amplitude_uV": 3.0,
                "p300_latency_ms": 400.0,
                "p300_amplitude_Pz_uV": 3.0,
                "p300_latency_Pz_ms": 400.0,
                "diff_amplitude_Pz_uV": 2.5,
                "diff_latency_Pz_ms": 450.0,
                "diff_mmn_amplitude_Fz_uV": -1.0,
                "diff_mmn_latency_Fz_ms": 150.0,
                "p300_n_valid_electrodes": 3,
                "p300_subtype": "P3b",
                "qc_notes": "ok",
            },
        )
        monkeypatch.setattr(pipeline, "_save_outputs", lambda **kwargs: None)
        monkeypatch.setattr(pipeline.viz, "plot_p300_focus", lambda *args, **kwargs: plt.figure())
        monkeypatch.setattr(
            pipeline.viz,
            "plot_mmn_focus",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
        )

        with pytest.raises(RuntimeError, match="plot failed"):
            pipeline.analyze()

        assert all(not path.exists() for path in stale_paths.values())

    def test_run_returns_dataframe(self, pipeline, mock_loader, aligned_oddball_df, mock_eng03_epochs, monkeypatch):
        mock_loader.load_aligned_events.return_value = aligned_oddball_df
        mock_loader.load_clean_epochs.return_value = mock_eng03_epochs
        _patch_required_plotters(monkeypatch, pipeline)

        result = pipeline.run("P01")

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        assert "session_id" in result.columns
        assert "p300_amplitude_uV" in result.columns or "p300_amplitude_Pz_uV" in result.columns

    def test_run_custom_electrodes(self, pipeline, mock_loader, aligned_oddball_df, mock_eng03_epochs, monkeypatch):
        mock_loader.load_aligned_events.return_value = aligned_oddball_df
        mock_loader.load_clean_epochs.return_value = mock_eng03_epochs
        _patch_required_plotters(monkeypatch, pipeline)

        result = pipeline.run("P01", custom_electrodes=["Pz", "Cz"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        assert "p300_amplitude_Pz_uV" in result.columns
        assert "p300_amplitude_Cz_uV" in result.columns


# ── Table builders and persistence ───────────────────────────────────────────


class TestTableBuildersAndPersistence:
    def test_build_clinical_table(self, pipeline):
        features = {
            "patient_id": "P01",
            "session_id": "s1",
            "date": "2024-01-01",
            "n_epochs": 5,
            "n_standard_epochs": 3,
            "baseline_std_uV": 0.5,
            "p300_amplitude_uV": 4.0,
            "p300_latency_ms": 380,
            "p300_best_electrode": "Pz",
            "p300_subtype": "P3b",
            "p300_n_valid_electrodes": 3,
            "qc_notes": "ok",
        }
        df = pipeline._build_clinical_table("P01", "s1", features)
        assert len(df) == 1
        assert "session_id" in df.columns
        assert "session_date" in df.columns
        assert "qc_pass" in df.columns
        assert df.iloc[0]["session_id"] == "s1"

    def test_build_electrode_detail_table(self, pipeline):
        features = {
            "date": "2024-01-01",
            "p300_amplitude_Pz_uV": 4.0,
            "p300_latency_Pz_ms": 380,
            "p300_amplitude_Cz_uV": 3.0,
            "p300_latency_Cz_ms": 390,
            "p300_amplitude_Fz_uV": 2.0,
            "p300_latency_Fz_ms": 400,
        }
        df = pipeline._build_electrode_detail_table("P01", "s1", features)
        assert len(df) == 3
        assert set(df["electrode"].unique()) == {"Fz", "Cz", "Pz"}
        assert "is_valid" in df.columns
        assert "flagged_reason" in df.columns

    def test_build_mapping_qc_table(self, pipeline):
        features = {
            "date": "2024-01-01",
            "n_rare_events": 10,
            "n_mapped": 8,
            "n_unmapped": 2,
            "n_boundary_clipped": 0,
            "mapping_rate": 0.8,
            "n_standard_events": 20,
            "n_standard_epochs": 15,
            "processing_timestamp": "2024-01-01T00:00:00",
        }
        df = pipeline._build_mapping_qc_table("P01", "s1", features)
        assert len(df) == 1
        assert df.iloc[0]["n_rare_mapped"] == 8
        assert "rare_mapping_rate" in df.columns

    def test_update_master_feature_tables(self, pipeline, temp_output_dir):
        clinical1 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "n_rare_epochs": 3,
                    "p300_amplitude_uV": 4.0,
                }
            ]
        )
        detail1 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "electrode": "Pz",
                    "p300_amplitude_uV": 4.0,
                }
            ]
        )
        qc1 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "n_rare_mapped": 3,
                }
            ]
        )

        pipeline._update_master_feature_tables(clinical1, detail1, qc1)

        clinical2 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "n_rare_epochs": 5,
                    "p300_amplitude_uV": 6.0,
                }
            ]
        )
        detail2 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "electrode": "Pz",
                    "p300_amplitude_uV": 6.0,
                }
            ]
        )
        qc2 = pd.DataFrame(
            [
                {
                    "patient_id": "P01",
                    "session_id": "s1",
                    "session_date": "2024-01-01",
                    "n_rare_mapped": 5,
                }
            ]
        )
        pipeline._update_master_feature_tables(clinical2, detail2, qc2)

        clinical_path = pipeline._output_paths.features / "p300_oddball_clinical.parquet"
        clinical = pd.read_parquet(clinical_path)
        assert len(clinical) == 1
        assert clinical.iloc[0]["n_rare_epochs"] == 5
        assert clinical.iloc[0]["p300_amplitude_uV"] == 6.0

    def test_save_outputs(self, pipeline, temp_output_dir, mock_evoked):
        epochs = mne.EpochsArray(
            np.zeros((2, 3, 461)),
            mne.create_info(["Fz", "Cz", "Pz"], 512, "eeg"),
            tmin=-0.2,
            verbose=False,
        )
        features = {
            "patient_id": "P01",
            "session_id": "s1",
            "date": "2024-01-01",
            "n_epochs": 2,
            "p300_amplitude_uV": 4.0,
            "p300_latency_ms": 380,
            "p300_n_valid_electrodes": 3,
            "p300_subtype": "P3b",
            "qc_notes": "ok",
            "n_mapped": 2,
            "n_rare_events": 2,
            "n_unmapped": 0,
            "n_boundary_clipped": 0,
            "mapping_rate": 1.0,
        }
        pipeline._save_outputs(
            patient_id="P01",
            session_id="s1",
            epochs=epochs,
            erp=mock_evoked,
            features=features,
        )

        erp_file = pipeline._output_paths.erps / "P01_s1_oddball-ave.fif"
        assert erp_file.exists()
        clinical_file = pipeline._output_paths.features / "p300_oddball_clinical.parquet"
        assert clinical_file.exists()
        detail_file = pipeline._output_paths.features / "p300_oddball_electrode_detail.parquet"
        assert detail_file.exists()
        qc_file = pipeline._output_paths.features / "p300_oddball_mapping_qc.parquet"
        assert qc_file.exists()


# ── generate_summary and helpers ───────────────────────────────────────────────


class TestGenerateSummaryAndHelpers:
    def test_generate_summary_no_data(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.results = None
        summary = pipeline.generate_summary()
        assert summary["status"] == "NO_DATA"
        assert summary["n_sessions"] == 0

    def test_generate_summary_empty_df(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.results = pd.DataFrame()
        summary = pipeline.generate_summary()
        assert summary["status"] == "NO_DATA"
        assert summary["n_sessions"] == 0

    def test_generate_summary_p300_plus(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.results = pd.DataFrame(
            [
                {"session_id": "s1", "p300_amplitude_uV": 3.0, "p300_latency_ms": 380},
            ]
        )
        summary = pipeline.generate_summary()
        assert summary["status"] == "P300+"
        assert summary["n_sessions"] == 1

    def test_generate_summary_p300_minus(self, pipeline):
        pipeline.patient_id = "P01"
        pipeline.results = pd.DataFrame(
            [
                {"session_id": "s1", "p300_amplitude_uV": 1.0, "p300_latency_ms": 380},
            ]
        )
        summary = pipeline.generate_summary()
        assert summary["status"] == "P300-"
        assert summary["n_sessions"] == 1

    def test_sanitize_session_id(self):
        assert P300OddballPipeline._sanitize_session_id("s/a\\b:c") == "s_a_b_c"
        assert P300OddballPipeline._sanitize_session_id("normal_id") == "normal_id"

    def test_get_output_paths_custom_dir(self, mock_loader, tmp_path):
        pl = P300OddballPipeline(loader=mock_loader, output_dir=tmp_path, verbose=False)
        p = pl._output_paths
        assert p.erps == tmp_path / "erps"
        assert p.features == tmp_path / "features"
        assert p.plots_erp == tmp_path / "plots" / "erp"
        assert p.qc == tmp_path / "qc"

    def test_get_output_paths_default_dir(self, mock_loader):
        pl = P300OddballPipeline(loader=mock_loader, output_dir=config.PROCESSED_DATA_DIR, verbose=False)
        p = pl._output_paths
        assert p.erps == config.ERPS_DIR
        assert p.features == config.FEATURES_DIR
        assert p.plots_erp == config.ERP_PLOTS_DIR
        assert p.qc == config.QC_REPORTS_DIR


# ── Plotting (smoke) ──────────────────────────────────────────────────────────


class TestPlotting:
    def test_plot_erp_figure(self, pipeline, temp_output_dir, mock_evoked):
        pipeline.patient_id = "P01"
        features = {
            "p300_amplitude_uV": 4.0,
            "p300_latency_ms": 380,
            "p300_best_electrode": "Pz",
        }
        label = "P01 | s1 (2024-01-01)"
        fig = pipeline.viz.plot_erp_figure(
            rare_erp=mock_evoked,
            rare_sem=mock_evoked,
            standard_erp=None,
            standard_sem=None,
            diff_erp=None,
            features=features,
            label=label,
        )
        plot_path = pipeline._output_paths.plots_erp / "P01_s1_oddball_erp.png"
        pipeline._save_fig(fig, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_plot_p300_focus(self, pipeline, mock_evoked):
        features = {
            "p300_amplitude_Pz_uV": 2.49,
            "p300_latency_Pz_ms": 471.0,
        }
        label = "P01 | s1 (2024-01-01)"
        fig = pipeline.viz.plot_p300_focus(
            rare_erp=mock_evoked,
            standard_erp=mock_evoked,
            diff_erp=mock_evoked,
            features=features,
            label=label,
        )
        axis_texts = [text.get_text() for text in fig.axes[0].texts]
        plot_text = " ".join(axis_texts)
        annotation = next(text for text in fig.axes[0].texts if "P300 Candidate" in text.get_text())
        plot_path = pipeline._output_paths.plots_erp / "P01_s1_oddball_p300.png"
        pipeline._save_fig(fig, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
        assert "P300 Candidate" in plot_text
        assert "2.49uV" in plot_text
        assert "300-600 ms" in plot_text
        assert annotation.get_ha() == "right"

    def test_plot_mmn_focus(self, pipeline, mock_evoked):
        features = {
            "diff_mmn_amplitude_Fz_uV": -2.5,
            "diff_mmn_latency_Fz_ms": 150.0,
        }
        label = "P01 | s1 (2024-01-01)"
        fig = pipeline.viz.plot_mmn_focus(
            rare_erp=mock_evoked,
            standard_erp=mock_evoked,
            diff_erp=mock_evoked,
            features=features,
            label=label,
        )
        plot_text = " ".join(text.get_text() for text in fig.axes[0].texts)
        plot_path = pipeline._output_paths.plots_erp / "P01_s1_oddball_mmn.png"
        pipeline._save_fig(fig, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0
        assert "100-250 ms" in plot_text

    def test_plot_erp_image(self, pipeline, temp_output_dir, mock_evoked):
        # Need >= 3 epochs or plot_erp_image returns None
        epochs = mne.EpochsArray(
            np.zeros((3, 3, 461)),
            mne.create_info(["Fz", "Cz", "Pz"], 512, "eeg"),
            tmin=-0.2,
            verbose=False,
        )
        label = "P01 | s1 (2024-01-01)"
        fig = pipeline.viz.plot_erp_image(epochs, label)
        assert fig is not None
        plot_path = pipeline._output_paths.plots_erp / "P01_s1_oddball_erp_image.png"
        pipeline._save_fig(fig, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


# ── STANDARD_EVENT_LABELS ─────────────────────────────────────────────────────


def test_standard_event_labels():
    assert "standard" in STANDARD_EVENT_LABELS
    assert "frequent" in STANDARD_EVENT_LABELS
