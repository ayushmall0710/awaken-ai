"""
Tests for Command Following Analysis pipeline.

Tests the CommandFollowingAnalysis class, including data loading, ERD calculation,
and visualization helpers, ensuring robust integration.
"""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from src.pipelines.command_following import CommandFollowingAnalysis, CommandPair, deduplicate_and_label


class TestDeduplicateAndLabel:
    """Test event deduplication and labeling logic."""

    def test_deduplicate_single_events(self):
        """Test simple sequence with no overlaps."""
        events = [
            {"event_start": 10.0, "event_end": 20.0},
            {"event_start": 30.0, "event_end": 40.0},
            {"event_start": 50.0, "event_end": 60.0},
        ]
        labeled = deduplicate_and_label(events, start_with="keep")

        assert len(labeled) == 3
        assert labeled[0]["type"] == "keep"
        assert labeled[1]["type"] == "stop"
        assert labeled[2]["type"] == "keep"
        assert labeled[0]["start"] == 10.0

    def test_deduplicate_overlapping(self):
        """Test merging of overlapping events (same command detected twice)."""
        events = [
            {"event_start": 10.0, "event_end": 20.0, "correlation_score": 0.8},
            {"event_start": 10.1, "event_end": 20.1, "correlation_score": 0.9},  # Overlap
            {"event_start": 50.0, "event_end": 60.0, "correlation_score": 0.7},
        ]

        labeled = deduplicate_and_label(events, start_with="keep")

        assert len(labeled) == 2  # Merged first two
        assert labeled[0]["start"] == 10.0
        assert labeled[0]["end"] == 20.1  # Extended end
        assert labeled[0]["corr"] == 0.9  # Took max correlation
        assert labeled[0]["type"] == "keep"
        assert labeled[1]["type"] == "stop"

    def test_deduplicate_invalid(self):
        """Test handling of invalid/incomplete events."""
        events = [
            {"event_start": 10.0},  # Missing end
            {"event_end": 20.0},  # Missing start
            {},
        ]
        labeled = deduplicate_and_label(events)
        assert len(labeled) == 0


@pytest.fixture
def mock_epochs():
    """Create mock MNE epochs for testing."""
    info = mne.create_info(ch_names=["C3", "C4", "Cz", "Fz", "Pz", "O1", "O2"], sfreq=100, ch_types="eeg")
    # create 10 epochs, 2 seconds long
    data = np.random.randn(10, 7, 200)
    events = np.array([[i * 100, 0, 1] for i in range(10)])
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, verbose=False)
    return epochs


@pytest.fixture
def analysis_instance(mock_loader):
    """Create an instance of CommandFollowingAnalysis with mocked loader."""
    return CommandFollowingAnalysis(loader=mock_loader)


class TestCommandFollowingAnalysis:
    def test_initialization(self, analysis_instance):
        """Test proper initialization of the class."""
        assert analysis_instance.bands is not None
        assert "Alpha" in analysis_instance.bands
        assert analysis_instance.roi_channels is not None
        assert analysis_instance.pairs == []
        assert analysis_instance.patient_id is None

    @patch("src.pipelines.command_following.CommandFollowingAnalysis._find_matching_epoch")
    def test_load_epochs_logic(self, mock_find, analysis_instance, mock_epochs, mock_loader):
        """Test the logic of pairing Keep/Stop epochs."""
        # Mock dataframe returned by load_unified_data
        mock_loader.load_unified_data.return_value = pd.DataFrame()  # Return empty means no epochs

        analysis_instance.patient_id = "TEST_PAT"
        # load_epochs expects self.aligned_events to be set (usually by run())
        # We set it to a dummy DF with one command trial to trigger load_clean_epochs call
        analysis_instance.aligned_events = pd.DataFrame(
            [
                {
                    "trial_type": "left_command",
                    "date": "2024-01-01",
                    "session_id": "s_TEST_20240101",
                    "trial_id": "lct1",
                    "start_time": 1000,
                }
            ]
        )

        # Mock load_clean_epochs to return empty (simulating dropped epochs)
        mock_loader.load_clean_epochs.return_value = mne.EpochsArray(
            np.zeros((1, 1, 1)), mne.create_info(1, 100, "eeg"), tmin=0, verbose=False
        )[:0]

        analysis_instance.load()

        # Verify it TRIED to load epochs for the trial type we put in aligned_events
        mock_loader.load_clean_epochs.assert_called_once()

    def test_calculate_erd_positive_for_desynchronization(self, analysis_instance):
        """ERD = stop_power − keep_power; positive when stop > keep (desynchronization).

        Stop data is constructed with twice the amplitude of keep (4× the power),
        so erd_dB = 10*log10(stop/keep) > 0.
        """
        info = mne.create_info(["C3", "C4"], 100, "eeg")
        times = np.linspace(0, 3, 300)

        pairs = []
        for i in range(5):
            keep_data = (np.sin(2 * np.pi * 10 * times) * 1 + np.random.randn(2, 300) * 0.05)[np.newaxis]
            stop_data = (np.sin(2 * np.pi * 10 * times) * 2 + np.random.randn(2, 300) * 0.05)[np.newaxis]
            pairs.append(
                CommandPair(
                    keep=mne.EpochsArray(keep_data, info, tmin=0, verbose=False),
                    stop=mne.EpochsArray(stop_data, info, tmin=0, verbose=False),
                    side="left",
                    trial_id=f"cft{i}",
                    keep_start=0,
                    stop_start=10,
                )
            )

        analysis_instance.pairs = pairs
        analysis_instance._run_mixed_model = MagicMock(return_value=0.01)

        erd_df = analysis_instance.calculate_erd()

        assert not erd_df.empty
        assert "erd_dB" in erd_df.columns
        assert "p_value" in erd_df.columns
        assert "cohens_d" in erd_df.columns

        # stop amplitude 2 > keep amplitude 1 → erd_dB = stop_power − keep_power > 0
        left_c3 = erd_df[(erd_df.side == "left") & (erd_df.channel == "C3")]
        assert left_c3["erd_dB"].mean() > 0

    def test_generate_summary_cmd_positive(self, analysis_instance):
        """CMD+ when a contralateral channel is significant with positive ERD and large effect."""
        df = pd.DataFrame(
            {
                "side": ["left", "right"],
                "channel": ["C3", "C4"],
                "band": ["Alpha", "Alpha"],
                "erd_dB": [1.8, -0.5],  # C3 has strong positive ERD
                "p_value": [0.01, 0.5],
                "significant": [True, False],
                "is_contralateral": [True, False],
                "cohens_d": [0.8, -0.1],  # positive = stop > keep
                "p_value_raw": [0.01, 0.5],
            }
        )
        analysis_instance.erd_results = df
        analysis_instance.patient_id = "TEST_PAT"
        analysis_instance.pairs = [MagicMock(side="left")] * 5 + [MagicMock(side="right")] * 5

        summary = analysis_instance.generate_summary()

        assert summary["cmd_status"] == "CMD+"
        assert summary["n_significant_contra"] == 1
        assert "classification_chance_level" in summary

    def test_generate_summary_cmd_negative(self, analysis_instance):
        """CMD- when no contralateral channel meets all criteria."""
        df = pd.DataFrame(
            {
                "side": ["left", "right"],
                "channel": ["C3", "C4"],
                "band": ["Alpha", "Alpha"],
                "erd_dB": [0.3, -0.5],  # below erd_threshold_dB=1.0
                "p_value": [0.2, 0.5],
                "significant": [False, False],
                "is_contralateral": [True, True],
                "cohens_d": [0.2, -0.1],
                "p_value_raw": [0.2, 0.5],
            }
        )
        analysis_instance.erd_results = df
        analysis_instance.patient_id = "TEST_PAT"
        analysis_instance.pairs = [MagicMock()] * 6

        summary = analysis_instance.generate_summary()

        assert summary["cmd_status"] == "CMD-"
        assert summary["n_significant_contra"] == 0


class TestCommandFollowingVisualizer:
    @pytest.fixture
    def visualizer(self):
        from src.viz.command_following_viz import CommandFollowingVisualizer

        return CommandFollowingVisualizer({"Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)})

    def test_plot_erd_bar_returns_figure(self, visualizer):
        """Bar chart renders without error for standard two-sided results."""
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            {
                "side": ["left", "right"],
                "channel": ["C4", "C3"],
                "erd_dB": [1.5, -0.5],
                "erd_std": [0.3, 0.2],
                "band": ["Alpha", "Beta"],
            }
        )
        cmap = {"left": "C4", "right": "C3"}
        fig = visualizer.plot_erd_bar(df, cmap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psd_overlay_returns_figure(self, visualizer, mock_epochs):
        """PSD overlay returns a two-panel figure for a valid channel."""
        import matplotlib.pyplot as plt

        fig = visualizer.plot_psd_overlay(mock_epochs, mock_epochs, channel="C3", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psd_overlay_missing_channel(self, visualizer, mock_epochs):
        """Missing channel logs a warning and returns an empty figure — no crash."""
        import matplotlib.pyplot as plt

        fig = visualizer.plot_psd_overlay(mock_epochs, mock_epochs, channel="NONEXISTENT", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @patch("src.viz.command_following_viz.plt.colorbar")
    @patch("mne.viz.plot_topomap")
    def test_plot_topomap_returns_figure(self, mock_plot_topomap, mock_colorbar, visualizer, mock_epochs):
        """Topomap renders with compute_welch_psd backend; mne.viz.plot_topomap is called."""
        import matplotlib.pyplot as plt

        mock_plot_topomap.return_value = (MagicMock(), MagicMock())

        fig = visualizer.plot_topomap(mock_epochs, "Test Topo", fmin=8.0, fmax=13.0)

        assert isinstance(fig, plt.Figure)
        mock_plot_topomap.assert_called_once()
        plt.close(fig)
