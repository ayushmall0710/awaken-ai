import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.signal_processing import compute_welch_psd

logger = logging.getLogger(__name__)

# Set plotting defaults
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

# Visualization Constants
KEEP_COLOR = "#22c55e"  # green
STOP_COLOR = "#ef4444"  # red
RESPONSE_KEEP_COLOR = "#bbf7d0"  # light green
RESPONSE_STOP_COLOR = "#fecaca"  # light red
INSTRUCTION_COLOR = "#fbbf24"  # yellow


class CommandFollowingVisualizer:
    """
    Handles all visualization generation for the Command Following (ENG-04) pipeline.
    Produces PSD overlays, ERD Time Courses, ERD Bar charts, and Topomaps.
    """

    def __init__(self, bands: Dict[str, Tuple[float, float]]):
        self.bands = bands

    def plot_erd_bar(self, results_df: pd.DataFrame, contralateral_map: Dict[str, str]) -> plt.Figure:
        """Bar plot showing average ERD (dB) by channel and frequency band, faceted by side."""
        sides = results_df["side"].unique()
        n_sides = len(sides)

        if n_sides == 0:
            return plt.figure()

        fig, axes = plt.subplots(1, n_sides, figsize=(6 * n_sides, 6), sharey=True, squeeze=False)
        axes = axes.flatten()

        band_names = list(self.bands.keys())
        n_bands = len(band_names)
        bar_width = 0.35
        x = np.arange(len(results_df["channel"].unique()))

        for i, side in enumerate(sides):
            side_df = results_df[results_df["side"] == side]
            channels = side_df["channel"].unique()
            x = np.arange(len(channels))
            ax = axes[i]

            for b_idx, band_name in enumerate(band_names):
                band_df = side_df[side_df["band"] == band_name].set_index("channel").reindex(channels)
                offset = (b_idx - (n_bands - 1) / 2) * bar_width

                heights = band_df["erd_dB"].fillna(0).values
                errors = band_df["erd_std"].fillna(0).values

                color = "#2563eb" if band_name.lower() == "alpha" else "#7c3aed"
                ax.bar(
                    x + offset,
                    heights,
                    bar_width,
                    yerr=errors,
                    capsize=4,
                    label=band_name,
                    color=color,
                    alpha=0.8,
                    error_kw={"elinewidth": 1.2, "ecolor": "#374151"},
                )

            contra_ch = contralateral_map.get(side.lower())
            ax.set_title(f"{side.capitalize()} Command\n(Expect {contra_ch} Desync)")
            ax.set_xticks(x)
            ax.set_xticklabels(channels)
            ax.set_xlabel("Channel")
            if i == 0:
                ax.set_ylabel("ERD (dB)\n(Positive = Desynchronization)")
            else:
                ax.set_ylabel("")

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.legend()

        plt.tight_layout()
        return fig

    def plot_psd_overlay(
        self,
        keep_epochs: mne.Epochs,
        stop_epochs: mne.Epochs,
        channel: str,
        title: str,
    ) -> plt.Figure:
        """Power spectrum overlay: Keep (green) vs Stop (red) for one channel.

        Shows absolute PSD (dB re 1 V²/Hz) across 8-30 Hz. Alpha and beta band
        boundaries are marked with vertical dashed lines and shaded regions.
        A difference trace (ERD = Stop − Keep, positive = desynchronization) is
        plotted in a separate panel below.

        Replaces TFR plots, which are unreliable for short, non-time-locked epochs.
        """
        if channel not in keep_epochs.ch_names or channel not in stop_epochs.ch_names:
            logger.warning("Channel %s not found in epochs for PSD overlay.", channel)
            return plt.figure()

        sfreq = keep_epochs.info["sfreq"]
        ch_idx_k = keep_epochs.ch_names.index(channel)
        ch_idx_s = stop_epochs.ch_names.index(channel)

        keep_data = keep_epochs.get_data()[:, ch_idx_k, :]  # (n_epochs, n_times)
        stop_data = stop_epochs.get_data()[:, ch_idx_s, :]

        # Full spectrum (8-30 Hz) averaged across epochs, converted to dB
        freqs, keep_psd = compute_welch_psd(keep_data, sfreq=sfreq, fmin=8.0, fmax=30.0)
        _, stop_psd = compute_welch_psd(stop_data, sfreq=sfreq, fmin=8.0, fmax=30.0)

        keep_mean = 10 * np.log10(np.maximum(keep_psd.mean(axis=0), 1e-30))
        stop_mean = 10 * np.log10(np.maximum(stop_psd.mean(axis=0), 1e-30))
        diff = stop_mean - keep_mean  # positive = ERD

        fig, (ax_psd, ax_diff) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # ── PSD overlay ──
        ax_psd.plot(freqs, keep_mean, color=KEEP_COLOR, linewidth=2, label="Keep (motor imagery)")
        ax_psd.plot(freqs, stop_mean, color=STOP_COLOR, linewidth=2, label="Stop (rest)")
        ax_psd.set_ylabel("Power (dB re 1 V²/Hz)")
        ax_psd.legend(fontsize=9)
        ax_psd.grid(True, alpha=0.3)

        # ── Difference trace ──
        ax_diff.plot(freqs, diff, color="#6366f1", linewidth=2)
        ax_diff.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax_diff.fill_between(freqs, 0, diff, where=(diff > 0), color=KEEP_COLOR, alpha=0.25, label="ERD (desync)")
        ax_diff.fill_between(freqs, 0, diff, where=(diff < 0), color=STOP_COLOR, alpha=0.25, label="ERS (sync)")
        ax_diff.set_ylabel("Diff (dB)\nStop − Keep")
        ax_diff.set_xlabel("Frequency (Hz)")
        ax_diff.legend(fontsize=9)
        ax_diff.grid(True, alpha=0.3)

        # Band boundary lines on both axes
        band_colors = {"Alpha": "#fbbf24", "Beta": "#a78bfa"}
        for ax in (ax_psd, ax_diff):
            for band_name, (fmin, fmax) in self.bands.items():
                bc = band_colors.get(band_name, "#9ca3af")
                ax.axvline(fmin, color=bc, linestyle=":", linewidth=1.2, alpha=0.8)
                ax.axvline(fmax, color=bc, linestyle=":", linewidth=1.2, alpha=0.8)
                ax.axvspan(fmin, fmax, color=bc, alpha=0.05)

        # Band labels on PSD panel
        ypos = ax_psd.get_ylim()[0] + 0.05 * (ax_psd.get_ylim()[1] - ax_psd.get_ylim()[0])
        for band_name, (fmin, fmax) in self.bands.items():
            ax_psd.text(
                (fmin + fmax) / 2,
                ypos,
                band_name,
                ha="center",
                fontsize=8,
                color=band_colors.get(band_name, "#6b7280"),
                alpha=0.9,
            )

        plt.tight_layout()
        return fig

    def plot_topomap(self, epochs: mne.Epochs, title: str, fmin: float, fmax: float) -> plt.Figure:
        """
        Plots a topographical map of band power.
        Useful for inspecting spatial distribution of activation.
        """
        fig, ax = plt.subplots(figsize=(5, 5))

        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        sfreq = epochs.info["sfreq"]
        _, psds = compute_welch_psd(data, sfreq=sfreq, fmin=fmin, fmax=fmax)
        psds_db = 10 * np.log10(np.maximum(psds.mean(axis=(0, 2)), 1e-10))

        im, _ = mne.viz.plot_topomap(
            psds_db, epochs.info, axes=ax, show=False, cmap="RdBu_r", contours=6, extrapolate="local"
        )
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Power (dB)", rotation=270, labelpad=15)

        return fig
