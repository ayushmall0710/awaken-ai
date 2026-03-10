"""Visualization helpers for the P300/Oddball ERP pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np


class OddballVisualizer:
    """Generate ERP, ERP-image, and topomap figures for the oddball paradigm.

    This class is intentionally stateless: all methods accept the data they need
    and return a ``matplotlib.figure.Figure`` instance. Saving to disk is the
    responsibility of the caller (the pipeline or report), which also chooses
    filenames and directories.
    """

    def plot_erp_figure(
        self,
        rare_erp: mne.Evoked,
        rare_sem: mne.Evoked,
        standard_erp: Optional[mne.Evoked],
        standard_sem: Optional[mne.Evoked],
        diff_erp: Optional[mne.Evoked],
        features: Dict[str, Any],
        label: str,
    ) -> plt.Figure:
        """4-panel ERP figure.

        Panel layout (when diff_erp and standard_erp are available):
          1) All channels (butterfly), rare ERP only
          2) Rare vs standard for Fz, Cz, Pz (with SEM)
          3) Rare-only midline traces (Fz, Cz, Pz) titled \"P300\"
          4) Difference wave (Rare − Standard) midline traces titled \"MNN\"

        When diff_erp or standard_erp is missing, falls back to a 2-panel
        legacy layout (butterfly + selected electrodes).
        """
        if diff_erp is None or standard_erp is None:
            return self._plot_individual_erp_legacy(rare_erp, label)

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.55)

        times = rare_erp.times * 1000
        data = rare_erp.data * 1e6
        ch_names_upper = [ch.upper() for ch in rare_erp.ch_names]
        electrodes = ["Fz", "Cz", "Pz"]
        colors = {"Fz": "red", "Cz": "green", "Pz": "blue"}

        # Panel 1: butterfly (rare only)
        ax1 = fig.add_subplot(gs[0])
        for ch_idx in range(data.shape[0]):
            ax1.plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)
        ax1.axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        ax1.axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.set_title(f"{label} - All Channels (Butterfly)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Panel 2: rare vs standard (Fz/Cz/Pz)
        ax2 = fig.add_subplot(gs[1])
        for electrode in electrodes:
            if electrode.upper() not in ch_names_upper:
                continue
            ch_idx = ch_names_upper.index(electrode.upper())
            rare_trace = rare_erp.data[ch_idx, :] * 1e6
            rare_sem_trace = rare_sem.data[ch_idx, :] * 1e6
            color = colors[electrode]

            ax2.plot(times, rare_trace, linewidth=2, color=color, label=f"{electrode} (rare)")
            ax2.fill_between(
                times,
                rare_trace - rare_sem_trace,
                rare_trace + rare_sem_trace,
                alpha=0.2,
                color=color,
            )

            if standard_erp is not None and standard_sem is not None:
                std_trace = standard_erp.data[ch_idx, :] * 1e6
                std_sem_trace = standard_sem.data[ch_idx, :] * 1e6
                ax2.plot(
                    times,
                    std_trace,
                    linewidth=1.5,
                    color=color,
                    linestyle="--",
                    label=f"{electrode} (std)",
                )
                ax2.fill_between(
                    times,
                    std_trace - std_sem_trace,
                    std_trace + std_sem_trace,
                    alpha=0.1,
                    color=color,
                )

        ax2.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax2.axvspan(300, 600, alpha=0.1, color="gray")
        ax2.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Amplitude (µV)")
        subtype = features.get("p300_subtype", "unknown")
        ax2.set_title(f"Rare vs Standard — {subtype}")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: rare-only midline traces (P300)
        ax3 = fig.add_subplot(gs[2])
        for electrode in electrodes:
            if electrode.upper() not in ch_names_upper:
                continue
            ch_idx = ch_names_upper.index(electrode.upper())
            rare_trace = rare_erp.data[ch_idx, :] * 1e6
            color = colors[electrode]
            ax3.plot(times, rare_trace, linewidth=2, color=color, label=electrode)

        ax3.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax3.axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        ax3.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Amplitude (µV)")
        ax3.set_title("P300: Rare-Stimulus Evoked Response at Midline")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        # Panel 4: difference wave (Rare - Standard) — MNN
        ax4 = fig.add_subplot(gs[3])
        diff_data = diff_erp.data * 1e6
        for electrode in electrodes:
            if electrode.upper() not in ch_names_upper:
                continue
            ch_idx = ch_names_upper.index(electrode.upper())
            diff_trace = diff_data[ch_idx, :]
            color = colors[electrode]
            ax4.plot(times, diff_trace, linewidth=2, color=color, label=electrode)

        ax4.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax4.axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        ax4.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax4.set_xlabel("Time (ms)")
        ax4.set_ylabel("Amplitude (µV)")
        ax4.set_title("MNN: Difference Wave (Rare − Standard)")
        ax4.legend(loc="upper right")
        ax4.grid(True, alpha=0.3)

        fig.tight_layout(pad=1.5, h_pad=2.0)
        return fig

    def plot_erp_image(self, epochs: mne.Epochs, label: str) -> Optional[plt.Figure]:
        """Single-trial ERP image at Pz.

        Returns:
            Figure if an image can be generated (>= 3 epochs), otherwise None.
        """
        if len(epochs) < 3:
            return None

        try:
            ret = mne.viz.plot_epochs_image(
                epochs,
                picks=["Pz"],
                show=False,
            )
            fig = ret[0] if isinstance(ret, (list, tuple)) else ret

            fig.set_size_inches(12, 10)
            fig.subplots_adjust(top=0.85, bottom=0.15, hspace=0.5)

            title_text = f"ERP Image: Single-Trial Responses to Rare (Target) Stimuli at Pz — {label}"
            fig.text(
                0.5,
                0.98,
                title_text,
                ha="center",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            )

            caption = "Top: Each row = one trial. Bottom: Average. Time 0 = stimulus. Color = voltage (µV)."
            fig.text(
                0.5,
                0.03,
                caption,
                ha="center",
                fontsize=9,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            )
            return fig
        except Exception:
            # Caller will decide whether to log; returning None signals failure.
            return None

    def plot_topomap(self, diff_erp: mne.Evoked, label: str) -> plt.Figure:
        """Topomap series for the difference ERP."""
        times_to_plot = np.arange(-0.2, 0.75, 0.1)

        fig = diff_erp.plot_topomap(
            times=times_to_plot,
            show=False,
            colorbar=True,
            size=5,
            show_names=True,
        )

        # mne.Evoked.plot_topomap sometimes returns a list of figures
        if isinstance(fig, (list, tuple)):
            fig_obj = fig[0]
        else:
            fig_obj = fig

        fig_obj.suptitle(f"Difference Topomaps — {label}")
        return fig_obj

    # ------------------------------------------------------------------
    # Legacy helper
    # ------------------------------------------------------------------

    def _plot_individual_erp_legacy(self, erp: mne.Evoked, label: str) -> plt.Figure:
        """2-panel legacy ERP figure (butterfly + selected electrodes)."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        times = erp.times * 1000
        data = erp.data * 1e6
        ch_names_upper = [ch.upper() for ch in erp.ch_names]

        # Top panel: butterfly
        for ch_idx in range(data.shape[0]):
            axes[0].plot(times, data[ch_idx, :], alpha=0.3, linewidth=0.5)
        axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1, label="Stimulus")
        axes[0].axvspan(300, 600, alpha=0.2, color="green", label="P300 Window")
        axes[0].set_xlabel("Time (ms)")
        axes[0].set_ylabel("Amplitude (µV)")
        axes[0].set_title(f"{label} - All Channels")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Bottom panel: selected electrodes
        electrodes_to_plot: List[str] = ["Fz", "Cz", "Pz"]
        colors = ["red", "green", "blue"]

        for idx, electrode in enumerate(electrodes_to_plot):
            if electrode.upper() not in ch_names_upper:
                continue
            elec_idx = ch_names_upper.index(electrode.upper())
            color = colors[idx % len(colors)]
            axes[1].plot(times, data[elec_idx, :], linewidth=2, color=color, label=electrode)

        axes[1].axvline(x=0, color="k", linestyle="--", linewidth=1)
        axes[1].axvspan(300, 600, alpha=0.1, color="gray", label="P300 Window")
        axes[1].axhline(y=0, color="gray", linestyle=":", linewidth=1)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Amplitude (µV)")
        axes[1].set_title(f"{label} - Midline Electrodes (Composite Scoring)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
