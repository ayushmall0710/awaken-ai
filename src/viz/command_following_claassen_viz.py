"""Visualizations for the Claassen SVM Command Following pipeline.

Produces:
  - ROC curves (one subplot per command side)
  - Permutation null distribution histograms (one subplot per command side)
  - SVM channel weight topomaps (one subplot per command side)
"""

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

UW_PURPLE = "#4b2e83"
ACCENT_GREEN = "#16a34a"
ACCENT_RED = "#dc2626"


class ClaassenVisualizer:
    """Generates all plots for the Claassen SVM pipeline report."""

    def __init__(self, bands: Dict[str, Tuple[float, float]]):
        self.bands = bands

    def plot_roc_curves(self, side_results: List[Dict]) -> plt.Figure:
        """ROC curve per command side, combined in one figure."""
        n_sides = len(side_results)
        fig, axes = plt.subplots(1, n_sides, figsize=(6 * n_sides, 5.5), squeeze=False)
        axes = axes.flatten()

        for i, sr in enumerate(side_results):
            ax = axes[i]
            y_true = sr["y_true"]
            y_scores = sr["y_scores"]
            auc = sr["auc"]
            side = sr["side"].capitalize()

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax.plot(fpr, tpr, color=UW_PURPLE, linewidth=2.5, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1, label="Chance (0.5)")

            ax.fill_between(fpr, 0, tpr, alpha=0.1, color=UW_PURPLE)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{side} Command")
            ax.legend(loc="lower right", fontsize=10)
            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.02])
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        fig.suptitle("ROC Curve — LOO-SVM Classification", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig

    def plot_permutation_distributions(self, side_results: List[Dict]) -> plt.Figure:
        """Permutation null distribution with observed AUC marker, one subplot per side."""
        n_sides = len(side_results)
        fig, axes = plt.subplots(1, n_sides, figsize=(6 * n_sides, 5), squeeze=False)
        axes = axes.flatten()

        for i, sr in enumerate(side_results):
            ax = axes[i]
            perm_aucs = sr["perm_aucs"]
            observed_auc = sr["auc"]
            p_value = sr["p_value_perm"]
            side = sr["side"].capitalize()

            ax.hist(perm_aucs, bins=30, color="#c4b5fd", edgecolor="#7c3aed", alpha=0.8, label="Null distribution")
            ax.axvline(
                observed_auc,
                color=ACCENT_RED,
                linewidth=2.5,
                linestyle="--",
                label=f"Observed AUC = {observed_auc:.3f}",
            )

            # Shade the area beyond observed AUC
            ax.axvspan(observed_auc, ax.get_xlim()[1], alpha=0.1, color=ACCENT_RED)

            ax.set_xlabel("AUC (permuted labels)")
            ax.set_ylabel("Count")
            ax.set_title(f"{side} Command (p = {p_value:.4f})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Permutation Test — Null Distribution", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig

    def plot_channel_weight_topomaps(
        self,
        side_results: List[Dict],
        epochs_info: mne.Info,
    ) -> plt.Figure:
        """Topomap of absolute SVM weights averaged across frequency bands, one per side.

        Features are ordered band-first: [band0_ch0, band0_ch1, ..., band1_ch0, ...].
        We reshape to (n_bands, n_channels), take the mean absolute weight across
        bands for each channel, and plot on the scalp with a single shared colorbar.
        """
        n_sides = len(side_results)
        n_bands = len(self.bands)

        channel_importances = self._compute_channel_importances(side_results, n_bands)
        vmin = 0.0
        # Ensure vmax > vmin to avoid ValueError in mne.viz.plot_topomap if all coefs are zero
        vmax = max(1e-6, max(ci.max() for ci in channel_importances))

        fig, axes = plt.subplots(
            1,
            n_sides,
            figsize=(5 * n_sides + 1.5, 5),
            squeeze=False,
        )
        axes = axes.flatten()

        im = None
        for i, sr in enumerate(side_results):
            ax = axes[i]
            side = sr["side"].capitalize()

            im, _ = mne.viz.plot_topomap(
                channel_importances[i],
                epochs_info,
                axes=ax,
                show=False,
                cmap="Reds",
                vlim=(vmin, vmax),
                contours=6,
                extrapolate="head",
                sphere=(0.0, 0.0, 0.0, 0.095),  # Standard head radius to ensure clipping
            )
            ax.set_title(f"{side} Command", fontsize=12, fontweight="bold", pad=10)

        # Use a dedicated axis to prevent the colorbar from overlapping the topoplots
        fig.subplots_adjust(right=0.86, wspace=0.2)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Mean |SVM Weight|", fontsize=11)

        fig.suptitle(
            "SVM Channel Importance",
            fontsize=14,
            fontweight="bold",
            y=1.0,
        )
        return fig

    @staticmethod
    def _compute_channel_importances(
        side_results: List[Dict],
        n_bands: int,
    ) -> List[np.ndarray]:
        importances = []
        for sr in side_results:
            coefs = sr["svm_coefs"]
            n_channels = sr["n_channels"]
            weight_matrix = np.abs(coefs).reshape(n_bands, n_channels)
            importances.append(weight_matrix.mean(axis=0))
        return importances
