"""
Visualization module for language tracking results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import mne

import src.reports.style_utils as style_utils
from src.pipelines.language_tracking import LanguageTrackingAnalysis

_TARGET_WORD_FREQ = LanguageTrackingAnalysis.TARGET_WORD_FREQ
_TARGET_PHRASE_FREQ = LanguageTrackingAnalysis.TARGET_PHRASE_FREQ
_TARGET_SENTENCE_FREQ = LanguageTrackingAnalysis.TARGET_SENTENCE_FREQ

_ITPC_TARGET_SPECS = [
    (_TARGET_WORD_FREQ, "Word", "#b2182b"),
    (_TARGET_PHRASE_FREQ, "Phrase", "#2166ac"),
    (_TARGET_SENTENCE_FREQ, "Sentence", "#4dac26"),
]


def _setup_figure_and_ax(figsize=(10, 5), title=None, xlabel=None, ylabel=None):
    """Generic setup for matplotlib figure and axis."""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    return fig, ax


def _save_and_close(fig, path, dpi=150):
    """Save figure and close it to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _add_itpc_annotations(ax, metrics, method_label, y_top, plot_freqs):
    """Add vertical lines and text annotations for target ITPC frequencies."""
    method_key = method_label.lower()
    for freq, label, color in _ITPC_TARGET_SPECS:
        if freq < float(plot_freqs[0]) or freq > float(plot_freqs[-1]):
            continue

        l_lbl = label.lower()
        p_val = metrics.get(f"p_{l_lbl}", metrics.get(f"{method_key}_p_{l_lbl}", 1.0))
        itpc_val = metrics.get(f"itpc_{l_lbl}", metrics.get(f"{method_key}_itpc_{l_lbl}", 0.0))

        itpc_str = style_utils.format_with_significance(itpc_val, p_val)
        ax.axvline(freq, color=color, linestyle="--", linewidth=1.5)
        ax.text(
            freq + 0.04,
            y_top * 0.95,
            f"{label}\n{itpc_str}",
            color=color,
            fontsize=8,
            verticalalignment="top",
        )


def plot_itpc_results(itc, patient_id: str, output_dir: str, metrics: dict):
    """
    Generate and save enhanced ITPC plots (Topomap and TFR).

    Args:
        itc: MNE AverageTFR object.
        patient_id: Patient ID string.
        output_dir: Path to save outputs.
        metrics: Metrics dictionary from _extract_itpc_metrics.
    """

    output_dir = Path(output_dir)
    target_freq = metrics["freq_sentence_hz"]
    phrase_freq = metrics["freq_phrase_hz"]
    word_freq = metrics["freq_word_hz"]

    vlim_max = max(float(np.percentile(itc.data, 95)) * 1.2, 0.05)

    fig_topo, ax_topo = _setup_figure_and_ax(
        figsize=(10, 8),
        title=f"ITPC Topomap @ {target_freq:.3f} Hz\n{patient_id}",
    )
    itc.plot_topomap(
        tmin=0,
        tmax=None,
        fmin=target_freq * 0.9,
        fmax=target_freq * 1.1,
        baseline=None,
        mode=None,
        axes=ax_topo,
        show=False,
        cmap="viridis",
        colorbar=True,
        vlim=(0, vlim_max),
    )
    _save_and_close(fig_topo, output_dir / f"{patient_id}_language_ITPC_topomap.png", dpi=300)

    fig_tfr, ax_tfr = _setup_figure_and_ax(
        figsize=(14, 8),
        title=f"ITPC Time-Frequency ({patient_id}) - Hemisphere Mean",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )
    itc.plot(
        baseline=None,
        mode=None,
        axes=ax_tfr,
        show=False,
        combine="mean",
        cmap="viridis",
        vlim=(0, vlim_max),
        colorbar=True,
    )
    ax_tfr.axhline(y=target_freq, color="white", linestyle="--", linewidth=2, label=f"Sentence ({target_freq:.3f} Hz)")
    ax_tfr.text(itc.times[0], target_freq, " Sentence", color="white", verticalalignment="bottom", fontweight="bold")
    ax_tfr.axhline(y=phrase_freq, color="white", linestyle="-.", linewidth=2, label=f"Phrase ({phrase_freq:.3f} Hz)")
    ax_tfr.text(itc.times[0], phrase_freq, " Phrase", color="white", verticalalignment="bottom", fontweight="bold")
    ax_tfr.axhline(y=word_freq, color="white", linestyle=":", linewidth=2, label=f"Word ({word_freq:.3f} Hz)")
    ax_tfr.text(itc.times[0], word_freq, " Word", color="white", verticalalignment="bottom", fontweight="bold")

    tfr_path = _save_and_close(fig_tfr, output_dir / f"{patient_id}_language_ITPC_tfr.png", dpi=300)
    return tfr_path


def plot_itpc_channel_bar(
    itpc_data: np.ndarray,
    ch_names: list,
    patient_id: str,
    output_dir: str,
    n_trials: int,
    freqs: np.ndarray,
    sentence_band: tuple,
    phrase_band: tuple,
    word_band: tuple,
) -> Path:
    """Backward compatibility wrapper for vertical bar chart."""
    # We'll just point it to the new horizontal one if that's what's desired by spec,
    # but the spec says "Convert existing vertical... into horizontal".
    # So I'll rename the original to _legacy and make this one horizontal.
    return plot_itpc_channels_horizontal(
        itpc_data,
        ch_names,
        patient_id,
        output_dir,
        n_trials,
        freqs,
        sentence_band,
        phrase_band,
        word_band,
        method_label="Morlet",
    )


def plot_itpc_channels_horizontal(
    itpc_data: np.ndarray,
    ch_names: list,
    patient_id: str,
    output_dir: str,
    n_trials: int,
    freqs: np.ndarray,
    sentence_band: tuple,
    phrase_band: tuple,
    word_band: tuple,
    method_label: str = "ITPC",
) -> Path:
    """
    Horizontal bar chart of per-channel ITPC with chance-level reference.

    Supports both Morlet (3D: ch, freq, time) and DFT (2D: ch, freq) data.

    Parameters
    ----------
    itpc_data : np.ndarray
        ITPC array. Shape (n_channels, n_freqs, n_times) or (n_channels, n_freqs).
    ch_names : list
        Channel names.
    patient_id : str
        Patient identifier.
    output_dir : str
        Directory to save the plot.
    n_trials : int
        Number of trials for chance level.
    freqs : np.ndarray
        Frequency axis.
    sentence_band, phrase_band, word_band : tuple
        (low, high) Hz bounds.
    method_label : str
        Label for filename and title (e.g. "Morlet", "DFT").
    """
    sent_mask = (freqs >= sentence_band[0]) & (freqs <= sentence_band[1])
    phrase_mask = (freqs >= phrase_band[0]) & (freqs <= phrase_band[1])
    word_mask = (freqs >= word_band[0]) & (freqs <= word_band[1])

    sent_per_ch = (
        np.mean(itpc_data[:, sent_mask], axis=1)
        if itpc_data.ndim == 2
        else np.mean(itpc_data[:, sent_mask, :], axis=(1, 2))
    )
    phrase_per_ch = (
        np.mean(itpc_data[:, phrase_mask], axis=1)
        if itpc_data.ndim == 2
        else np.mean(itpc_data[:, phrase_mask, :], axis=(1, 2))
    )
    word_per_ch = (
        np.mean(itpc_data[:, word_mask], axis=1)
        if itpc_data.ndim == 2
        else np.mean(itpc_data[:, word_mask, :], axis=(1, 2))
    )

    chance = 1.0 / np.sqrt(n_trials)
    y = np.arange(len(ch_names))[::-1]  # Reverse order
    height = 0.25

    fig, ax = _setup_figure_and_ax(
        figsize=(10, 8),
        title=f"{patient_id}: Per-Channel {method_label} ITPC",
        xlabel="ITPC",
    )
    ax.barh(y + height, sent_per_ch, height, label="Sentence band", color="#2166ac")
    ax.barh(y, phrase_per_ch, height, label="Phrase band", color="#f4a582")
    ax.barh(y - height, word_per_ch, height, label="Word band", color="#b2182b")

    ax.axvline(chance, color="gray", linestyle="--", linewidth=1.5, label=f"Chance (1/sqrt({n_trials}) = {chance:.3f})")

    ax.set_yticks(y)
    ax.set_yticklabels(ch_names)
    ax.legend(loc="lower right")

    # Add value labels
    for i, (s, p, w) in enumerate(zip(sent_per_ch, phrase_per_ch, word_per_ch)):
        yi = y[i]
        ax.text(s + 0.002, yi + height, f"{s:.3f}", va="center", fontsize=8)
        ax.text(p + 0.002, yi, f"{p:.3f}", va="center", fontsize=8)
        ax.text(w + 0.002, yi - height, f"{w:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, max(np.max([sent_per_ch, phrase_per_ch, word_per_ch]) * 1.15, chance * 1.5, 0.15))
    plt.tight_layout()

    return _save_and_close(fig, Path(output_dir) / f"{patient_id}_lang_{method_label.lower()}_channels_horizontal.png")


def plot_itpc_spectrum(
    itpc_spectrum: np.ndarray,
    freqs: np.ndarray,
    patient_id: str,
    output_dir: str,
    metrics: dict,
    method_label: str = "DFT",
    focus_label: Optional[str] = None,
    title: Optional[str] = None,
) -> Path:
    """
    Line plot of channel-averaged ITPC vs frequency (0.5-4.0 Hz).

    Vertical dashed lines mark word (3.125 Hz), phrase (1.56 Hz), and sentence
    (0.78 Hz) rates. Each line is annotated with the ITPC value and p-value.
    The plot title and output filename reflect method_label and focus_label.

    Parameters
    ----------
    itpc_spectrum : np.ndarray
        Shape (n_channels, n_freqs). ITPC per channel and frequency.
    freqs : np.ndarray
        Frequency axis in Hz, matching axis 1 of itpc_spectrum.
    patient_id : str
        Patient identifier for title and filename.
    output_dir : str or Path
        Directory to save the PNG.
    metrics : dict
        Must contain: itpc_word, itpc_phrase, itpc_sentence.
        P-values are read from keys p_word, p_phrase, p_sentence (generic form).
        For backward compatibility, dft_p_word, dft_p_phrase, dft_p_sentence or
        morlet_p_word etc. are accepted as fallbacks when generic keys are absent.
    method_label : str, optional
        Label for the analysis method (e.g., "DFT", "Morlet"). Used in the
        plot title and output filename. Defaults to "DFT".
    focus_label : str, optional
        Label for the focus (e.g., "Clinical", "Optimal", "LH", "RH").
        Used in the plot title and output filename.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    mask = (freqs >= 0.5) & (freqs <= 4.0)
    plot_freqs = freqs[mask]
    mean_itpc = np.mean(itpc_spectrum, axis=0)[mask]

    if not title:
        title = f"{patient_id}: {method_label} ITPC Frequency Spectrum"
        if focus_label:
            title += f" ({focus_label})"

    fig, ax = _setup_figure_and_ax(figsize=(10, 5), title=title, xlabel="Frequency (Hz)", ylabel="ITPC")
    ax.plot(plot_freqs, mean_itpc, color="#1a1a1a", linewidth=1.5, label="Mean ITPC")

    y_top = float(np.max(mean_itpc)) if len(mean_itpc) > 0 else 0.1
    _add_itpc_annotations(ax, metrics, method_label, y_top, plot_freqs)

    ax.set_xlim(0.5, 4.0)
    ax.legend(fontsize=9)
    plt.tight_layout()

    fname = f"{patient_id}_lang_{method_label.lower()}_spectrum"
    if focus_label:
        fname += f"_{focus_label.lower()}"

    return _save_and_close(fig, Path(output_dir) / f"{fname}.png")


def plot_itpc_topomap(
    itpc_spectrum: np.ndarray,
    freqs: np.ndarray,
    info: "mne.Info",
    target_freq: float,
    label: str,
    patient_id: str,
    output_dir: str,
    vlim: tuple = None,
    method_label: str = "DFT",
    highlight_channels: Optional[list[str]] = None,
    title: Optional[str] = None,
    show_colorbar: bool = True,
) -> Path:
    """
    Topomap of ITPC values across electrodes at a single target frequency.

    Extracts the closest frequency bin to target_freq and renders a scalp
    topography using mne.viz.plot_topomap. Montage must already be set on the
    info object.

    Parameters
    ----------
    itpc_spectrum : np.ndarray
        Shape (n_channels, n_freqs).
    freqs : np.ndarray
        Frequency axis matching itpc_spectrum axis 1.
    info : mne.Info
        Channel info with positions (montage must be set).
    target_freq : float
        Target frequency in Hz (0.78, 1.56, or 3.125).
    label : str
        Human-readable label e.g. "Sentence", "Phrase", "Word".
    patient_id : str
        Patient identifier.
    output_dir : str or Path
        Directory to save the PNG.
    vlim : tuple, optional
        (vmin, vmax). If None, auto-scales from data 95th percentile.
    method_label : str, optional
        Label for the analysis method (e.g., "DFT", "Morlet"). Used in the
        plot title and output filename. Defaults to "DFT".
    highlight_channels : list of str, optional
        List of channel names to highlight with a marker.

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    import mne as _mne

    bin_idx = int(np.argmin(np.abs(freqs - target_freq)))
    per_channel_itpc = itpc_spectrum[:, bin_idx]

    if vlim is None:
        vmax = max(float(np.percentile(per_channel_itpc, 95)) * 1.2, 0.1)
        vlim = (0.0, vmax)

    # Setup mask for highlighted channels
    mask = None
    mask_params = None
    if highlight_channels:
        mask = np.array([ch in highlight_channels for ch in info.ch_names])
        mask_params = dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=6)

    actual_freq = float(freqs[bin_idx])
    if not title:
        title = f"{label} ({actual_freq:.3f} Hz)\n{patient_id}"

    fig, ax = _setup_figure_and_ax(figsize=(5, 5), title=title)
    im, _ = _mne.viz.plot_topomap(
        per_channel_itpc,
        info,
        axes=ax,
        show=False,
        cmap="RdYlBu_r",
        vlim=vlim,
        contours=4,
        mask=mask,
        mask_params=mask_params,
    )
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ITPC")
    plt.tight_layout()

    safe_label = label.lower()
    return _save_and_close(fig, Path(output_dir) / f"{patient_id}_lang_topomap_{method_label.lower()}_{safe_label}.png")


def plot_focus_comparison_bar(
    df: pd.DataFrame,
    patient_id: str,
    output_dir: str | Path,
) -> Path:
    """
    Compare Comprehension ITPC and significance across focuses.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame containing 'focus', 'itpc_comprehension',
        and 'dft_p_comprehension'.
    patient_id : str
        Patient identifier.
    output_dir : str or Path
        Directory to save the plot.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    plot_df = df[df["focus"].isin(["clinical", "lh", "rh", "optimal"])].copy()
    # Sort for consistent display
    order = {"clinical": 0, "lh": 1, "rh": 2, "optimal": 3}
    plot_df["sort_idx"] = plot_df["focus"].map(order)
    plot_df = plot_df.sort_values("sort_idx").dropna(subset=["itpc_comprehension"])

    if plot_df.empty:
        # Create an empty plot if no data available
        fig, ax = _setup_figure_and_ax(figsize=(6, 4), title=f"{patient_id}: Comprehension ITPC by Focus")
        ax.text(0.5, 0.5, "No focus data available", ha="center", va="center")
        return _save_and_close(fig, Path(output_dir) / f"{patient_id}_lang_focus_comparison.png")

    fig, ax = _setup_figure_and_ax(
        figsize=(8, 5),
        title=f"{patient_id}: Language Comprehension Tracking by Focus",
        ylabel="Comprehension ITPC (Sentence + Phrase) / 2",
    )

    # Colors: use different colors for focuses
    colors = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"]  # Gray, Blue, Red, Green
    focus_colors = {f: colors[i] for i, f in enumerate(["clinical", "lh", "rh", "optimal"])}

    bars = ax.bar(
        plot_df["focus"].str.upper(),
        plot_df["itpc_comprehension"],
        color=[focus_colors.get(f, "#7f7f7f") for f in plot_df["focus"]],
        alpha=0.7,
        edgecolor="black",
    )

    # Add p-value stars
    for i, bar in enumerate(bars):
        p_val = plot_df.iloc[i]["dft_p_comprehension"]
        if pd.isna(p_val):
            continue

        stars = ""
        if p_val < 0.001:
            stars = style_utils.ICON_SIG_3
        elif p_val < 0.01:
            stars = style_utils.ICON_SIG_2
        elif p_val < 0.05:
            stars = style_utils.ICON_SIG_1

        if stars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                stars,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

    ax.set_ylim(0, max(plot_df["itpc_comprehension"].max() * 1.2, 0.2))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return _save_and_close(fig, Path(output_dir) / f"{patient_id}_lang_focus_comparison.png")
