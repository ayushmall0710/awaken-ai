"""
Visualization module for language tracking results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import mne

# Stimulus linguistic rate constants (Hz) — must match LanguageTrackingAnalysis.TARGET_*_FREQ
_TARGET_WORD_FREQ = 3.125
_TARGET_PHRASE_FREQ = 1.56
_TARGET_SENTENCE_FREQ = 0.78

_ITPC_TARGET_SPECS = [
    (_TARGET_WORD_FREQ, "Word", "#b2182b"),
    (_TARGET_PHRASE_FREQ, "Phrase", "#2166ac"),
    (_TARGET_SENTENCE_FREQ, "Sentence", "#4dac26"),
]


def plot_itpc_results(itc, patient_id: str, output_dir: str, metrics: dict):
    """
    Generate and save enhanced ITPC plots (Topomap and TFR).

    Args:
        itc: MNE AverageTFR object.
        patient_id: Patient ID string.
        output_dir: Path to save outputs.
        metrics: Metrics dictionary from extract_itpc_metrics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_freq = metrics["freq_sentence_hz"]
    phrase_freq = metrics["freq_phrase_hz"]
    word_freq = metrics["freq_word_hz"]

    vlim_max = max(float(np.percentile(itc.data, 95)) * 1.2, 0.05)

    fig_topo, ax_topo = plt.subplots(1, 1, figsize=(10, 8))
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
    ax_topo.set_title(f"ITPC Topomap @ {target_freq:.3f} Hz\n{patient_id}", fontsize=14, fontweight="bold")
    topo_path = output_dir / f"{patient_id}_language_ITPC_topomap.png"
    fig_topo.savefig(topo_path, dpi=300, bbox_inches="tight")
    plt.close(fig_topo)

    fig_tfr, ax_tfr = plt.subplots(1, 1, figsize=(14, 8))
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
    ax_tfr.set_title(f"ITPC Time-Frequency ({patient_id}) - Hemisphere Mean", fontsize=16)
    ax_tfr.set_xlabel("Time (s)", fontsize=12)
    ax_tfr.set_ylabel("Frequency (Hz)", fontsize=12)

    tfr_path = output_dir / f"{patient_id}_language_ITPC_tfr.png"
    fig_tfr.savefig(tfr_path, dpi=300, bbox_inches="tight")
    plt.close(fig_tfr)


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
) -> None:
    """
    Bar chart of per-channel ITPC at sentence and word bands with chance-level reference.

    Parameters
    ----------
    itpc_data : np.ndarray
        Morlet ITPC array, shape (n_channels, n_freqs, n_times).
    ch_names : list
        Channel names corresponding to itpc_data axis 0.
    patient_id : str
        Patient identifier.
    output_dir : str
        Directory to save the plot.
    n_trials : int
        Number of trials (for chance-level computation).
    freqs : np.ndarray
        Frequency axis matching itpc_data axis 1.
    sentence_band : tuple
        (low, high) Hz bounds for the sentence band.
    phrase_band : tuple
        (low, high) Hz bounds for the phrase band.
    word_band : tuple
        (low, high) Hz bounds for the word band.
    """
    sent_mask = (freqs >= sentence_band[0]) & (freqs <= sentence_band[1])
    phrase_mask = (freqs >= phrase_band[0]) & (freqs <= phrase_band[1])
    word_mask = (freqs >= word_band[0]) & (freqs <= word_band[1])

    sent_per_ch = np.mean(itpc_data[:, sent_mask, :], axis=(1, 2))
    phrase_per_ch = np.mean(itpc_data[:, phrase_mask, :], axis=(1, 2))
    word_per_ch = np.mean(itpc_data[:, word_mask, :], axis=(1, 2))
    chance = 1.0 / np.sqrt(n_trials)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(ch_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25
    ax.bar(x - width, sent_per_ch, width, label="Sentence band", color="#2166ac")
    ax.bar(x, phrase_per_ch, width, label="Phrase band", color="#f4a582")
    ax.bar(x + width, word_per_ch, width, label="Word band", color="#b2182b")
    ax.axhline(
        chance,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Chance (1/sqrt({n_trials}) = {chance:.3f})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45)
    ax.set_ylabel("ITPC")
    ax.set_title(f"{patient_id}: Per-Channel ITPC")
    ax.legend()

    for i, (s, p, w) in enumerate(zip(sent_per_ch, phrase_per_ch, word_per_ch)):
        ax.text(i - width, s + 0.003, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, p + 0.003, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, w + 0.003, f"{w:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"{patient_id}_language_ITPC_channels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itpc_spectrum(
    itpc_spectrum: np.ndarray,
    freqs: np.ndarray,
    patient_id: str,
    output_dir: str,
    metrics: dict,
    method_label: str = "DFT",
) -> Path:
    """
    Line plot of channel-averaged ITPC vs frequency (0.5-4.0 Hz).

    Vertical dashed lines mark word (3.125 Hz), phrase (1.56 Hz), and sentence
    (0.78 Hz) rates. Each line is annotated with the ITPC value and p-value.
    The plot title and output filename reflect method_label.

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
        For backward compatibility, dft_p_word, dft_p_phrase, dft_p_sentence are
        accepted as fallbacks when the generic keys are absent.
    method_label : str, optional
        Label for the analysis method (e.g., "DFT", "Morlet"). Used in the
        plot title and output filename. Defaults to "DFT".

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = (freqs >= 0.5) & (freqs <= 4.0)
    plot_freqs = freqs[mask]
    mean_itpc = np.mean(itpc_spectrum, axis=0)[mask]

    target_specs = [
        (
            freq,
            lbl,
            metrics.get("itpc_" + lbl.lower(), 0),
            metrics.get("p_" + lbl.lower(), metrics.get("dft_p_" + lbl.lower(), 1)),
            color,
        )
        for freq, lbl, color in _ITPC_TARGET_SPECS
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_freqs, mean_itpc, color="#1a1a1a", linewidth=1.5, label="Mean ITPC")

    y_top = float(np.max(mean_itpc)) if len(mean_itpc) > 0 else 0.1
    for freq, label, itpc_val, p_val, color in target_specs:
        if freq < float(plot_freqs[0]) or freq > float(plot_freqs[-1]):
            continue
        p_str = "<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        ax.axvline(freq, color=color, linestyle="--", linewidth=1.5)
        ax.text(
            freq + 0.04,
            y_top * 0.95,
            f"{label}\n{itpc_val:.4f}\n{p_str}",
            color=color,
            fontsize=8,
            verticalalignment="top",
        )

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("ITPC", fontsize=11)
    ax.set_title(f"{patient_id}: {method_label} ITPC Frequency Spectrum", fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 4.0)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = out_dir / f"{patient_id}_lang_{method_label.lower()}_spectrum.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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

    Returns
    -------
    Path
        Absolute path to the saved PNG.
    """
    import mne as _mne

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_idx = int(np.argmin(np.abs(freqs - target_freq)))
    per_channel_itpc = itpc_spectrum[:, bin_idx]

    if vlim is None:
        vmax = max(float(np.percentile(per_channel_itpc, 95)) * 1.2, 0.1)
        vlim = (0.0, vmax)

    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = _mne.viz.plot_topomap(
        per_channel_itpc,
        info,
        axes=ax,
        show=False,
        cmap="RdYlBu_r",
        vlim=vlim,
        contours=4,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ITPC")
    actual_freq = float(freqs[bin_idx])
    ax.set_title(
        f"{label} ({actual_freq:.3f} Hz)\n{patient_id}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()

    safe_label = label.lower()
    out_path = out_dir / f"{patient_id}_lang_topomap_{method_label.lower()}_{safe_label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
