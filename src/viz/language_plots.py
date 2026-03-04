"""
Visualization module for language tracking results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        fmin=target_freq - 0.01,
        fmax=target_freq + 0.01,
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
    import matplotlib.pyplot as plt

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
