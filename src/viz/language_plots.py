"""
Visualization module for language tracking results.
"""

from pathlib import Path

import matplotlib.pyplot as plt


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
    word_freq = metrics["freq_word_hz"]

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
        vlim=(0, 0.3),
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
        vlim=(0, 0.3),
        colorbar=True,
    )
    ax_tfr.axhline(y=target_freq, color="white", linestyle="--", linewidth=2, label=f"Sentence ({target_freq:.3f} Hz)")
    ax_tfr.text(itc.times[0], target_freq, " Sentence", color="white", verticalalignment="bottom", fontweight="bold")
    ax_tfr.axhline(y=word_freq, color="white", linestyle=":", linewidth=2, label=f"Word ({word_freq:.3f} Hz)")
    ax_tfr.text(itc.times[0], word_freq, " Word", color="white", verticalalignment="bottom", fontweight="bold")
    ax_tfr.set_title(f"ITPC Time-Frequency ({patient_id}) - Hemisphere Mean", fontsize=16)
    ax_tfr.set_xlabel("Time (s)", fontsize=12)
    ax_tfr.set_ylabel("Frequency (Hz)", fontsize=12)

    tfr_path = output_dir / f"{patient_id}_language_ITPC_tfr.png"
    fig_tfr.savefig(tfr_path, dpi=300, bbox_inches="tight")
    plt.close(fig_tfr)
