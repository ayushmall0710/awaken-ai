"""
Comparison of Morlet wavelet vs DFT-based ITPC for ENG-05.

Runs both ITPC methods on the same preprocessed epochs and produces
a side-by-side comparison to validate consistency across approaches.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.data_loading.config as config
from src.data_processing.language_optimization import LanguageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def plot_comparison(patient_id, morlet_metrics, dft_metrics, dft_freqs, dft_spectrum, output_dir):
    """Save a comparison figure: DFT spectrum with sentence/word markers vs Morlet values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"ITPC Method Comparison - {patient_id}", fontsize=13)

    # Left: DFT ITPC spectrum (mean across channels)
    ax = axes[0]
    mean_spectrum = dft_spectrum.mean(axis=0)
    ax.plot(dft_freqs, mean_spectrum, lw=1.5, color="steelblue")
    ax.axvline(
        dft_metrics["freq_sentence_hz"],
        color="orange",
        linestyle="--",
        label=f"Sentence ({dft_metrics['freq_sentence_hz']:.3f} Hz)",
    )
    ax.axvline(
        dft_metrics["freq_word_hz"],
        color="red",
        linestyle="--",
        label=f"Word ({dft_metrics['freq_word_hz']:.3f} Hz)",
    )
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ITPC (DFT)")
    ax.set_title("DFT ITPC Spectrum (mean channels)")
    ax.legend(fontsize=8)

    # Right: Bar comparison of Sentence vs Word ITPC for both methods
    ax = axes[1]
    x = np.array([0, 1])
    width = 0.35
    morlet_vals = [morlet_metrics["itpc_sentence"], morlet_metrics["itpc_word"]]
    dft_vals = [dft_metrics["itpc_sentence"], dft_metrics["itpc_word"]]
    bars1 = ax.bar(x - width / 2, morlet_vals, width, label="Morlet", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, dft_vals, width, label="DFT", color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sentence Rate (~0.065 Hz)", "Word Rate (~0.77 Hz)"])
    ax.set_ylabel("ITPC")
    ax.set_title("Morlet vs DFT: Key Frequency ITPC")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    plt.tight_layout()
    out_path = output_dir / f"{patient_id}_itpc_method_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot: {out_path}")


def compare_patient(processor, patient_id, focus="LH"):
    """Run both ITPC methods and return comparison metrics."""
    import mne

    epochs = processor.process_patient(patient_id, focus=focus)
    if epochs is None:
        logger.warning(f"Skipping {patient_id}: no data.")
        return None

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning(f"Montage error: {e}")

    # Morlet
    itpc_data, _itc_obj = processor.compute_itpc(epochs)
    morlet_metrics = processor.extract_itpc_metrics(itpc_data)

    # DFT
    dft_spectrum, dft_freqs = processor.compute_itpc_dft(epochs)
    dft_metrics = processor.extract_itpc_metrics_dft(dft_spectrum, dft_freqs)

    output_dir = config.LOCAL_DATA_ROOT / "outputs" / patient_id
    plot_comparison(patient_id, morlet_metrics, dft_metrics, dft_freqs, dft_spectrum, output_dir)

    print(f"\n--- {patient_id} ---")
    print(f"{'Metric':<30} {'Morlet':>10} {'DFT':>10}")
    print("-" * 52)
    print(
        f"{'Sentence ITPC (0.065 Hz)':<30} "
        f"{morlet_metrics['itpc_sentence']:>10.4f} "
        f"{dft_metrics['itpc_sentence']:>10.4f}"
    )
    print(
        f"{'Word ITPC (0.77 Hz)':<30} "
        f"{morlet_metrics['itpc_word']:>10.4f} "
        f"{dft_metrics['itpc_word']:>10.4f}"
    )
    print(
        f"{'Ratio (Sentence/Word)':<30} "
        f"{morlet_metrics['ratio_sent_word']:>10.2f} "
        f"{dft_metrics['ratio_sent_word']:>10.2f}"
    )
    sent_morlet_gt = morlet_metrics["itpc_sentence"] > morlet_metrics["itpc_word"]
    sent_dft_gt = dft_metrics["itpc_sentence"] > dft_metrics["itpc_word"]
    print(
        f"{'Sentence > Word?':<30} {str(sent_morlet_gt):>10} {str(sent_dft_gt):>10}"
    )

    return {
        "patient_id": patient_id,
        "sfreq_hz": epochs.info["sfreq"],
        "n_trials": len(epochs),
        "morlet_sentence": morlet_metrics["itpc_sentence"],
        "morlet_word": morlet_metrics["itpc_word"],
        "morlet_ratio": morlet_metrics["ratio_sent_word"],
        "dft_sentence": dft_metrics["itpc_sentence"],
        "dft_word": dft_metrics["itpc_word"],
        "dft_ratio": dft_metrics["ratio_sent_word"],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Morlet vs DFT ITPC")
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--focus", default="LH", choices=["LH", "Clinical"])
    args = parser.parse_args()

    processor = LanguageProcessor()
    rows = []
    for pid in args.patients:
        result = compare_patient(processor, pid, focus=args.focus)
        if result:
            rows.append(result)

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Summary ===")
        print(df.to_string(index=False))
        out = config.LOCAL_DATA_ROOT / "processed" / "features" / "itpc_method_comparison.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"Saved comparison CSV: {out}")


if __name__ == "__main__":
    main()
