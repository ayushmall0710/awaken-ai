"""
Re-analysis of ITPC using both Morlet wavelet and DFT methods.

Updates run_itpc_analysis.py results with both Morlet ITPC (existing) and
DFT ITPC (Sokoliuk 2021 method) to allow cross-method comparison.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import src.data_loading.config as config
from src.data_processing.language_optimization import LanguageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def analyze_patient(processor, patient_id, focus="LH"):
    """Run full ITPC analysis for a patient using both Morlet and DFT methods."""
    import mne

    logger.info(f"Processing {patient_id}...")

    epochs = processor.process_patient(patient_id, focus=focus)
    if epochs is None:
        logger.warning(f"Skipping {patient_id}: No data found.")
        return None

    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning(f"Montage error for {patient_id}: {e}")

    # --- Morlet ITPC ---
    logger.info(f"[{patient_id}] Computing Morlet ITPC...")
    itpc_data, itc_obj = processor.compute_itpc(epochs)
    morlet_metrics = processor.extract_itpc_metrics(itpc_data)

    output_dir = config.LOCAL_DATA_ROOT / "outputs" / patient_id
    processor.plot_itpc_results(itc_obj, patient_id, output_dir, morlet_metrics)

    # --- DFT ITPC ---
    logger.info(f"[{patient_id}] Computing DFT ITPC...")
    itpc_spectrum, dft_freqs = processor.compute_itpc_dft(epochs)
    dft_metrics = processor.extract_itpc_metrics_dft(itpc_spectrum, dft_freqs)

    # Build combined result
    result = {
        "patient_id": patient_id,
        "n_trials": len(epochs),
        "sfreq": epochs.info["sfreq"],
        # Morlet
        "morlet_itpc_sentence": morlet_metrics["itpc_sentence"],
        "morlet_itpc_word": morlet_metrics["itpc_word"],
        "morlet_ratio_sent_word": morlet_metrics["ratio_sent_word"],
        "morlet_freq_sentence_hz": morlet_metrics["freq_sentence_hz"],
        "morlet_freq_word_hz": morlet_metrics["freq_word_hz"],
        # DFT
        "dft_itpc_sentence": dft_metrics["itpc_sentence"],
        "dft_itpc_word": dft_metrics["itpc_word"],
        "dft_ratio_sent_word": dft_metrics["ratio_sent_word"],
        "dft_freq_sentence_hz": dft_metrics["freq_sentence_hz"],
        "dft_freq_word_hz": dft_metrics["freq_word_hz"],
    }

    logger.info(
        f"[{patient_id}] Morlet -- Sentence: {morlet_metrics['itpc_sentence']:.4f} | "
        f"Word: {morlet_metrics['itpc_word']:.4f} | Ratio: {morlet_metrics['ratio_sent_word']:.2f}"
    )
    logger.info(
        f"[{patient_id}] DFT    -- Sentence: {dft_metrics['itpc_sentence']:.4f} | "
        f"Word: {dft_metrics['itpc_word']:.4f} | Ratio: {dft_metrics['ratio_sent_word']:.2f}"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ITPC Analysis (Morlet + DFT)")
    parser.add_argument("--patients", nargs="+", required=True, help="Patient IDs (e.g., CON008 CON009)")
    parser.add_argument("--focus", type=str, default="LH", choices=["LH", "Clinical"], help="Channel focus")
    args = parser.parse_args()

    processor = LanguageProcessor()
    results = []

    for pid in args.patients:
        res = analyze_patient(processor, pid, focus=args.focus)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results)

        print("\n=== ITPC Analysis Summary (Morlet vs DFT) ===")
        print(
            df[
                [
                    "patient_id",
                    "n_trials",
                    "sfreq",
                    "morlet_itpc_sentence",
                    "morlet_itpc_word",
                    "morlet_ratio_sent_word",
                    "dft_itpc_sentence",
                    "dft_itpc_word",
                    "dft_ratio_sent_word",
                ]
            ].to_string(index=False)
        )

        out_path = config.LOCAL_DATA_ROOT / "processed" / "features" / "language_itpc_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved summary to {out_path}")
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
