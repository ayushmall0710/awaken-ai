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
from src.pipelines.language_tracking import LanguageTrackingAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ITPC Analysis (Morlet + DFT)")
    parser.add_argument("--patients", nargs="+", required=True, help="Patient IDs (e.g., CON008 CON009)")

    parser.add_argument("--focus", type=str, default="LH", choices=["LH", "RH", "Clinical"], help="Channel focus")
    args = parser.parse_args()

    results = []

    for pid in args.patients:
        pipeline = LanguageTrackingAnalysis(focus=args.focus)
        res = pipeline.run(patient_id=pid)
        if res is not None and not res.empty:
            results.append(res)

    if results:
        df = pd.concat(results, ignore_index=True)

        print("\n=== ITPC Analysis Summary (Morlet vs DFT) ===")
        print(
            df[
                [
                    "patient_id",
                    "n_trials",
                    "sfreq",
                    "focus",
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
