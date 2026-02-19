
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_processing.language_optimization import LanguageProcessor
import src.data_loading.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)


def analyze_patient(processor, patient_id, focus="LH"):
    """Run analysis for a single patient."""
    logger.info(f"Processing {patient_id}...")
    
    # 1. Load Data
    epochs = processor.process_patient(patient_id, focus=focus)
    if epochs is None:
        logger.warning(f"Skipping {patient_id}: No data found.")
        return None

    # Note: Montage is now handled better if applied early, but processor.plot_itpc_results
    # handles the plotting. However, topomap needs montage on the info.
    # We should ensure montage is set. get_data() or compute_itpc() doesn't need it,
    # but plot_topomap does. LanguageProcessor.process_patient doesn't set montage explicitly
    # unless we add it there. For now, we set it here to be safe as before.
    try:
        import mne
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning(f"Montage error for {patient_id}: {e}")

    # 2. Compute ITPC (uses default frequencies/cycles from class)
    itpc_data, itc_obj = processor.compute_itpc(epochs)
    
    # 3. Extract Metrics
    metrics = processor.extract_itpc_metrics(itpc_data)
    
    # 4. Plotting
    output_dir = config.LOCAL_DATA_ROOT / "outputs" / patient_id
    processor.plot_itpc_results(itc_obj, patient_id, output_dir, metrics)
    
    # Add metadata to metrics
    metrics["patient_id"] = patient_id
    metrics["n_trials"] = len(epochs)
    
    logger.info(f"[{patient_id}] Sentence: {metrics['itpc_sentence']:.4f} | Word: {metrics['itpc_word']:.4f} | Ratio: {metrics['ratio_sent_word']:.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run ITPC Analysis (Batch)")
    parser.add_argument("--patients", nargs="+", required=True, help="List of patient IDs (e.g., CON008 CON009)")
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
        print("\n=== Analysis Summary ===")
        print(df[["patient_id", "itpc_sentence", "itpc_word", "ratio_sent_word"]])
        
        # Save Summary
        out_path = config.LOCAL_DATA_ROOT / "processed" / "features" / "language_itpc_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved summary to {out_path}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
