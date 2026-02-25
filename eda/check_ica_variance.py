"""
Quick script to check how much variance is removed by ICA.
"""

import logging
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.data_processing.artifact_rejection import ArtifactRejector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)


def check_variance(patient_id):
    print(f"\n--- Checking ICA Variance for {patient_id} ---")
    loader = UnifiedDataLoader()
    rejector = ArtifactRejector(loader=loader)

    # Get the session date
    sessions = loader.get_patient_sessions(patient_id)
    if not sessions:
        print(f"No sessions found for {patient_id}")
        return
    date = sessions[0]

    # Load raw data
    raw = loader.load_edf(patient_id, date)
    if raw is None:
        print(f"Failed to load raw true EEG for {patient_id}")
        return

    # 1. Global Variance before ICA
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
    data_raw_global = raw.get_data(picks=picks_eeg)
    var_raw_global = np.var(data_raw_global, axis=1).mean()

    # 2. LH Focus Channels Variance before ICA
    from src.data_processing.language_optimization import LanguageProcessor

    processor = LanguageProcessor()
    lh_picks = []
    for ch_name in processor.LH_FOCUS_CHANNELS:
        for i, raw_ch in enumerate(raw.ch_names):
            if ch_name.lower() in raw_ch.lower():
                lh_picks.append(i)
                break

    if lh_picks:
        print(f"Tracking {len(lh_picks)} LH channels: {[raw.ch_names[i] for i in lh_picks]}")
        data_raw_lh = raw.get_data(picks=lh_picks)
        var_raw_lh = np.var(data_raw_lh, axis=1).mean()
    else:
        var_raw_lh = None

    # 3. Apply ICA
    print("Applying ICA pipeline (fitting & rejection)...")
    raw_clean, ica_summary = rejector._apply_ica(raw)

    print(f"Excluded components: {ica_summary.excluded}")

    if not ica_summary.excluded:
        print("No components removed.")
        return

    # 4. Global Variance after ICA
    data_clean_global = raw_clean.get_data(picks=picks_eeg)
    var_clean_global = np.var(data_clean_global, axis=1).mean()
    var_removed_global = var_raw_global - var_clean_global
    pct_removed_global = (var_removed_global / var_raw_global) * 100

    print("\n--- GLOBAL EEG Variance ---")
    print(f"Variance Before: {var_raw_global:.4e}")
    print(f"Variance After:  {var_clean_global:.4e}")
    print(f"Variance Removed: {pct_removed_global:.2f}%")

    # 5. LH Channels Variance after ICA
    if var_raw_lh is not None:
        data_clean_lh = raw_clean.get_data(picks=lh_picks)
        var_clean_lh = np.var(data_clean_lh, axis=1).mean()
        var_removed_lh = var_raw_lh - var_clean_lh
        pct_removed_lh = (var_removed_lh / var_raw_lh) * 100

        print("\n--- LH FOCUS CHANNELS Variance ---")
        print(f"Variance Before: {var_raw_lh:.4e}")
        print(f"Variance After:  {var_clean_lh:.4e}")
        print(f"Variance Removed: {pct_removed_lh:.2f}%")


if __name__ == "__main__":
    check_variance("CON008")
    check_variance("CON009")
