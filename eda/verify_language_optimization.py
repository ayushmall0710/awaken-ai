"""
Deep data validation for Language Optimization (ENG-05).
Verifies channel selection correctness and signal quality using the current API.
"""

import logging
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.data_processing.language_optimization import LanguageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_patient(patient_id="CON008"):
    """Deep verification of language optimization for a patient."""

    print(f"\n{'=' * 60}")
    print(f"DEEP VERIFICATION: {patient_id}")
    print(f"{'=' * 60}\n")

    loader = UnifiedDataLoader()
    processor = LanguageProcessor(loader=loader)

    # 1. Load and process
    print("1. EPOCH DATA")
    print("-" * 60)

    epochs = processor.process_patient(patient_id, focus="LH")

    if epochs is None:
        print("[FAIL] ERROR: No epochs returned!")
        return False

    print(f"Epochs created: {len(epochs)}")
    print(f"Channels: {len(epochs.ch_names)}")
    print(f"Channel names: {epochs.ch_names}")
    print(f"Epoch duration: {epochs.times[-1] - epochs.times[0]:.1f}s")
    print(f"Sampling rate: {epochs.info['sfreq']} Hz")
    print(f"Epoch shape: {epochs.get_data().shape}")  # (n_epochs, n_channels, n_times)

    # 2. Channel selection
    print("\n2. CHANNEL SELECTION")
    print("-" * 60)

    selected_channels = set(epochs.ch_names)
    lh_priority = {"F7", "T7", "P7", "F3", "C3", "P3"}
    lh_found = lh_priority.intersection(selected_channels)
    lh_missing = lh_priority - selected_channels

    print(f"LH priority channels found: {lh_found}")
    if lh_missing:
        print(f"[WARN] LH priority channels missing: {lh_missing}")
    else:
        print("[PASS] All LH priority channels present")

    clinical_20 = {
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "Cz", "C3", "C4",
        "T7", "T8", "Pz", "P3", "P4", "P7", "P8", "O1", "O2",
    }
    clinical_found = clinical_20.intersection(selected_channels)
    print(f"Clinical 20 channels found: {len(clinical_found)}/19 expected")

    # 3. Signal quality
    print("\n3. SIGNAL QUALITY")
    print("-" * 60)

    data = epochs.get_data() * 1e6  # Volts to uV

    mean_amplitudes = np.abs(data).mean(axis=(0, 2))
    std_amplitudes = data.std(axis=(0, 2))

    print(f"Mean amplitude range: {mean_amplitudes.min():.2f} - {mean_amplitudes.max():.2f} uV")
    print(f"Std dev range: {std_amplitudes.min():.2f} - {std_amplitudes.max():.2f} uV")

    flat_threshold = 0.1
    flat_channels = np.where(std_amplitudes < flat_threshold)[0]
    if len(flat_channels) > 0:
        print(f"[WARN] {len(flat_channels)} flat channels (std < {flat_threshold})")
        for idx in flat_channels:
            print(f"    - {epochs.ch_names[idx]}")
    else:
        print("[PASS] No flat channels detected")

    extreme_threshold = 500  # uV
    if np.any(np.abs(data) > extreme_threshold):
        n_extreme = np.sum(np.abs(data) > extreme_threshold)
        print(f"[WARN] {n_extreme} samples exceed {extreme_threshold} uV (possible artifacts)")
    else:
        print(f"[PASS] No extreme artifacts (>{extreme_threshold} uV)")

    # 4. Filter verification
    print("\n4. FILTER VERIFICATION")
    print("-" * 60)

    print(f"Highpass: {epochs.info['highpass']} Hz (expected: 0.5 Hz)")
    print(f"Lowpass:  {epochs.info['lowpass']} Hz (expected: 30.0 Hz)")

    if epochs.info["highpass"] != 0.5 or epochs.info["lowpass"] != 30.0:
        print("[WARN] Filter settings don't match specification!")
    else:
        print("[PASS] Filters correctly applied")

    # 5. Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {patient_id}")
    print(f"{'=' * 60}")

    checks_passed = 0
    total_checks = 5

    if epochs is not None and len(epochs) > 0:
        print("[PASS] Epochs created")
        checks_passed += 1
    else:
        print("[FAIL] No epochs")

    if len(lh_missing) == 0:
        print("[PASS] All LH channels present")
        checks_passed += 1
    else:
        print(f"[WARN] Missing LH channels: {lh_missing}")

    if 15 <= len(epochs.ch_names) <= 25:
        print(f"[PASS] Channel count reasonable ({len(epochs.ch_names)})")
        checks_passed += 1
    else:
        print(f"[WARN] Unexpected channel count: {len(epochs.ch_names)}")

    if len(flat_channels) == 0:
        print("[PASS] No flat channels")
        checks_passed += 1
    else:
        print(f"[WARN] {len(flat_channels)} flat channels")

    if epochs.info["highpass"] == 0.5 and epochs.info["lowpass"] == 30.0:
        print("[PASS] Filters correct")
        checks_passed += 1
    else:
        print("[WARN] Filter mismatch")

    print(f"\nPASSED: {checks_passed}/{total_checks} checks")
    print(f"{'=' * 60}\n")

    return checks_passed == total_checks


if __name__ == "__main__":
    for patient_id in ["CON008", "CON009"]:
        all_passed = verify_patient(patient_id)
        if not all_passed:
            print(f"[WARN] Some checks failed for {patient_id}")
