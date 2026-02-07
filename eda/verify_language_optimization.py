"""
Deep data validation for Language Optimization (ENG-05).
Verifies epoch alignment, channel selection correctness, and signal quality.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.language_optimization import LanguageProcessor
from src.data_loading.unified_data_loader import UnifiedDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_patient(patient_id="CON008"):
    """Deep verification of language optimization for a patient."""

    print(f"\n{'=' * 60}")
    print(f"DEEP VERIFICATION: {patient_id}")
    print(f"{'=' * 60}\n")

    # Initialize
    loader = UnifiedDataLoader()
    processor = LanguageProcessor(loader=loader)

    # 1. Load trials and inspect
    print("1. TRIAL DATA INSPECTION")
    print("-" * 60)
    trials = loader.get_patient_trials(patient_id)
    lang_trials = trials[trials["trial_type"] == "language"]

    print(f"Total trials for {patient_id}: {len(trials)}")
    print(f"Language trials: {len(lang_trials)}")
    print(f"\nLanguage trial details:")
    print(f"  Date: {lang_trials['date'].unique()}")
    print(f"  Duration range: {lang_trials['duration'].min():.1f}s - {lang_trials['duration'].max():.1f}s")
    print(f"  Mean duration: {lang_trials['duration'].mean():.1f}s")

    # Check if trial durations make sense
    if lang_trials["duration"].min() < 10 or lang_trials["duration"].max() > 20:
        print(f"⚠️  WARNING: Unusual trial durations detected!")

    # 2. Load Raw EDF and inspect
    print(f"\n2. RAW EDF INSPECTION")
    print("-" * 60)
    raw = loader.load_edf(patient_id)

    print(f"EDF total duration: {raw.times[-1]:.1f}s ({raw.times[-1] / 60:.1f} min)")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Total channels: {len(raw.ch_names)}")
    print(f"EDF measurement date: {raw.info['meas_date']}")

    # Check EDF start time vs trial times
    edf_start_unix = raw.info["meas_date"].timestamp()
    first_trial_time = lang_trials["start_time"].min()
    last_trial_time = lang_trials["end_time"].max()

    print(f"\nTimestamp alignment check:")
    print(f"  EDF start (unix): {edf_start_unix}")
    print(f"  First trial (unix): {first_trial_time}")
    print(f"  Last trial (unix): {last_trial_time}")
    print(f"  Offset (hours): {(first_trial_time - edf_start_unix) / 3600:.1f}h")

    # 3. Process and get epochs
    print(f"\n3. EPOCH DATA INSPECTION")
    print("-" * 60)
    epochs = processor.process_patient(patient_id, focus="LH")

    if epochs is None:
        print("❌ ERROR: No epochs returned!")
        return False

    print(f"Epochs created: {len(epochs)}")
    print(f"Channels in epochs: {len(epochs.ch_names)}")
    print(f"Channel names: {epochs.ch_names}")
    print(f"Epoch duration: {epochs.times[-1] - epochs.times[0]:.1f}s")
    print(f"Epoch shape: {epochs.get_data().shape}")  # (n_epochs, n_channels, n_times)

    # 4. Verify channel selection logic
    print(f"\n4. CHANNEL SELECTION VERIFICATION")
    print("-" * 60)

    original_channels = set(raw.ch_names)
    selected_channels = set(epochs.ch_names)

    # Expected LH priority channels
    lh_priority = {"F7", "T7", "P7", "F3", "C3", "P3"}
    lh_found = lh_priority.intersection(selected_channels)
    lh_missing = lh_priority - selected_channels

    print(f"LH priority channels found: {lh_found}")
    if lh_missing:
        print(f"⚠️  LH priority channels missing: {lh_missing}")
    else:
        print(f"✅ All LH priority channels present")

    # Check Clinical 20
    clinical_20 = {
        "Fp1",
        "Fp2",
        "Fz",
        "F3",
        "F4",
        "F7",
        "F8",
        "Cz",
        "C3",
        "C4",
        "T7",
        "T8",
        "Pz",
        "P3",
        "P4",
        "P7",
        "P8",
        "O1",
        "O2",
    }
    clinical_found = clinical_20.intersection(selected_channels)
    print(f"Clinical 20 channels found: {len(clinical_found)}/19 expected")

    # 5. Signal quality inspection
    print(f"\n5. SIGNAL QUALITY INSPECTION")
    print("-" * 60)

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)

    # Check for expected signal characteristics
    print(f"Data statistics (first epoch):")
    first_epoch = data[0]  # (n_channels, n_times)

    for i, ch in enumerate(epochs.ch_names[:5]):  # First 5 channels
        ch_data = first_epoch[i]
        print(
            f"  {ch:4s}: mean={ch_data.mean():8.2f} µV, std={ch_data.std():7.2f} µV, "
            f"range=[{ch_data.min():7.2f}, {ch_data.max():7.2f}] µV"
        )

    # Check for suspicious patterns
    mean_amplitudes = np.abs(data).mean(axis=(0, 2))  # Mean across epochs and time
    std_amplitudes = data.std(axis=(0, 2))

    print(f"\nChannel-wise statistics (all epochs):")
    print(f"  Mean amplitude range: {mean_amplitudes.min():.2f} - {mean_amplitudes.max():.2f} µV")
    print(f"  Std dev range: {std_amplitudes.min():.2f} - {std_amplitudes.max():.2f} µV")

    # Check for flat channels (bad)
    flat_threshold = 0.1
    flat_channels = np.where(std_amplitudes < flat_threshold)[0]
    if len(flat_channels) > 0:
        print(f"⚠️  WARNING: {len(flat_channels)} channels appear flat (std < {flat_threshold})")
        for idx in flat_channels:
            print(f"    - {epochs.ch_names[idx]}")
    else:
        print(f"✅ No flat channels detected")

    # Check for extreme values (artifacts)
    extreme_threshold = 500  # µV
    if np.any(np.abs(data) > extreme_threshold):
        n_extreme = np.sum(np.abs(data) > extreme_threshold)
        print(f"⚠️  WARNING: {n_extreme} samples exceed {extreme_threshold} µV (possible artifacts)")
    else:
        print(f"✅ No extreme artifacts detected (>{extreme_threshold} µV)")

    # 6. Verify epoch timing alignment
    print(f"\n6. EPOCH TIMING VERIFICATION")
    print("-" * 60)

    # Get the events that were used to create epochs
    # We need to check if epochs align with expected trial times

    # Sample check: First 3 language trials
    print("Checking first 3 language trials alignment:")
    for i, (idx, trial) in enumerate(lang_trials.head(3).iterrows()):
        if i >= len(epochs):
            break

        trial_start = trial["start_time"]
        trial_duration = trial["duration"]

        # The epoch should start at trial_start (after timezone correction)
        # We can't directly verify without knowing the event times, but we can check duration
        print(f"\n  Trial {i + 1}:")
        print(f"    Expected start (unix): {trial_start}")
        print(f"    Expected duration: {trial_duration:.1f}s")
        print(f"    Epoch duration: {epochs.times[-1] - epochs.times[0]:.1f}s")

        if abs(trial_duration - (epochs.times[-1] - epochs.times[0])) > 1.0:
            print(f"    ⚠️  Duration mismatch > 1s")
        else:
            print(f"    ✅ Duration matches expected")

    # 7. Filter verification
    print(f"\n7. FILTER APPLICATION VERIFICATION")
    print("-" * 60)

    print(f"Highpass filter: {epochs.info['highpass']} Hz (expected: 0.5 Hz)")
    print(f"Lowpass filter: {epochs.info['lowpass']} Hz (expected: 30.0 Hz)")

    if epochs.info["highpass"] != 0.5 or epochs.info["lowpass"] != 30.0:
        print(f"⚠️  WARNING: Filter settings don't match specification!")
    else:
        print(f"✅ Filters correctly applied")

    # 8. Final summary
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION SUMMARY FOR {patient_id}")
    print(f"{'=' * 60}")

    checks_passed = 0
    total_checks = 7

    # Check 1: Epochs created
    if epochs is not None and len(epochs) > 0:
        print("✅ Epochs successfully created")
        checks_passed += 1
    else:
        print("❌ Failed to create epochs")

    # Check 2: LH channels present
    if len(lh_missing) == 0:
        print("✅ All LH priority channels present")
        checks_passed += 1
    else:
        print(f"⚠️  Missing LH channels: {lh_missing}")

    # Check 3: Channel count reasonable
    if 15 <= len(epochs.ch_names) <= 25:
        print(f"✅ Channel count reasonable ({len(epochs.ch_names)})")
        checks_passed += 1
    else:
        print(f"⚠️  Unexpected channel count: {len(epochs.ch_names)}")

    # Check 4: No flat channels
    if len(flat_channels) == 0:
        print("✅ No flat/dead channels")
        checks_passed += 1
    else:
        print(f"⚠️  {len(flat_channels)} flat channels detected")

    # Check 5: Filters applied
    if epochs.info["highpass"] == 0.5 and epochs.info["lowpass"] == 30.0:
        print("✅ Filters correctly applied")
        checks_passed += 1
    else:
        print("⚠️  Filter settings incorrect")

    # Check 6: Epoch count matches trial count
    if len(epochs) == len(lang_trials):
        print(f"✅ Epoch count matches trial count ({len(epochs)})")
        checks_passed += 1
    else:
        print(f"⚠️  Epoch count mismatch: {len(epochs)} epochs vs {len(lang_trials)} trials")

    # Check 7: Signal quality
    if np.any(np.abs(data) > extreme_threshold):
        print(f"⚠️  Extreme artifacts present")
    else:
        print("✅ No extreme artifacts")
        checks_passed += 1

    print(f"\n{'=' * 60}")
    print(f"PASSED: {checks_passed}/{total_checks} checks")
    print(f"{'=' * 60}\n")

    return checks_passed == total_checks


if __name__ == "__main__":
    # Test both patients
    for patient_id in ["CON008", "CON009"]:
        all_passed = verify_patient(patient_id)
        if not all_passed:
            print(f"\n⚠️  Some checks failed for {patient_id}")
