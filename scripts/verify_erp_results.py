#!/usr/bin/env python3
"""
Verification Script for ERP Pipeline Results

This script helps verify that ERP processing ran correctly by:
1. Loading and inspecting saved epochs and ERPs
2. Checking P300 features
3. Comparing with ENG-02 aligned event timestamps
4. Visualizing results
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import config


def verify_patient(patient_id: str, date: str):
    """Verify ERP results for a specific patient and session."""

    print(f"\n{'=' * 70}")
    print(f"  ERP Verification: {patient_id} - {date}")
    print(f"{'=' * 70}")

    # 1. Check aligned events from ENG-02
    print("\n[1] Checking ENG-02 Aligned Events...")
    aligned_path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"

    if not aligned_path.exists():
        print(f"  ✗ Aligned events not found: {aligned_path}")
        return False

    df = pd.read_parquet(aligned_path)
    oddball = df[df["trial_type"] == "oddball"]

    # Count rare events
    total_rare = 0
    for _, trial in oddball.iterrows():
        sentences = trial["sentences"]
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        for event in sentences:
            if isinstance(event, dict) and event.get("event") == "rare":
                total_rare += 1

    print(f"  ✓ Found {len(oddball)} oddball trials")
    print(f"  ✓ Total rare events: {total_rare}")

    # Load features early for diagnostics if available
    features_file = config.PROCESSED_DATA_DIR / "features" / f"{patient_id}_{date}_p300_features.parquet"
    session_features = None
    if features_file.exists():
        try:
            session_features = pd.read_parquet(features_file).iloc[0].to_dict()
        except Exception:
            session_features = None

    # 2. Check epochs
    print("\n[2] Checking Saved Epochs...")
    epochs_file = config.PROCESSED_DATA_DIR / "epochs" / f"{patient_id}_{date}_oddball-epo.fif"

    if not epochs_file.exists():
        print(f"  ✗ Epochs file not found: {epochs_file}")
        return False

    epochs = mne.read_epochs(epochs_file, verbose=False)
    print(f"  ✓ Epochs loaded: {len(epochs)} epochs")
    print(f"  ✓ Shape: {epochs.get_data().shape} (epochs, channels, timepoints)")
    print(f"  ✓ Time range: {epochs.tmin * 1000:.0f}ms to {epochs.tmax * 1000:.0f}ms")
    print(f"  ✓ Channels: {len(epochs.ch_names)}")

    # Check if rare events match epochs
    if len(epochs) == total_rare:
        print(f"  ✓ Epoch count matches rare events ({len(epochs)} == {total_rare})")
    else:
        print(f"  ⚠ Epoch count mismatch: {len(epochs)} epochs vs {total_rare} rare events")

    # 2b. Diagnostics from feature extraction (if available)
    print("\n[2b] Event Conversion Diagnostics...")
    if session_features:
        n_valid_pre_mne = session_features.get("n_valid_events_pre_mne")
        n_dropped_by_mne = session_features.get("n_dropped_by_mne")
        n_out_recording = session_features.get("n_out_of_recording")
        n_too_start = session_features.get("n_too_close_to_start")
        n_too_end = session_features.get("n_too_close_to_end")
        if n_valid_pre_mne is not None:
            print(f"  Pre-MNE valid events: {int(n_valid_pre_mne)} / {total_rare}")
        if n_dropped_by_mne is not None:
            print(f"  Post-MNE dropped events: {int(n_dropped_by_mne)}")
        if n_out_recording is not None:
            print(f"  Rejected out-of-recording: {int(n_out_recording)}")
        if n_too_start is not None:
            print(f"  Rejected near start boundary: {int(n_too_start)}")
        if n_too_end is not None:
            print(f"  Rejected near end boundary: {int(n_too_end)}")
    else:
        print("  ⚠ No session feature diagnostics found")

    # 3. Check ERP
    print("\n[3] Checking Averaged ERP...")
    erp_file = config.PROCESSED_DATA_DIR / "erps" / f"{patient_id}_{date}_oddball-ave.fif"

    if not erp_file.exists():
        print(f"  ✗ ERP file not found: {erp_file}")
        return False

    erp = mne.read_evokeds(erp_file, verbose=False)
    if isinstance(erp, list):
        erp = erp[0]  # read_evokeds returns a list, get first element
    print("  ✓ ERP loaded")
    print(f"  ✓ Shape: {erp.data.shape} (channels, timepoints)")
    print(f"  ✓ Averaged from {len(epochs)} epochs")

    # 4. Check baseline correction
    print("\n[4] Verifying Baseline Correction...")
    baseline_mask = (erp.times >= -0.2) & (erp.times <= 0)
    baseline_data = erp.data[:, baseline_mask]
    baseline_mean = np.mean(baseline_data) * 1e6  # Convert to µV

    if abs(baseline_mean) < 0.1:
        print(f"  ✓ Baseline is flat: {baseline_mean:.4f} µV (close to 0)")
    else:
        print(f"  ⚠ Baseline not centered: {baseline_mean:.4f} µV")

    # 5. Check P300 features
    print("\n[5] Analyzing P300 Features...")

    midline_electrodes = ["Fz", "Cz", "Pz"]
    p300_results = {}

    for electrode in midline_electrodes:
        try:
            ch_idx = [ch.upper() for ch in erp.ch_names].index(electrode.upper())
            data = erp.data[ch_idx, :] * 1e6  # µV
            times = erp.times * 1000  # ms

            # P300 window: 300-600ms
            window_mask = (erp.times >= 0.3) & (erp.times <= 0.6)
            window_data = data[window_mask]
            window_times = times[window_mask]

            peak_idx = np.argmax(window_data)
            amplitude = window_data[peak_idx]
            latency = window_times[peak_idx]

            p300_results[electrode] = {"amplitude": amplitude, "latency": latency}
            print(f"  {electrode}: {amplitude:6.2f} µV at {latency:5.1f} ms")

        except ValueError:
            print(f"  {electrode}: Not found in data")

    # 6. Expected values check
    print("\n[6] Validation Against Expected Values...")

    if "Pz" in p300_results:
        pz_amp = p300_results["Pz"]["amplitude"]
        pz_lat = p300_results["Pz"]["latency"]

        # Expected for controls: 3-10 µV, 300-500ms
        amp_valid = 3.0 <= pz_amp <= 10.0
        lat_valid = 300 <= pz_lat <= 500

        print(f"  Amplitude (Pz): {pz_amp:.2f} µV", end="")
        print(f"  {'✓ Within expected range (3-10 µV)' if amp_valid else '⚠ Outside expected range'}")

        print(f"  Latency (Pz):   {pz_lat:.1f} ms", end="")
        print(f"  {'✓ Within expected range (300-500ms)' if lat_valid else '⚠ Outside expected range'}")

    # 7. Research-Grade QC Analysis (Cleaned Schema)
    print("\n[7] Research-Grade QC Analysis...")

    if session_features and "p300_n_valid_electrodes" in session_features:
        # Display QC notes prominently
        qc_notes = session_features.get("qc_notes", "")
        if qc_notes:
            if "inverted" in qc_notes.lower() or "flagged" in qc_notes.lower():
                print(f"  ⚠ QC Notes: {qc_notes}")
            else:
                print(f"  ✓ QC Notes: {qc_notes}")

        # Display composite metrics
        n_valid = session_features.get("p300_n_valid_electrodes", 0)
        n_flagged = session_features.get("p300_n_flagged_electrodes", 0)
        composite_amp = session_features.get("p300_composite_amplitude_uV", np.nan)
        best_elec = session_features.get("p300_best_electrode", None)

        print(f"  Valid electrodes: {n_valid}/3")
        if not np.isnan(composite_amp):
            print(f"  Composite amplitude: {composite_amp:.2f} µV")
        if best_elec:
            print(f"  Best electrode: {best_elec}")
        if n_flagged > 0:
            print(f"  Flagged electrodes: {n_flagged}/3")
    elif session_features and "qc_notes" in session_features:
        # Custom electrode mode
        qc_notes = session_features.get("qc_notes", "")
        print(f"  {qc_notes}")
    else:
        print("  ⚠ Composite features not found (pipeline may need re-run with updated code)")

    # 8. Check plot
    print("\n[8] Checking ERP Plot...")
    plot_file = config.PROCESSED_DATA_DIR / "plots" / "erp" / f"{patient_id}_{date}_oddball_erp.png"

    if plot_file.exists():
        print(f"  ✓ Plot saved: {plot_file}")
        print(f"  ✓ File size: {plot_file.stat().st_size / 1024:.1f} KB")
    else:
        print("  ✗ Plot not found")

    # 9. Cross-check with aligned events
    print("\n[9] Cross-Checking Timestamps with ENG-02...")

    # Get timestamps from aligned events
    aligned_timestamps = []
    for _, trial in oddball.iterrows():
        sentences = trial["sentences"]
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        for event in sentences:
            if isinstance(event, dict) and event.get("event") == "rare" and "event_start" in event:
                aligned_timestamps.append(event["event_start"])

    print(f"  ENG-02 aligned timestamps: {len(aligned_timestamps)} rare events")
    print(f"  ENG-02b created epochs:    {len(epochs)} epochs")

    if len(epochs) == len(aligned_timestamps):
        print("  ✓ Perfect match: All aligned events converted to epochs")
    else:
        print(f"  ⚠ Mismatch: {len(aligned_timestamps) - len(epochs)} events not converted")

    # 10. Timezone and confidence diagnostics
    print("\n[10] Timezone / Confidence Diagnostics...")
    if session_features:
        tz_hours = session_features.get("timezone_offset_hours")
        tz_conf = session_features.get("timezone_confidence")
        tz_warn = session_features.get("timezone_warning_flag")
        diagnostic_note = session_features.get("diagnostic_note")
        if tz_hours is not None:
            print(f"  Timezone offset applied: {tz_hours:.2f} hours")
        if tz_conf is not None:
            marker = "✓" if tz_conf == "high" and not tz_warn else "⚠"
            print(f"  {marker} Timezone confidence: {tz_conf}")
        if diagnostic_note:
            print(f"  Note: {diagnostic_note}")
    else:
        print("  ⚠ No timezone diagnostics available for this session")

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Verification Summary")
    print(f"{'=' * 70}")
    status_marker = "✓ PASSED" if len(epochs) > 0 else "✗ FAILED"
    print(f"  Status: {status_marker}")
    print(f"  Epochs created: {len(epochs)}")
    print("  Files saved: Epochs, ERP, Plot")
    if "Pz" in p300_results:
        print(f"  P300 detected: {p300_results['Pz']['amplitude']:.2f} µV at {p300_results['Pz']['latency']:.1f} ms")
    print(f"{'=' * 70}\n")

    return True


def compare_with_alignment_report(patient_id: str):
    """Compare ERP results with ENG-02 alignment report."""

    print(f"\n{'=' * 70}")
    print("  Comparing with ENG-02 Alignment Report")
    print(f"{'=' * 70}")

    # Load aligned events
    aligned_path = config.ALIGNED_EVENTS_DIR / f"{patient_id}_events.parquet"
    df = pd.read_parquet(aligned_path)

    # Get oddball stats
    oddball = df[df["trial_type"] == "oddball"]

    print("\nENG-02 Alignment Statistics:")
    print(f"  Oddball trials: {len(oddball)}")

    # Count events by type
    total_events = 0
    rare_events = 0
    standard_events = 0

    for _, trial in oddball.iterrows():
        sentences = trial["sentences"]
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        for event in sentences:
            if isinstance(event, dict):
                total_events += 1
                if event.get("event") == "rare":
                    rare_events += 1
                elif event.get("event") == "standard":
                    standard_events += 1

    print(f"  Total events:    {total_events}")
    print(f"  Rare events:     {rare_events}")
    print(f"  Standard events: {standard_events}")

    # Check ERP results
    epochs_files = list((config.PROCESSED_DATA_DIR / "epochs").glob(f"{patient_id}_*_oddball-epo.fif"))

    if epochs_files:
        print("\nENG-02b ERP Results:")
        total_epochs = 0
        for epochs_file in epochs_files:
            epochs = mne.read_epochs(epochs_file, verbose=False)
            total_epochs += len(epochs)
            print(f"  {epochs_file.name}: {len(epochs)} epochs")

        print("\nComparison:")
        print(f"  ENG-02 rare events: {rare_events}")
        print(f"  ENG-02b epochs:     {total_epochs}")

        if total_epochs == rare_events:
            print("  ✓ Perfect match!")
        else:
            print(f"  ⚠ Difference: {rare_events - total_epochs} events not converted to epochs")
    else:
        print(f"\n✗ No ERP results found for {patient_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify ERP pipeline results")
    parser.add_argument("--patient", type=str, required=True, help="Patient ID")
    parser.add_argument("--date", type=str, help="Session date (YYYY-MM-DD)")
    parser.add_argument("--compare", action="store_true", help="Compare with ENG-02 alignment report")

    args = parser.parse_args()

    # Auto-detect date if not provided
    if not args.date:
        # Find available dates for this patient
        epochs_files = list((config.PROCESSED_DATA_DIR / "epochs").glob(f"{args.patient}_*_oddball-epo.fif"))
        if epochs_files:
            # Extract date from filename
            date = epochs_files[0].stem.split("_")[1]
            args.date = date
            print(f"Auto-detected date: {date}")
        else:
            print(f"Error: No epochs found for {args.patient}. Please specify --date")
            sys.exit(1)

    # Run verification
    verify_patient(args.patient, args.date)

    # Optional comparison with alignment
    if args.compare:
        compare_with_alignment_report(args.patient)
