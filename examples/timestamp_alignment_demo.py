"""
Example: Timestamp Alignment Demo

This script demonstrates how to use the EEGDataLoader and TimestampAligner
classes to synchronize CSV timestamps with EDF data using the DC audio channel.

Usage:
    python examples/timestamp_alignment_demo.py --edf /path/to/file.EDF --csv /path/to/file.csv

Requirements:
    - EDF file with DC audio channel
    - CSV file with trial timing information
"""

import argparse
from pathlib import Path
import sys

# Try package import first, fall back to path manipulation if not installed
try:
    from data_loading import EEGDataLoader, TimestampAligner
except ImportError:
    # Development mode: add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from data_loading import EEGDataLoader, TimestampAligner

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate timestamp alignment using DC audio channel"
    )
    parser.add_argument(
        '--edf', 
        type=str, 
        required=True,
        help='Path to EDF file'
    )
    parser.add_argument(
        '--csv', 
        type=str,
        required=False,
        help='Path to CSV stimulus log file'
    )
    parser.add_argument(
        '--dc-channel',
        type=str,
        help='Name of DC audio channel (auto-detected if not provided)'
    )
    parser.add_argument(
        '--trial-type',
        type=str,
        default='oddball',
        help='Trial type to analyze (default: oddball)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EEG Timestamp Alignment Demo")
    print("=" * 70)
    
    # Step 1: Load EDF and CSV data
    print("\n[1] Loading data...")
    loader = EEGDataLoader(
        edf_path=args.edf,
        csv_path=args.csv if args.csv else None,
        preload=True
    )
    
    info = loader.get_info()
    print(f"    Patient ID: {info['patient_id']}")
    print(f"    Sampling frequency: {info['sampling_frequency']} Hz")
    print(f"    Duration: {info['duration_seconds']:.1f} seconds")
    print(f"    Channels: {info['n_channels']}")
    print(f"    Recording start: {info['recording_start']}")
    
    # Step 2: Initialize timestamp aligner
    print("\n[2] Initializing timestamp aligner...")
    aligner = TimestampAligner(
        eeg_loader=loader,
        dc_channel_name=args.dc_channel
    )
    print(f"    DC channel: {aligner.dc_channel_name}")
    
    # Step 3: Extract DC channel and detect peaks
    print("\n[3] Extracting DC channel and detecting stimulus onsets...")
    dc_data, dc_times = aligner.extract_dc_channel()
    peak_times, peak_values = aligner.detect_stimulus_onsets(
        dc_data, 
        dc_times,
        min_distance=0.5  # Minimum 0.5s between peaks
    )
    print(f"    DC channel duration: {dc_times[-1]:.1f} seconds")
    print(f"    Detected {len(peak_times)} stimulus onsets")
    
    if len(peak_times) > 0:
        # Convert to Unix timestamps
        peak_times_unix = aligner.edf_time_to_unix(peak_times)
        
        print(f"\n    First 5 detected onsets (EDF time):")
        for i in range(min(5, len(peak_times))):
            print(f"      {i+1}. {peak_times[i]:.3f}s (Unix: {peak_times_unix[i]:.3f})")
        
        # Calculate inter-stimulus intervals
        if len(peak_times) > 1:
            isis = np.diff(peak_times)
            print(f"\n    Inter-stimulus intervals:")
            print(f"      Mean: {np.mean(isis):.3f}s")
            print(f"      Std:  {np.std(isis):.3f}s")
            print(f"      Range: [{np.min(isis):.3f}, {np.max(isis):.3f}]s")
    
    # Step 4: Analyze specific trials if CSV is provided
    if loader.stimulus_df is not None and args.trial_type:
        print(f"\n[4] Analyzing {args.trial_type} trials...")
        
        # Get trials of specified type
        trial_mask = loader.stimulus_df['trial_type'].str.contains(
            args.trial_type, 
            case=False, 
            na=False
        )
        trials = loader.stimulus_df[trial_mask]
        
        print(f"    Found {len(trials)} {args.trial_type} trials")
        
        # Analyze first trial as example
        if len(trials) > 0:
            trial = trials.iloc[0]
            print(f"\n    Example trial:")
            print(f"      Start: {trial['start_time']}")
            print(f"      End: {trial['end_time']}")
            print(f"      Duration: {trial['duration']:.2f}s")
            
            # Synchronize this trial
            alignment_df, metrics = aligner.synchronize_trial(
                trial_start_unix=trial['start_time'],
                trial_end_unix=trial['end_time']
            )
            
            print(f"\n    Synchronization results:")
            print(f"      Peaks detected: {metrics['n_peaks_detected']}")
            print(f"      Trial duration: {metrics['trial_duration']:.2f}s")
            if not np.isnan(metrics['mean_isi']):
                print(f"      Mean ISI: {metrics['mean_isi']:.3f}s")
            
            if len(alignment_df) > 0:
                print(f"\n    First 5 aligned events:")
                print(alignment_df.head().to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Validate alignment precision with known stimulus timing")
    print("  2. Use aligned timestamps for epoch extraction")
    print("  3. Apply to full dataset for ERP analysis")
    

if __name__ == '__main__':
    main()
