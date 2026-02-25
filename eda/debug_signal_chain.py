"""
Debug script to investigate why epoch data is all zeros.
"""

import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.data_processing.language_optimization import LanguageProcessor
from src.data_processing.timestamp_aligner import TimestampAligner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_signal_chain(patient_id="CON008"):
    """Trace through the signal processing chain to find where data becomes zero."""

    print(f"\n{'=' * 60}")
    print(f"DEBUGGING SIGNAL CHAIN: {patient_id}")
    print(f"{'=' * 60}\n")

    # 0. Align Timestamps
    print("STEP 0: Align Timestamps")
    print("-" * 60)
    aligner = TimestampAligner(patient_id=patient_id)
    aligned_events = aligner.align(save=False)[patient_id]
    print(f"Aligned {len(aligned_events)} events")

    loader = UnifiedDataLoader()
    processor = LanguageProcessor(loader=loader)

    # Step 1: Load raw EDF
    print("STEP 1: Load Raw EDF")
    print("-" * 60)
    raw = loader.load_edf(patient_id)

    # Check raw data
    raw_data = raw.get_data()
    print(f"Raw data shape: {raw_data.shape}")
    print("Raw data stats (all channels):")
    print(f"  Min: {raw_data.min():.6f}")
    print(f"  Max: {raw_data.max():.6f}")
    print(f"  Mean: {raw_data.mean():.6f}")
    print(f"  Std: {raw_data.std():.6f}")

    # Check a specific channel
    ch_idx = raw.ch_names.index("F7") if "F7" in raw.ch_names else 0
    ch_name = raw.ch_names[ch_idx]
    ch_data = raw_data[ch_idx]
    print(f"\n{ch_name} channel (before any processing):")
    print(f"  First 10 samples: {ch_data[:10]}")
    print(f"  Mean: {ch_data.mean():.6f}")
    print(f"  Std: {ch_data.std():.6f}")

    # Step 2: After channel selection
    print("\nSTEP 2: After Channel Selection")
    print("-" * 60)
    raw_selected = processor.select_optimal_channels(raw.copy(), focus="LH")
    data_selected = raw_selected.get_data()

    print(f"Selected data shape: {data_selected.shape}")
    print("Selected data stats:")
    print(f"  Min: {data_selected.min():.6f}")
    print(f"  Max: {data_selected.max():.6f}")
    print(f"  Mean: {data_selected.mean():.6f}")
    print(f"  Std: {data_selected.std():.6f}")

    if ch_name in raw_selected.ch_names:
        ch_idx_sel = raw_selected.ch_names.index(ch_name)
        ch_data_sel = data_selected[ch_idx_sel]
        print(f"\n{ch_name} after selection:")
        print(f"  First 10 samples: {ch_data_sel[:10]}")
        print(f"  Mean: {ch_data_sel.mean():.6f}")
        print(f"  Std: {ch_data_sel.std():.6f}")

    # Step 3: After filtering
    print("\nSTEP 3: After Filtering (0.5-30 Hz)")
    print("-" * 60)
    raw_filtered = processor.preprocess_signal(raw_selected)
    data_filtered = raw_filtered.get_data()

    print(f"Filtered data shape: {data_filtered.shape}")
    print("Filtered data stats:")
    print(f"  Min: {data_filtered.min():.6f}")
    print(f"  Max: {data_filtered.max():.6f}")
    print(f"  Mean: {data_filtered.mean():.6f}")
    print(f"  Std: {data_filtered.std():.6f}")

    if ch_name in raw_filtered.ch_names:
        ch_idx_filt = raw_filtered.ch_names.index(ch_name)
        ch_data_filt = data_filtered[ch_idx_filt]
        print(f"\n{ch_name} after filtering:")
        print(f"  First 10 samples: {ch_data_filt[:10]}")
        print(f"  Mean: {ch_data_filt.mean():.6f}")
        print(f"  Std: {ch_data_filt.std():.6f}")

    # Step 4: Check units
    print("\nSTEP 4: Check Data Units")
    print("-" * 60)

    # MNE stores data internally in Volts, but displays in µV
    # Let's check the actual scaling
    print(f"Raw data range in Volts: [{raw_data.min():.6e}, {raw_data.max():.6e}]")
    print(f"Raw data range in µV: [{raw_data.min() * 1e6:.2f}, {raw_data.max() * 1e6:.2f}]")

    # Step 5: Check epochs
    print("\nSTEP 5: Create Epochs")
    print("-" * 60)

    epochs = processor.process_patient(patient_id, focus="LH")
    epoch_data = epochs.get_data()

    print(f"Epoch data shape: {epoch_data.shape}")
    print("Epoch data stats:")
    print(f"  Min: {epoch_data.min():.6e}")
    print(f"  Max: {epoch_data.max():.6e}")
    print(f"  Mean: {epoch_data.mean():.6e}")
    print(f"  Std: {epoch_data.std():.6e}")

    # Check in µV
    print("\nEpoch data in µV:")
    print(f"  Min: {epoch_data.min() * 1e6:.2f} µV")
    print(f"  Max: {epoch_data.max() * 1e6:.2f} µV")
    print(f"  Mean: {epoch_data.mean() * 1e6:.2f} µV")
    print(f"  Std: {epoch_data.std() * 1e6:.2f} µV")

    # Check first epoch, first channel
    print(f"\nFirst epoch, first channel ({epochs.ch_names[0]}):")
    first_ch_first_epoch = epoch_data[0, 0, :]
    print(f"  First 10 samples (V): {first_ch_first_epoch[:10]}")
    print(f"  First 10 samples (µV): {first_ch_first_epoch[:10] * 1e6}")
    print(f"  Mean (µV): {first_ch_first_epoch.mean() * 1e6:.2f}")
    print(f"  Std (µV): {first_ch_first_epoch.std() * 1e6:.2f}")


if __name__ == "__main__":
    debug_signal_chain("CON008")
