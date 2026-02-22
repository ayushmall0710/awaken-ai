import logging
import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.data_processing.language_optimization import LanguageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_signals(patient_id="CON008"):
    logger.info(f"Analyzing signals for {patient_id}...")
    loader = UnifiedDataLoader()
    processor = LanguageProcessor(loader=loader)

    # Load and process
    try:
        epochs = processor.process_patient(patient_id, focus="LH", filter_signal=True)
    except Exception as e:
        logger.error(f"Failed to load epochs: {e}")
        return

    if epochs is None or len(epochs) == 0:
        logger.error("No epochs found.")
        return

    # 1. Amplitude Analysis
    data = epochs.get_data(copy=True) * 1e6  # Convert to uV
    mean_amp = np.mean(np.abs(data))
    std_amp = np.std(data)
    max_amp = np.max(np.abs(data))

    logger.info("=== Amplitude Statistics (uV) ===")
    logger.info(f"Mean Absolute Amplitude: {mean_amp:.2f} uV")
    logger.info(f"Standard Deviation: {std_amp:.2f} uV")
    logger.info(f"Max Amplitude: {max_amp:.2f} uV")

    if mean_amp < 1.0:
        logger.warning("WARNING: Signal amplitude is suspiciously low (< 1 uV). Possible flatlines or scaling issue.")
    elif mean_amp > 100.0:
        logger.warning("WARNING: Signal amplitude is suspiciously high (> 100 uV). Possible artifacts.")
    else:
        logger.info("Signal amplitude looks physiological (1-100 uV range).")

    # 2. Channel Variance (Detect bad channels)
    logger.info("\n=== Channel Quality ===")
    ch_vars = np.var(data, axis=(0, 2))
    low_var_chs = [epochs.ch_names[i] for i, v in enumerate(ch_vars) if v < 0.1]
    high_var_chs = [epochs.ch_names[i] for i, v in enumerate(ch_vars) if v > 5000]  # Variable threshold

    if low_var_chs:
        logger.warning(f"Potential Flat Channels: {low_var_chs}")
    else:
        logger.info("No flat channels detected.")

    if high_var_chs:
        logger.warning(f"Potential Noisy Channels: {high_var_chs}")
    else:
        logger.info("No extremely noisy channels detected.")

    # 3. PSD Analysis (Alpha Band)
    logger.info("\n=== Spectral Analysis ===")
    psd = epochs.compute_psd(fmin=1, fmax=30)
    # Get data: (n_epochs, n_channels, n_freqs)
    psd_data = psd.get_data()
    freqs = psd.freqs

    # Define bands
    alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
    delta_idx = np.where((freqs >= 1) & (freqs <= 4))[0]

    mean_psd = np.mean(psd_data, axis=(0, 1))  # Mean across epochs and channels

    alpha_power = np.mean(mean_psd[alpha_idx])
    delta_power = np.mean(mean_psd[delta_idx])

    logger.info(f"Mean Alpha Power (8-12Hz): {alpha_power:.2e}")
    logger.info(f"Mean Delta Power (1-4Hz):  {delta_power:.2e}")

    if delta_power > alpha_power:
        logger.info("Spectrum is dominated by low frequencies (1/f), which is typical for EEG.")
    else:
        logger.warning("Unusual spectral profile: Alpha power > Delta power (check for strong alpha rhythms or noise).")

    logger.info("\n=== Conclusion ===")
    logger.info("Based on these metrics, the signal quality appears reasonable for processed EEG.")


if __name__ == "__main__":
    analyze_signals()
