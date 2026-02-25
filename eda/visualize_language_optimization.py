import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading.unified_data_loader import UnifiedDataLoader
from src.data_processing.language_optimization import LanguageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eda/figures/language_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_patient(patient_id: str = "CON009"):
    logger.info("Initializing LanguageProcessor...")
    loader = UnifiedDataLoader()
    processor = LanguageProcessor(loader=loader)

    logger.info(f"Processing patient {patient_id}...")
    # 1. Load Data (cleaned epochs) and optimize
    try:
        epochs = processor.process_patient(patient_id, focus="LH", filter_signal=True)
    except Exception as e:
        logger.error(f"Failed to process patient: {e}")
        return

    if epochs is None:
        logger.error("No epochs returned.")
        return

    logger.info(f"Processed {len(epochs)} epochs.")

    # 2. Visualize Channel Selection (Topomap)
    # create_info requires a montage to plot topomaps.
    # We'll try to set standard_1020 if not already set.
    try:
        epochs.set_montage("standard_1020")
    except ValueError:
        logger.warning("Could not set standard_1020 montage. Skipping topomap.")

    fig_sensors = epochs.plot_sensors(show_names=True, show=False)
    fig_sensors.savefig(OUTPUT_DIR / f"{patient_id}_sensors.png")
    logger.info(f"Saved sensor map to {OUTPUT_DIR / f'{patient_id}_sensors.png'}")

    # 3. Evoked Response (Average)
    evoked = epochs.average()
    fig_erp = evoked.plot(spatial_colors=True, show=False, gfp=True)
    # plot() returns a figure or list of figures. Handle both.
    if isinstance(fig_erp, list):
        fig_erp = fig_erp[0]
    fig_erp.savefig(OUTPUT_DIR / f"{patient_id}_erp.png")
    logger.info(f"Saved ERP plot to {OUTPUT_DIR / f'{patient_id}_erp.png'}")

    # 4. Power Spectral Density (PSD)
    logger.info("Computing PSD...")
    fig_psd = epochs.compute_psd(fmin=0.5, fmax=30.0).plot(show=False)
    fig_psd.savefig(OUTPUT_DIR / f"{patient_id}_psd.png")
    logger.info(f"Saved PSD plot to {OUTPUT_DIR / f'{patient_id}_psd.png'}")

    # 5. Spectrogram of a key language channel (F7)
    target_ch = "F7"
    if target_ch in epochs.ch_names:
        logger.info(f"Generating spectrogram for {target_ch}...")
        # Get data for F7: (n_epochs, n_times) -> flattened
        # get_data returns (n_epochs, n_channels, n_times)
        f7_data = epochs.get_data(picks=[target_ch])[:, 0, :].flatten()

        plt.figure(figsize=(10, 6))
        # Reduce Fs to epoch sfreq
        plt.specgram(f7_data, Fs=epochs.info["sfreq"], NFFT=1024, noverlap=512, cmap="viridis")
        plt.title(f"Spectrogram - {target_ch} ({patient_id})")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (seconds)")
        plt.ylim(0, 30)  # Focus on relevant band
        plt.colorbar(label="Intensity (dB)")
        plt.savefig(OUTPUT_DIR / f"{patient_id}_spectrogram_{target_ch}.png")
        plt.close()
        logger.info(f"Saved spectrogram to {OUTPUT_DIR / f'{patient_id}_spectrogram_{target_ch}.png'}")
    else:
        logger.warning(f"{target_ch} not found in epochs. Skipping spectrogram.")


if __name__ == "__main__":
    visualize_patient("CON009")
