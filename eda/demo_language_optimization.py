"""
Demo script for Language Optimization Pipeline (ENG-05).
Run this to verify the pipeline on actual patient data (e.g., CON008).
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.language_optimization import LanguageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_demo(patient_id="CON008"):
    logger.info(f"--- Running Language Optimization Demo for {patient_id} ---")

    processor = LanguageProcessor()

    # 1. Process with Left Hemisphere Focus
    logger.info("1. Processing with LH Focus...")

    # Try CON009 as well if CON008 fails or just run both
    for pid in [patient_id, "CON009"]:
        logger.info(f"\n--- Processing {pid} ---")
        try:
            epochs_lh = processor.process_patient(pid, focus="LH")

            if epochs_lh:
                logger.info(f"SUCCESS: Loaded {len(epochs_lh)} epochs for {pid}.")
                logger.info(f"Channels ({len(epochs_lh.ch_names)}): {epochs_lh.ch_names}")
            else:
                logger.error(f"FAILED: No epochs returned for {pid} LH focus.")
        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")


if __name__ == "__main__":
    run_demo()
