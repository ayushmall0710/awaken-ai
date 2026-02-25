"""
Demo script for Language Optimization Pipeline (ENG-05).
Run this to verify the pipeline on actual patient data (e.g., CON008).
"""

import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.language_optimization import LanguageProcessor
from src.data_processing.timestamp_aligner import TimestampAligner

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_demo(patient_id="CON008"):
    logger.info(f"--- Running Language Optimization Demo for {patient_id} ---")

    try:
        # 1. Align Timestamps (REQUIRED)
        logger.info("1. Aligning Timestamps...")
        aligner = TimestampAligner(patient_id=patient_id)
        # align() returns a dict {patient_id: events_df}
        aligned_events = aligner.align(save=False)[patient_id]
        logger.info(f"   Aligned {len(aligned_events)} events.")

        # 2. Process with Language Processor
        logger.info("2. Processing with LH Focus...")
        processor = LanguageProcessor()

        epochs_lh = processor.process_patient(
            patient_id,
            aligned_events=aligned_events,  # Pass aligned events
            focus="LH",
        )

        if epochs_lh:
            logger.info(f"SUCCESS: Loaded {len(epochs_lh)} epochs for {patient_id}.")
            logger.info(f"Channels ({len(epochs_lh.ch_names)}): {epochs_lh.ch_names}")
        else:
            logger.error(f"FAILED: No epochs returned for {patient_id} LH focus.")

    except Exception as e:
        logger.error(f"Error processing {patient_id}: {e}")


if __name__ == "__main__":
    run_demo("CON008")
