#!/usr/bin/env python3
"""
Run ERP/Oddball Pipeline (ENG-02b)

This script provides a command-line interface to the ERP pipeline for processing
oddball trials and extracting P300 features.

Usage Examples:
    # Process a single patient
    python scripts/run_erp_pipeline.py --patient CON008

    # Process all patients with oddball data
    python scripts/run_erp_pipeline.py --all

    # Process all patients and compute grand average
    python scripts/run_erp_pipeline.py --all --grand-average

    # Process single patient with specific session date
    python scripts/run_erp_pipeline.py --patient CON008 --date 2025-08-14

    # Custom output directory
    python scripts/run_erp_pipeline.py --all --output-dir /path/to/output

    # Verbose logging
    python scripts/run_erp_pipeline.py --patient CON008 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loading import config
from src.data_processing.erp_pipeline import OddballERPPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ERP/Oddball Pipeline (ENG-02b)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --patient CON008
  %(prog)s --all
  %(prog)s --all --grand-average
  %(prog)s --patient CON008 --date 2025-08-14
        """,
    )

    # Processing mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--patient", type=str, help="Process single patient by ID")
    mode.add_argument("--all", action="store_true", help="Process all patients with oddball data")
    mode.add_argument(
        "--list",
        action="store_true",
        help="List all patients with oddball data (no processing)",
    )
    mode.add_argument(
        "--list-electrodes",
        action="store_true",
        help="List available electrodes in EEG data and exit",
    )

    # Optional arguments
    parser.add_argument("--date", type=str, help="Specific session date (YYYY-MM-DD) for single patient")
    parser.add_argument(
        "--electrodes",
        type=str,
        default=None,
        help='Comma-separated electrodes to analyze (e.g., "Oz,T7,T8"). '
        "When set, this replaces the default Pz/Cz/Fz set.",
    )
    parser.add_argument(
        "--grand-average",
        action="store_true",
        help="Compute grand average ERP after processing (only with --all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help=f"Output directory (default: {config.PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.LOCAL_DATA_ROOT,
        help=f"Data root directory (default: {config.LOCAL_DATA_ROOT})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-qc", action="store_true", help="Skip QC report generation")

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("src.data_processing.erp_pipeline").setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ERP/Oddball Pipeline (ENG-02b)")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = OddballERPPipeline(data_root=args.data_root, output_dir=args.output_dir, verbose=args.verbose)

    try:
        # List mode
        if args.list:
            patient_ids = pipeline._get_patients_with_oddball()
            print(f"\nFound {len(patient_ids)} patients with oddball data:")
            for patient_id in patient_ids:
                print(f"  - {patient_id}")
            return 0

        # List electrodes mode
        if args.list_electrodes:
            from src.data_loading.unified_data_loader import UnifiedDataLoader

            loader = UnifiedDataLoader()

            # Find one patient with oddball data
            patient_ids = pipeline._get_patients_with_oddball()
            if not patient_ids:
                print("\nNo patients with oddball data found")
                return 1

            patient_id = patient_ids[0]

            # Load one session and print channel names
            aligned_trials = loader.load_aligned_trials(task="oddball")
            patient_trials = aligned_trials[aligned_trials["patient_id"] == patient_id]

            if patient_trials.empty:
                print(f"\nNo oddball data found for {patient_id}")
                return 1

            session_date = patient_trials.iloc[0]["date"]
            raw = loader.load_eeg(patient_id, session_date)

            print("\n" + "=" * 70)
            print(f"Available electrodes in {patient_id} ({session_date}):")
            print("=" * 70)
            for i, ch in enumerate(raw.ch_names, 1):
                print(f"{i:2d}. {ch}")
            print("=" * 70)
            print(f"Total: {len(raw.ch_names)} electrodes\n")
            print("Use these electrode names with --electrodes flag")
            print('Example: --electrodes "Pz,Cz,Fz"')
            print("=" * 70)
            return 0

        # Single patient mode
        if args.patient:
            logger.info(f"Processing patient: {args.patient}")
            if args.date:
                logger.info(f"Session date: {args.date}")

            # Parse custom electrodes
            custom_electrodes = None
            if args.electrodes:
                custom_electrodes = [e.strip() for e in args.electrodes.split(",")]
                logger.info(f"Using custom electrodes: {custom_electrodes}")

            result = pipeline.process_patient(args.patient, date=args.date, custom_electrodes=custom_electrodes)

            if result.get("status") == "success":
                features = result.get("features")
                if isinstance(features, dict):
                    print(f"\n✓ Successfully processed {args.patient}")
                    print(f"  Epochs: {features['n_epochs']}")
                    print(f"  P300 Amplitude: {features['p300_amplitude_uV']:.2f} µV")
                    print(f"  P300 Latency: {features['p300_latency_ms']:.1f} ms")
                else:
                    print(f"\n✓ Successfully processed {args.patient}")
                    print(f"  Total sessions: {result.get('sessions', 1)}")
                    print(f"  Total epochs: {features['n_epochs'].sum()}")

                logger.info(f"Outputs saved to: {args.output_dir}")
                return 0
            else:
                logger.error(f"Failed to process {args.patient}: {result.get('status')}")
                return 1

        # All patients mode
        if args.all:
            logger.info("Processing all patients with oddball data")

            # Parse custom electrodes
            custom_electrodes = None
            if args.electrodes:
                custom_electrodes = [e.strip() for e in args.electrodes.split(",")]
                logger.info(f"Using custom electrodes: {custom_electrodes}")

            features_df = pipeline.process_all_patients(custom_electrodes=custom_electrodes)

            if features_df.empty:
                logger.warning("No features extracted from any patient")
                return 1

            print(f"\n{'=' * 60}")
            print("Batch Processing Summary")
            print(f"{'=' * 60}")
            print(f"  Patients processed: {features_df['patient_id'].nunique()}")
            print(f"  Total sessions: {len(features_df)}")
            print(f"  Total epochs: {features_df['n_epochs'].sum()}")
            print(f"  Mean P300 amplitude: {features_df['p300_amplitude_uV'].mean():.2f} µV")
            print(f"  Mean P300 latency: {features_df['p300_latency_ms'].mean():.1f} ms")
            print(f"{'=' * 60}\n")

            # Compute grand average if requested
            if args.grand_average:
                logger.info("Computing grand average ERP")
                grand_avg = pipeline.compute_grand_average()

                if grand_avg is not None:
                    print("✓ Grand average ERP computed and saved")
                else:
                    logger.warning("Failed to compute grand average")

            # Generate QC report unless skipped
            if not args.skip_qc:
                logger.info("Generating QC report")
                qc_report = pipeline.generate_qc_report()

                if qc_report.get("status") != "no_data":
                    print("\n✓ QC report generated")

            logger.info(f"All outputs saved to: {args.output_dir}")
            return 0

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
