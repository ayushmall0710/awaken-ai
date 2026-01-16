"""
Data Inventory & Sync Script
Verifies and syncs files from OneDrive to local data directory using pattern matching.
"""

import argparse
import shutil
import logging
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from . import config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


class DataInventory:
    """Manages inventory and synchronization of EEG data files from OneDrive"""

    def __init__(self, onedrive_root: str, local_data_root: str):
        self.onedrive_root = Path(onedrive_root)
        self.local_data_root = Path(local_data_root)
        self.local_data_root.mkdir(parents=True, exist_ok=True)

    def _get_pattern_files(self, root: Path) -> Dict[str, Dict]:
        """Directly scan for files matching patterns"""
        files = {}
        patterns = [
            ("EEG/**/*stimulus_results.csv", "csv", "stimulus_log", True),
            ("EEG/*stimulus_results.csv", "csv", "stimulus_log", True),
            ("EEG/patient_df*.csv", "csv", "patient_metadata", False),
            ("EEG/patient_history*.csv", "csv", "patient_metadata", False),
            ("EEG/patient_notes*.csv", "csv", "patient_metadata", False),
            ("EEG/edf/**/*.EDF", "edf", "eeg_raw", False),
            ("EEG/edf/**/*.edf", "edf", "eeg_raw", False),
            ("Audio/sentences/lang*.wav", "audio", "language_stimulus", True),
            ("Audio/prompts/*.wav", "audio", "prompt", True),
            ("Audio/prompts/*.mp3", "audio", "prompt", True),
            ("Audio/static/*.mp3", "audio", "command_stimulus", True),
            ("Audio/static/*.wav", "audio", "command_stimulus", True),
            ("Audio/Voice/Trimmed/*.wav", "audio", "voice_stimulus", False),
            ("Audio/Voice/Raw/*.m4a", "audio", "voice_stimulus", False),
            ("Audio/Voice/Raw/*.aup3", "audio", "voice_stimulus", False),
        ]

        if not root.exists():
            return files

        for pattern, file_type, category, required in patterns:
            try:
                if "**" in pattern:
                    base_pattern = pattern.split("**")[0].rstrip("/")
                    file_pattern = pattern.split("**")[-1].lstrip("/")
                    search_root = root / base_pattern if base_pattern else root
                    found_files = (
                        list(search_root.rglob(file_pattern))
                        if search_root.exists()
                        else []
                    )
                else:
                    found_files = list(root.glob(pattern))

                for file_path in found_files:
                    if file_path.is_file():
                        try:
                            rel_path_str = str(file_path.relative_to(root)).replace(
                                "\\", "/"
                            )
                            if rel_path_str not in files:
                                files[rel_path_str] = {
                                    "type": file_type,
                                    "category": category,
                                    "required": required,
                                }
                        except ValueError:
                            continue
            except Exception as e:
                logger.warning(f"Error processing pattern {pattern}: {e}")

        return files

    def verify_files(self, check_onedrive: bool = True) -> Dict[str, List[str]]:
        """Directly scan for files matching patterns and verify."""
        root = self.onedrive_root if check_onedrive else self.local_data_root
        files = self._get_pattern_files(root)
        found = list(files.keys())
        missing_required = [
            path
            for path, info in files.items()
            if info.get("required", False) and not (root / path).exists()
        ]
        return {"found": found, "missing": [], "missing_required": missing_required}

    def get_missing_files(self) -> List[str]:
        """Get list of files that exist in OneDrive but are missing locally."""
        onedrive_files = self._get_pattern_files(self.onedrive_root)
        missing = []

        for file_path in onedrive_files.keys():
            source = self.onedrive_root / file_path
            dest = self.local_data_root / file_path
            if source.exists() and not dest.exists():
                missing.append(file_path)

        return missing

    def sync_files(
        self, files_to_sync: List[str], overwrite: bool = False
    ) -> Dict[str, int]:
        """Sync specified files from OneDrive to local."""
        stats = {"copied": 0, "skipped": 0, "missing": 0, "errors": 0}

        if not files_to_sync:
            logger.info("No files to sync")
            return stats

        with tqdm(total=len(files_to_sync), desc="Syncing files", unit="file") as pbar:
            for file_path in files_to_sync:
                source = self.onedrive_root / file_path
                dest = self.local_data_root / file_path

                if not source.exists():
                    stats["missing"] += 1
                    pbar.set_postfix({"status": "missing"})
                    pbar.update(1)
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)

                if dest.exists() and not overwrite:
                    stats["skipped"] += 1
                    pbar.set_postfix({"status": "skipped"})
                    pbar.update(1)
                    continue

                try:
                    shutil.copy2(source, dest)
                    stats["copied"] += 1
                    pbar.set_postfix({"status": "copied", "copied": stats["copied"]})
                    pbar.write(f"Copied: {file_path}")
                except Exception as e:
                    stats["errors"] += 1
                    pbar.set_postfix({"status": "error"})
                    pbar.write(f"Error copying {file_path}: {e}")

                pbar.update(1)

        logger.info(f"Sync: {stats['copied']} copied, {stats['skipped']} skipped")
        return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify and sync EEG data files from OneDrive"
    )
    parser.add_argument(
        "--onedrive-root",
        type=str,
        default=config.ONEDRIVE_ROOT,
        help=f"OneDrive root path (default: {config.ONEDRIVE_ROOT})",
    )
    parser.add_argument(
        "--local-data-root",
        type=str,
        default=str(config.LOCAL_DATA_ROOT),
        help=f"Local data root path (default: {config.LOCAL_DATA_ROOT})",
    )
    parser.add_argument(
        "--sync", action="store_true", help="Actually sync files (default: just verify)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()

    inventory = DataInventory(args.onedrive_root, args.local_data_root)
    missing_files = inventory.get_missing_files()

    onedrive_results = inventory.verify_files(check_onedrive=True)
    local_results = inventory.verify_files(check_onedrive=False)

    print(f"\n{'=' * 60}")
    print(f"OneDrive: {len(onedrive_results['found'])} files found")
    print(f"Local: {len(local_results['found'])} files found")
    print(f"{'=' * 60}")

    # Show missing files
    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        print("-" * 60)
        for i, file_path in enumerate(missing_files, 1):
            print(f"  {i}. {file_path}")
        print("-" * 60)
    else:
        print("\nAll files are synced!")

    # Ask for sync confirmation if --sync not provided
    should_sync = args.sync
    if missing_files and not args.sync:
        print(
            f"\nDo you want to sync {len(missing_files)} missing file(s)? (y/n): ",
            end="",
        )
        response = input().strip().lower()
        should_sync = response in ["y", "yes"]

    if should_sync:
        print("\nSyncing files...")
        sync_stats = inventory.sync_files(missing_files, overwrite=args.overwrite)
        print("\nSync complete!")
        print(f"\tCopied: {sync_stats['copied']}")
        print(f"\tSkipped: {sync_stats['skipped']}")
        print(f"\tErrors: {sync_stats['errors']}")
    elif missing_files:
        print("\nSync cancelled.")
