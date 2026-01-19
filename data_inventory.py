#!/usr/bin/env python3
"""
DAT-01: Data Inventory & Sync Script
=====================================
Script to verify local OneDrive files against expected data.
Logs missing files and generates a comprehensive inventory report.

Author: AwakenAI Capstone Team
Date: January 15, 2026
Task: DAT-01 (Due: Jan 15, 2026)
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import hashlib


class DataInventory:
    """Manages EEG data inventory and synchronization checks."""

    def __init__(self, data_root: str):
        """
        Initialize the data inventory manager.

        Parameters:
        -----------
        data_root : str
            Path to the root data directory
        """
        self.data_root = Path(data_root)
        self.inventory = {
            'edf_files': [],
            'csv_files': [],
            'wav_files': [],
            'other_files': []
        }
        self.missing_files = []
        self.expected_patients = []

    def scan_directory(self, verbose=True):
        """Scan the data directory and catalog all files."""
        if verbose:
            print(f"Scanning directory: {self.data_root}")
            print("=" * 80)

        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                filepath = Path(root) / file
                rel_path = filepath.relative_to(self.data_root)

                file_info = {
                    'filename': file,
                    'relative_path': str(rel_path),
                    'full_path': str(filepath),
                    'size_mb': filepath.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                }

                # Categorize by file type
                if file.lower().endswith('.edf'):
                    file_info['type'] = 'EDF'
                    self.inventory['edf_files'].append(file_info)
                elif file.lower().endswith('.csv'):
                    file_info['type'] = 'CSV'
                    self.inventory['csv_files'].append(file_info)
                elif file.lower().endswith(('.wav', '.mp3', '.m4a')):
                    file_info['type'] = 'AUDIO'
                    self.inventory['wav_files'].append(file_info)
                else:
                    file_info['type'] = 'OTHER'
                    self.inventory['other_files'].append(file_info)

        if verbose:
            self._print_summary()

    def _print_summary(self):
        """Print inventory summary."""
        print(f"\n📊 Inventory Summary")
        print("-" * 80)
        print(f"EDF Files (EEG Recordings): {len(self.inventory['edf_files'])}")
        print(f"CSV Files (Trial Logs):     {len(self.inventory['csv_files'])}")
        print(f"Audio Files (Stimuli):      {len(self.inventory['wav_files'])}")
        print(f"Other Files:                {len(self.inventory['other_files'])}")
        print(f"\nTotal Files:                {self._total_files()}")
        print(f"Total Size:                 {self._total_size():.2f} GB")
        print("=" * 80)

    def _total_files(self):
        """Calculate total number of files."""
        return sum(len(files) for files in self.inventory.values())

    def _total_size(self):
        """Calculate total size in GB."""
        total_mb = 0
        for category in self.inventory.values():
            total_mb += sum(f['size_mb'] for f in category)
        return total_mb / 1024

    def extract_patient_ids(self):
        """Extract unique patient IDs from filenames."""
        patient_ids = set()

        # Extract from EDF files
        for file_info in self.inventory['edf_files']:
            filename = file_info['filename']
            # Pattern: CON008, CON009, etc.
            if filename.startswith('CON'):
                patient_id = filename.split('_')[0].split('.')[0]
                patient_ids.add(patient_id)
            elif filename.startswith('TEST'):
                patient_ids.add('TEST')

        # Extract from CSV stimulus files
        for file_info in self.inventory['csv_files']:
            filename = file_info['filename']
            if 'stimulus_results' in filename and filename.startswith('CON'):
                patient_id = filename.split('_')[0]
                patient_ids.add(patient_id)

        self.expected_patients = sorted(list(patient_ids))
        return self.expected_patients

    def match_files(self):
        """Match EDF files with their corresponding CSV stimulus logs."""
        matches = []
        unmatched_edf = []
        unmatched_csv = []

        # Create lookup dictionaries
        edf_by_patient = {}
        for edf in self.inventory['edf_files']:
            filename = edf['filename']
            patient_id = filename.split('_')[0].split('.')[0]
            if patient_id not in edf_by_patient:
                edf_by_patient[patient_id] = []
            edf_by_patient[patient_id].append(edf)

        csv_by_patient = {}
        for csv in self.inventory['csv_files']:
            if 'stimulus_results' in csv['filename']:
                patient_id = csv['filename'].split('_')[0]
                if patient_id not in csv_by_patient:
                    csv_by_patient[patient_id] = []
                csv_by_patient[patient_id].append(csv)

        # Match files
        all_patients = set(edf_by_patient.keys()) | set(csv_by_patient.keys())

        for patient_id in sorted(all_patients):
            edf_files = edf_by_patient.get(patient_id, [])
            csv_files = csv_by_patient.get(patient_id, [])

            match_info = {
                'patient_id': patient_id,
                'edf_files': [f['filename'] for f in edf_files],
                'csv_files': [f['filename'] for f in csv_files],
                'has_edf': len(edf_files) > 0,
                'has_csv': len(csv_files) > 0,
                'status': 'COMPLETE' if (len(edf_files) > 0 and len(csv_files) > 0) else 'INCOMPLETE'
            }

            matches.append(match_info)

            if len(edf_files) > 0 and len(csv_files) == 0:
                unmatched_edf.extend(edf_files)
            elif len(csv_files) > 0 and len(edf_files) == 0:
                unmatched_csv.extend(csv_files)

        return matches, unmatched_edf, unmatched_csv

    def check_missing_files(self):
        """Check for commonly missing files mentioned in documentation."""
        missing = []

        # Check for specific missing files mentioned in docs
        expected_missing = [
            'lang28.wav',
            'CON006.EDF'  # Mentioned as potentially problematic
        ]

        all_filenames = set()
        for category in self.inventory.values():
            for file_info in category:
                all_filenames.add(file_info['filename'])

        for expected_file in expected_missing:
            if expected_file not in all_filenames:
                missing.append({
                    'filename': expected_file,
                    'type': 'EXPECTED_MISSING',
                    'note': 'Mentioned in documentation as potentially missing'
                })

        # Check if we have WAV files at all
        if len(self.inventory['wav_files']) == 0:
            missing.append({
                'filename': 'Audio stimuli directory',
                'type': 'CRITICAL_MISSING',
                'note': 'No audio stimulus files (.wav, .mp3) found in entire dataset'
            })

        self.missing_files = missing
        return missing

    def generate_report(self, output_dir: str = None):
        """Generate comprehensive inventory report."""
        if output_dir is None:
            # Default to reports/ directory in the same directory as this script
            script_dir = Path(__file__).parent
            output_dir = script_dir / 'reports'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Save detailed inventory as JSON
        json_path = output_dir / f'data_inventory_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.inventory, f, indent=2)

        # 2. Save file matching report
        matches, unmatched_edf, unmatched_csv = self.match_files()
        match_df = pd.DataFrame(matches)
        match_path = output_dir / f'file_matching_{timestamp}.csv'
        match_df.to_csv(match_path, index=False)

        # 3. Save missing files report
        if self.missing_files:
            missing_df = pd.DataFrame(self.missing_files)
            missing_path = output_dir / f'missing_files_{timestamp}.csv'
            missing_df.to_csv(missing_path, index=False)

        # 4. Generate human-readable markdown report
        md_path = output_dir / f'inventory_report_{timestamp}.md'
        self._generate_markdown_report(md_path, matches, unmatched_edf, unmatched_csv)

        print(f"\n✅ Reports generated in: {output_dir}")
        print(f"   - JSON inventory: {json_path.name}")
        print(f"   - File matching: {match_path.name}")
        if self.missing_files:
            print(f"   - Missing files: {missing_path.name}")
        print(f"   - Markdown report: {md_path.name}")

        return str(md_path)

    def _generate_markdown_report(self, output_path, matches, unmatched_edf, unmatched_csv):
        """Generate human-readable markdown report."""
        with open(output_path, 'w') as f:
            f.write("# EEG Data Inventory Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Data Root:** `{self.data_root}`\n\n")
            f.write("---\n\n")

            # Summary
            f.write("## 📊 Inventory Summary\n\n")
            f.write(f"- **EDF Files:** {len(self.inventory['edf_files'])}\n")
            f.write(f"- **CSV Files:** {len(self.inventory['csv_files'])}\n")
            f.write(f"- **Audio Files:** {len(self.inventory['wav_files'])}\n")
            f.write(f"- **Other Files:** {len(self.inventory['other_files'])}\n")
            f.write(f"- **Total Size:** {self._total_size():.2f} GB\n\n")

            # Patient IDs
            patient_ids = self.extract_patient_ids()
            f.write(f"## 👥 Identified Patients ({len(patient_ids)})\n\n")
            f.write(", ".join(patient_ids) + "\n\n")

            # File Matching
            f.write("## 🔗 File Matching Status\n\n")
            f.write("| Patient ID | EDF Files | CSV Files | Status |\n")
            f.write("|:-----------|:----------|:----------|:-------|\n")
            for match in matches:
                status_icon = "✅" if match['status'] == 'COMPLETE' else "⚠️"
                f.write(f"| {match['patient_id']} | {len(match['edf_files'])} | "
                       f"{len(match['csv_files'])} | {status_icon} {match['status']} |\n")
            f.write("\n")

            # Detailed file lists
            f.write("## 📁 Detailed File Lists\n\n")

            f.write("### EDF Files (EEG Recordings)\n\n")
            for edf in sorted(self.inventory['edf_files'], key=lambda x: x['filename']):
                f.write(f"- `{edf['relative_path']}` ({edf['size_mb']:.1f} MB)\n")
            f.write("\n")

            f.write("### CSV Files (Trial Logs)\n\n")
            csv_categories = {
                'stimulus_results': [],
                'patient_df': [],
                'patient_history': [],
                'patient_notes': [],
                'other': []
            }

            for csv in self.inventory['csv_files']:
                filename = csv['filename']
                if 'stimulus_results' in filename:
                    csv_categories['stimulus_results'].append(csv)
                elif 'patient_df' in filename:
                    csv_categories['patient_df'].append(csv)
                elif 'patient_history' in filename:
                    csv_categories['patient_history'].append(csv)
                elif 'patient_notes' in filename:
                    csv_categories['patient_notes'].append(csv)
                else:
                    csv_categories['other'].append(csv)

            for category, files in csv_categories.items():
                if files:
                    f.write(f"\n**{category.replace('_', ' ').title()}:**\n")
                    for csv in sorted(files, key=lambda x: x['filename']):
                        f.write(f"- `{csv['relative_path']}` ({csv['size_mb']:.2f} MB)\n")
            f.write("\n")

            # Missing Files
            if self.missing_files:
                f.write("## ⚠️ Missing Files\n\n")
                for missing in self.missing_files:
                    f.write(f"- **{missing['filename']}** ({missing['type']})\n")
                    f.write(f"  - Note: {missing['note']}\n")
                f.write("\n")

            # Unmatched Files
            if unmatched_edf or unmatched_csv:
                f.write("## ❓ Unmatched Files\n\n")
                if unmatched_edf:
                    f.write("### EDF files without corresponding CSV logs:\n")
                    for edf in unmatched_edf:
                        f.write(f"- `{edf['filename']}`\n")
                    f.write("\n")
                if unmatched_csv:
                    f.write("### CSV logs without corresponding EDF files:\n")
                    for csv in unmatched_csv:
                        f.write(f"- `{csv['filename']}`\n")
                    f.write("\n")

            # Recommendations
            f.write("## 🎯 Next Steps\n\n")
            f.write("1. **Verify Missing Audio Files:** No WAV/MP3 stimulus files found. ")
            f.write("Check if they're in a separate location on OneDrive.\n")
            f.write("2. **CSV Schema Unification (DAT-03):** Multiple `patient_df` variants ")
            f.write("need harmonization.\n")
            f.write("3. **Validate File Completeness:** Ensure all patients have both EDF and CSV files.\n")
            f.write("4. **Create Master File List:** Use this inventory as the reference for ")
            f.write("pipeline input validation.\n")


def main():
    """Main execution function."""
    print("=" * 80)
    print("EEG Data Inventory Script (DAT-01)")
    print("AwakenAI Capstone Project")
    print("=" * 80)
    print()

    # Determine data root path
    script_dir = Path(__file__).parent
    data_root = script_dir.parent.parent / 'Data' / 'extracted'

    if not data_root.exists():
        print(f"❌ Error: Data directory not found at {data_root}")
        print("   Please update the data_root path in the script.")
        return

    # Run inventory
    inventory = DataInventory(str(data_root))
    inventory.scan_directory(verbose=True)

    # Extract patient IDs
    patient_ids = inventory.extract_patient_ids()
    print(f"\n👥 Identified Patients: {', '.join(patient_ids)}")

    # Check for missing files
    print(f"\n🔍 Checking for missing files...")
    missing = inventory.check_missing_files()
    if missing:
        print(f"\n⚠️  Missing Files Detected ({len(missing)}):")
        for item in missing:
            print(f"   - {item['filename']} ({item['type']})")
            print(f"     Note: {item['note']}")
    else:
        print("   ✅ No known missing files detected")

    # File matching
    print(f"\n🔗 Matching EDF files with CSV logs...")
    matches, unmatched_edf, unmatched_csv = inventory.match_files()

    complete = sum(1 for m in matches if m['status'] == 'COMPLETE')
    incomplete = len(matches) - complete

    print(f"   ✅ Complete pairs: {complete}")
    print(f"   ⚠️  Incomplete pairs: {incomplete}")

    if unmatched_edf:
        print(f"\n   EDF files without CSV: {len(unmatched_edf)}")
    if unmatched_csv:
        print(f"   CSV files without EDF: {len(unmatched_csv)}")

    # Generate reports
    print(f"\n📝 Generating inventory reports...")
    report_path = inventory.generate_report()

    print(f"\n✅ DAT-01 Complete!")
    print(f"\n📄 View the full report at:")
    print(f"   {report_path}")


if __name__ == '__main__':
    main()
