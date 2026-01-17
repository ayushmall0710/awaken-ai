"""
Script to digitize patient notes and history into structured JSON/Pandas format.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _clean_patient_dataframe(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Clean and standardize patient dataframes."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(how="all")
    combined = combined[
        (combined["patient_id"].notna())
        & (combined["patient_id"] != "")
        & (combined["patient_id"].astype(str).str.strip() != "nan")
    ]
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.drop_duplicates().sort_values(["patient_id", "date"])

    return combined.reset_index(drop=True)


def _load_patient_files(
    data_dir: Path, pattern: str, default_columns: List[str]
) -> pd.DataFrame:
    """Generic loader for patient CSV files."""
    files = list(data_dir.glob(pattern))
    logger.info(f"Found {len(files)} {pattern} files")

    dfs = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")

    if not dfs:
        return pd.DataFrame(columns=default_columns)

    return _clean_patient_dataframe(dfs)


def load_patient_notes(data_dir: Path) -> pd.DataFrame:
    """Load and combine all patient notes files."""
    return _load_patient_files(
        data_dir, "patient_notes*.csv", ["patient_id", "notes", "date"]
    )


def load_patient_history(data_dir: Path) -> pd.DataFrame:
    """Load and combine all patient history files."""
    return _load_patient_files(data_dir, "patient_history*.csv", ["patient_id", "date"])


def create_patient_records_structure(
    notes_df: pd.DataFrame, history_df: pd.DataFrame
) -> Dict:
    """Create a structured dictionary of patient records."""
    patient_records = {}

    # Get all unique patients
    all_patients = set(notes_df["patient_id"].unique()) | set(
        history_df["patient_id"].unique()
    )

    for patient_id in all_patients:
        patient_notes = notes_df[notes_df["patient_id"] == patient_id].copy()
        patient_history = history_df[history_df["patient_id"] == patient_id].copy()

        # Collect dates before conversion for first/last visit calculation
        all_dates = []
        if not patient_notes.empty:
            all_dates.extend(patient_notes["date"].dropna().tolist())
        if not patient_history.empty:
            all_dates.extend(patient_history["date"].dropna().tolist())

        if not patient_notes.empty:
            patient_notes_formatted = patient_notes.copy()
            patient_notes_formatted["date"] = patient_notes_formatted[
                "date"
            ].dt.strftime("%Y-%m-%d")
            patient_notes_formatted["notes"] = (
                patient_notes_formatted["notes"].fillna("").astype(str)
            )
            notes_list = patient_notes_formatted[["date", "notes"]].to_dict("records")
        else:
            notes_list = []

        if not patient_history.empty:
            patient_history_formatted = patient_history.copy()
            patient_history_formatted["date"] = patient_history_formatted[
                "date"
            ].dt.strftime("%Y-%m-%d")
            history_list = patient_history_formatted[["date"]].to_dict("records")
        else:
            history_list = []

        first_visit = min(all_dates).strftime("%Y-%m-%d") if all_dates else None
        last_visit = max(all_dates).strftime("%Y-%m-%d") if all_dates else None

        patient_records[patient_id] = {
            "patient_id": patient_id,
            "first_visit": first_visit,
            "last_visit": last_visit,
            "total_visits": len(history_list),
            "total_notes": len(notes_list),
            "notes": notes_list,
            "visit_history": history_list,
        }

    return patient_records


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data" / "EEG"
    output_dir = project_root / "data" / "processed"

    notes_df = load_patient_notes(data_dir)
    history_df = load_patient_history(data_dir)
    patient_records = create_patient_records_structure(notes_df, history_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "patient_records.json"
    with open(json_output, "w") as f:
        json.dump(patient_records, f, indent=2)
    logger.info(f"Saved JSON to: {json_output}")


if __name__ == "__main__":
    main()
