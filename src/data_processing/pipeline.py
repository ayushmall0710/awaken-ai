from pathlib import Path

import pandas as pd

from src.data_processing.normalization import normalize_sentences, normalize_trial_type

# Trial type -> short prefix for trial_id generation
TRIAL_TYPE_PREFIX = {
    "language": "lt",
    "oddball": "obt",
    "left_command": "lct",
    "right_command": "rct",
    "beep": "bt",
    "control": "ct",
    "loved_one_voice": "lovt",
}

# Standard columns for unified stimulus data
REQUIRED_COLS = [
    "patient_id",
    "session_id",
    "date",
    "trial_type",
    "trial_id",
    "sentences",
    "start_time",
    "end_time",
    "duration",
    "source_file",
]


def generate_session_ids(df: pd.DataFrame) -> pd.Series:
    """Generate session_id from patient_id + date.
    Added HHMM to the date in case there are multiple sessions in a day.

    Format: s_{patient_id}_{%Y%m%d%H%M}
    Example: s_CON008_202501101430
    """
    date_str = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d%H%M")
    fallback = df["date"].astype(str).str.replace(r"[^\w]", "", regex=True)
    date_str = date_str.fillna(fallback)

    return "s_" + df["patient_id"] + "_" + date_str


def generate_trial_ids(df: pd.DataFrame) -> pd.Series:
    """Generate trial_id per (session_id, trial_type) sorted by start_time.

    Format: {prefix}{sequential_index}  (e.g. lt1, obt2, lct3)
    Falls back to 'unk' prefix for unmapped trial types.
    """
    df = df.sort_values("start_time")
    prefixes = df["trial_type"].map(TRIAL_TYPE_PREFIX).fillna("unk")
    seq = df.groupby(["session_id", "trial_type"]).cumcount() + 1
    return prefixes + seq.astype(str)


def process_stimulus_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Process a single stimulus DataFrame - normalize trial types and sentences.
    This is the core transformation logic, separated from file I/O.

    Args:
        df: Raw stimulus DataFrame
        source_name: Name to use for source_file column (e.g., filename)

    Returns:
        Processed DataFrame with normalized columns
    """
    df = df.copy()

    # Lang_XX rescue logic - extract event ID from trial_type when sentences is empty
    if "trial_type" in df.columns and "sentences" in df.columns:
        tt_lower = df["trial_type"].astype(str).str.lower().str.strip()
        is_lang_pattern = tt_lower.str.match(r"^lang_\d+$")
        is_empty_sentences = df["sentences"].isna() | (df["sentences"] == "") | (df["sentences"] == "[]")
        mask = is_lang_pattern & is_empty_sentences
        df.loc[mask, "sentences"] = tt_lower[mask].str.extract(r"^lang_(\d+)$")[0]

    # Normalize trial_type
    if "trial_type" in df.columns:
        df["trial_type"] = df["trial_type"].apply(normalize_trial_type)
    else:
        df["trial_type"] = "unknown"

    # Normalize sentences
    if "sentences" in df.columns:
        df["sentences"] = df["sentences"].apply(normalize_sentences)
    else:
        df["sentences"] = [[] for _ in range(len(df))]

    # Add provenance
    df["source_file"] = source_name

    # Reindex to standard columns (session_id and trial_id added later in unify_stimulus_data)
    base_cols = [c for c in REQUIRED_COLS if c not in ("session_id", "trial_id")]
    return df.reindex(columns=base_cols)


def unify_stimulus_data(data_dir: Path, output_file: Path, verbose: bool = False):
    """
    Reads all stimulus CSVs from data_dir, normalizes them, and saves to output_file (Parquet).
    Generates session_id and trial_id after deduplication.
    """
    stimulus_files = list(data_dir.glob("*stimulus_results.csv"))
    patient_dfs = list(data_dir.glob("patient_df*.csv"))
    all_files = stimulus_files + patient_dfs

    dfs = []
    if verbose:
        print(f"Found {len(all_files)} files to unify from {data_dir}")

    for f in all_files:
        try:
            # Skip the unified csv/parquet if it exists in the list to avoid recursion
            if "unified" in f.name:
                continue

            df = pd.read_csv(f)
            processed_df = process_stimulus_df(df, source_name=f.name)
            dfs.append(processed_df)
            if verbose:
                print(f"Processed {f.name}: {len(df)} rows")

        except Exception as e:
            if verbose:
                print(f"Skipping {f.name} due to error: {e}")

    if dfs:
        unified_df = pd.concat(dfs, ignore_index=True)
        if verbose:
            print(f"\nTotal Unified Rows: {len(unified_df)}")

        # Deduplicate based on full row (excluding source_file, convert sentences to string for comparison)
        initial_count = len(unified_df)
        unified_df["_sentences_str"] = unified_df["sentences"].astype(str)
        unified_df = unified_df.drop_duplicates(
            subset=[
                "patient_id",
                "date",
                "trial_type",
                "_sentences_str",
                "start_time",
                "end_time",
                "duration",
            ],
            keep="first",
        )
        unified_df = unified_df.drop(columns=["_sentences_str"])
        duplicates_removed = initial_count - len(unified_df)
        if verbose:
            print(f"Removed {duplicates_removed} duplicate rows")

        # Generate IDs (post-dedup)
        unified_df["session_id"] = generate_session_ids(unified_df)
        unified_df["trial_id"] = generate_trial_ids(unified_df)

        # Reorder to standard column order
        unified_df = unified_df.reindex(columns=REQUIRED_COLS)

        if verbose:
            print(f"Final Row Count: {len(unified_df)}")

        # Save to Parquet
        unified_df.to_parquet(output_file, engine="pyarrow")
        if verbose:
            print(f"Successfully saved to {output_file}")
        return unified_df
    else:
        if verbose:
            print("No data found to unify.")
        return None
