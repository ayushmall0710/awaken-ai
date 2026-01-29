import pandas as pd
from pathlib import Path
from .normalization import normalize_trial_type, normalize_sentences

# Standard columns for unified stimulus data
REQUIRED_COLS = ['patient_id', 'date', 'trial_type', 'sentences',
                 'start_time', 'end_time', 'duration', 'source_file']


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
    if 'trial_type' in df.columns and 'sentences' in df.columns:
        tt_lower = df['trial_type'].astype(str).str.lower().str.strip()
        is_lang_pattern = tt_lower.str.match(r'^lang_\d+$')
        is_empty_sentences = df['sentences'].isna() | (df['sentences'] == "") | (df['sentences'] == "[]")
        mask = is_lang_pattern & is_empty_sentences
        df.loc[mask, 'sentences'] = tt_lower[mask].str.extract(r'^lang_(\d+)$')[0]

    # Normalize trial_type
    if 'trial_type' in df.columns:
        df['trial_type'] = df['trial_type'].apply(normalize_trial_type)
    else:
        df['trial_type'] = 'unknown'

    # Normalize sentences
    if 'sentences' in df.columns:
        df['sentences'] = df['sentences'].apply(normalize_sentences)
    else:
        df['sentences'] = [[] for _ in range(len(df))]

    # Add provenance
    df['source_file'] = source_name

    # Reindex to standard columns
    return df.reindex(columns=REQUIRED_COLS)


def unify_stimulus_data(data_dir: Path, output_file: Path):
    """
    Reads all stimulus CSVs from data_dir, normalizes them, and saves to output_file (Parquet).
    """
    stimulus_files = list(data_dir.glob("*stimulus_results.csv"))
    patient_dfs = list(data_dir.glob("patient_df*.csv"))
    all_files = stimulus_files + patient_dfs

    dfs = []
    print(f"Found {len(all_files)} files to unify from {data_dir}")

    for f in all_files:
        try:
            # Skip the unified csv/parquet if it exists in the list to avoid recursion
            if 'unified' in f.name:
                continue

            df = pd.read_csv(f)
            processed_df = process_stimulus_df(df, source_name=f.name)
            dfs.append(processed_df)
            print(f"Processed {f.name}: {len(df)} rows")

        except Exception as e:
            print(f"Skipping {f.name} due to error: {e}")

    if dfs:
        unified_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal Unified Rows: {len(unified_df)}")

        # Deduplicate based on full row (excluding source_file, convert sentences to string for comparison)
        initial_count = len(unified_df)
        unified_df['_sentences_str'] = unified_df['sentences'].astype(str)
        unified_df = unified_df.drop_duplicates(subset=['patient_id', 'date', 'trial_type', '_sentences_str',
                                                         'start_time', 'end_time', 'duration'],
                                                 keep='first')
        unified_df = unified_df.drop(columns=['_sentences_str'])
        duplicates_removed = initial_count - len(unified_df)
        print(f"Removed {duplicates_removed} duplicate rows")
        print(f"Final Row Count: {len(unified_df)}")

        # Save to Parquet
        unified_df.to_parquet(output_file, engine='pyarrow')
        print(f"Successfully saved to {output_file}")
        return unified_df
    else:
        print("No data found to unify.")
        return None
