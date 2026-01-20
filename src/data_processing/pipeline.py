import pandas as pd
import numpy as np
from pathlib import Path
from .normalization import normalize_trial_type, normalize_sentences

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
            
            # Pre-process lang_XX to extract sentence info if sentences is empty (vectorized)
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
                df['sentences'] = np.empty((len(df), 0)).tolist()

            # Add provenance tracking - source file name
            df['source_file'] = f.name

            # Ensure minimal columns exist
            required_cols = ['patient_id', 'date', 'trial_type', 'sentences', 'start_time', 'end_time', 'duration', 'source_file']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None

            # Select only standard columns
            dfs.append(df[required_cols])
            print(f"Processed {f.name}: {len(df)} rows")
            
        except Exception as e:
            print(f"Skipping {f.name} due to error: {e}")

    if dfs:
        unified_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal Unified Rows: {len(unified_df)}")
        
        # Save to Parquet
        unified_df.to_parquet(output_file, engine='pyarrow')
        print(f"Successfully saved to {output_file}")
        return unified_df
    else:
        print("No data found to unify.")
        return None
