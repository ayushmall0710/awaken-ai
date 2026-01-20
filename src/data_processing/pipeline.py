import pandas as pd
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
            
            # Pre-process lang_XX to extract sentence info if sentences is empty
            if 'trial_type' in df.columns and 'sentences' in df.columns:
                def extract_lang_info(row):
                    tt = str(row['trial_type']).lower().strip()
                    sentences = row['sentences']
                    
                    # Logic: If trial_type is lang_XX and sentences is empty/NaN,
                    # use XX as the event.
                    if tt.startswith('lang_') and tt[5:].isdigit():
                        # check if sentences is empty (NaN, empty string, or empty list)
                        is_empty = pd.isna(sentences) or sentences == "" or sentences == "[]"
                        if is_empty:
                            # Return logic that normalize_sentences will understand, or direct list[dict]
                            # normalize_sentences handles strings/ints well. 
                            return str(tt[5:]) # Return "11"
                    
                    return sentences

                # Apply this ONLY to rows where we need it is hard in apply, so applied to all
                # But need access to both columns.
                # using apply(axis=1) is slower but safe.
                df['sentences'] = df.apply(extract_lang_info, axis=1)

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
