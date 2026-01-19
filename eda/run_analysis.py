import pandas as pd
from pathlib import Path
import json
import ast
import sys

# Set display options to see full content
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Define Data Directory
DATA_DIR = Path("/Users/ayush/Desktop/Capstone Project/Data/extracted/EEG")

# Find files
stimulus_files = list(DATA_DIR.glob("*stimulus_results.csv"))
patient_dfs = list(DATA_DIR.glob("patient_df*.csv"))

all_files = stimulus_files + patient_dfs
print(f"Found {len(stimulus_files)} stimulus_results files")
print(f"Found {len(patient_dfs)} patient_df files")
for f in all_files:
    print(f" - {f.name}")

def analyze_sentences_column(series):
    """Analyzes the structure and content of the sentences column."""
    if series.dropna().empty:
        return "Empty"
    
    sample = series.dropna().iloc[0]
    result = {
        "sample_raw": sample,
        "type_raw": str(type(sample)),
        "parsed_successfully": False,
        "parsed_type": None,
        "content_structure": None
    }
    
    parsed = None
    if isinstance(sample, str):
        # Try JSON
        try:
            parsed = json.loads(sample)
            result["parsed_successfully"] = True
            result["parsed_type"] = "JSON -> " + str(type(parsed))
        except json.JSONDecodeError:
            # Try AST (for Python list/dict string representation)
            try:
                parsed = ast.literal_eval(sample)
                result["parsed_successfully"] = True
                result["parsed_type"] = "AST -> " + str(type(parsed))
            except (ValueError, SyntaxError):
                result["parsed_type"] = "String (Unparseable)"
    else:
        # Already an object (unlikely if read from CSV without converters, but possible)
        parsed = sample
        result["parsed_successfully"] = True
        result["parsed_type"] = str(type(parsed))
        
    if parsed is not None:
        if isinstance(parsed, list):
            if len(parsed) > 0:
                result["content_structure"] = f"List of {type(parsed[0])}"
                # Check if it contains indices or more info
                result["first_item"] = str(parsed[0])
            else:
                result["content_structure"] = "Empty List"
        elif isinstance(parsed, dict):
            result["content_structure"] = f"Dict with keys: {list(parsed.keys())}"
        else:
            result["content_structure"] = "Scalar/Other"
            
    return result

def analyze_file(filepath):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {filepath.name}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dimensions: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # 1. Trial Type Analysis
        if 'trial_type' in df.columns:
            unique_types = df['trial_type'].unique()
            print(f"\n[trial_type] Unique Values ({len(unique_types)}):")
            print(unique_types)
        else:
            print("\n[trial_type] Column WARNING: Not found")
            
        # 2. Sentences Analysis
        if 'sentences' in df.columns:
            print(f"\n[sentences] Analysis:")
            analysis = analyze_sentences_column(df['sentences'])
            for k, v in analysis.items():
                print(f"  {k}: {v}")
        else:
             print("\n[sentences] Column WARNING: Not found")

    except Exception as e:
        print(f"ERROR reading file: {e}")

# Run Analysis on All Files
for f in all_files:
    analyze_file(f)

# Analyze Unified Parquet if exists
parquet_file = DATA_DIR / "unified_stimulus_results.parquet"
if parquet_file.exists():
    print(f"\n{'='*80}")
    print(f"ANALYZING UNIFIED PARQUET: {parquet_file.name}")
    print(f"{'='*80}")
    df_p = pd.read_parquet(parquet_file)
    print(f"Dimensions: {df_p.shape}")
    print(f"\n[trial_type] Unique Values:")
    print(df_p['trial_type'].unique())
    print(f"\n[sentences] Sample (index 0):")
    print(df_p['sentences'].iloc[0])
    print(f"Type: {type(df_p['sentences'].iloc[0])}")

