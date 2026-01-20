import sys
from pathlib import Path

# Add project root to path so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing.pipeline import unify_stimulus_data
from src.data_loading.config import LOCAL_DATA_ROOT

def main():
    # Define paths using config
    # Start from extracting/EEG as seen in previous scripts
    # config.py defines LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
    # Previous run_analysis used: .../Data/extracted/EEG
    # We should ensure consistency with where the data actually resides
    
    # Check if we should use absolute path from before or try to use config
    # In run_analysis.py: Path("/Users/ayush/Desktop/Capstone Project/Data/extracted/EEG")
    # This seems to be outside the Repo/data folder? 
    # Let's double check. 
    # run_analysis.py: DATA_DIR = Path("/Users/ayush/Desktop/Capstone Project/Data/extracted/EEG")
    
    # We will use the hardcoded path for now to be safe, as moving data is out of scope 
    # unless confirmed.
    DATA_DIR = Path("data/EEG")
    OUTPUT_FILE = DATA_DIR / "unified_stimulus_results.parquet"
    
    unify_stimulus_data(DATA_DIR, OUTPUT_FILE)

if __name__ == "__main__":
    main()
