import sys
from pathlib import Path

# Add project root to path so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_processing.pipeline import unify_stimulus_data  # noqa: E402


def main():
    DATA_DIR = Path("data/EEG")
    OUTPUT_FILE = DATA_DIR / "unified_stimulus_results.parquet"

    unify_stimulus_data(DATA_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
