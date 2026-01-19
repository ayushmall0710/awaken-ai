
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

class TestDataIntegrity(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Define path relative to repository root or absolute
        # Assuming run from repo root
        cls.parquet_path = Path('Data/extracted/EEG/unified_stimulus_results.parquet').resolve()
        
        # Fallback to absolute path if running from tests dir or elsewhere if needed
        # But standard updates usually run from repo root.
        # Let's try to be robust:
        if not cls.parquet_path.exists():
             # Try one level up if in tests/
             cls.parquet_path = Path('../Data/extracted/EEG/unified_stimulus_results.parquet').resolve()
        
        if not cls.parquet_path.exists():
            # Hard fallback to known system path for this user context
            cls.parquet_path = Path('/Users/ayush/Desktop/Capstone Project/Data/extracted/EEG/unified_stimulus_results.parquet')

        if not cls.parquet_path.exists():
             raise FileNotFoundError(f"Parquet file not found at {cls.parquet_path}")

        print(f"Loading data from {cls.parquet_path}")
        cls.df = pd.read_parquet(cls.parquet_path)

    def test_schema_columns(self):
        """Verify all required columns are present."""
        REQUIRED_COLS = ['patient_id', 'date', 'trial_type', 'sentences', 'start_time', 'end_time', 'duration']
        missing = [c for c in REQUIRED_COLS if c not in self.df.columns]
        self.assertEqual(missing, [], f"Missing columns: {missing}")

    def test_trial_types_normalized(self):
        """Verify trial types are normalized and no lang_XX leaked."""
        actual_types = set(self.df['trial_type'].unique())
        
        # Check for lang_XX leakage
        lang_leakage = [t for t in actual_types if 'lang_' in str(t)]
        self.assertEqual(lang_leakage, [], f"Found un-normalized trial types: {lang_leakage}")

    def test_sentences_structure(self):
        """Verify sentences are stored as lists/arrays."""
        if len(self.df) > 0:
            sample = self.df['sentences'].iloc[0]
            self.assertTrue(isinstance(sample, (np.ndarray, list)), 
                            f"Sentences should be list/array, got {type(sample)}")

    def test_data_rescue_logic(self):
        """Verify that lang_XX data was correctly moved to sentences events."""
        # We look for specific known events like '11' or '70' that come from the lang_XX data
        has_rescued = False
        for sentences in self.df['sentences']:
            if len(sentences) > 0:
                # Handle numpy array of dicts or list of dicts
                events = [s.get('event') for s in sentences if isinstance(s, dict)]
                if '11' in events or '70' in events:
                    has_rescued = True
                    break
        
        self.assertTrue(has_rescued, "Failed to find rescued lang_XX data (events '11' or '70') in sentences column.")

    def test_no_critical_nulls(self):
        """Verify patient_id is never null."""
        null_count = self.df['patient_id'].isnull().sum()
        self.assertEqual(null_count, 0, f"Found {null_count} rows with null patient_id")

if __name__ == '__main__':
    unittest.main()
