import pandas as pd
import numpy as np
import json
from data_processing.pipeline import process_stimulus_df


class TestDataIntegrity:
    """Integration tests for unified stimulus data integrity."""

    def test_schema_columns(self, unified_df):
        """Verify all required columns are present."""
        required_cols = ['patient_id', 'date', 'trial_type', 'sentences',
                         'start_time', 'end_time', 'duration', 'source_file']
        missing = [c for c in required_cols if c not in unified_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_trial_types_normalized(self, unified_df):
        """Verify trial types are normalized and no lang_XX leaked."""
        actual_types = set(unified_df['trial_type'].unique())

        # Check for lang_XX leakage
        lang_leakage = [t for t in actual_types if 'lang_' in str(t)]
        assert lang_leakage == [], f"Found un-normalized trial types: {lang_leakage}"

    def test_sentences_structure(self, unified_df):
        """Verify sentences are stored as lists/arrays."""
        if len(unified_df) > 0:
            sample = unified_df['sentences'].iloc[0]
            assert isinstance(sample, (np.ndarray, list)), \
                f"Sentences should be list/array, got {type(sample)}"

    def test_data_rescue_logic(self, unified_df):
        """Verify that lang_XX data was correctly moved to sentences events."""
        # Look for specific known events like '11' or '70' that come from lang_XX data
        has_rescued = False
        for sentences in unified_df['sentences']:
            if len(sentences) > 0:
                events = [s.get('event') for s in sentences if isinstance(s, dict)]
                if '11' in events or '70' in events:
                    has_rescued = True
                    break

        assert has_rescued, \
            "Failed to find rescued lang_XX data (events '11' or '70') in sentences column."

    def test_no_critical_nulls(self, unified_df):
        """Verify patient_id is never null."""
        null_count = unified_df['patient_id'].isnull().sum()
        assert null_count == 0, f"Found {null_count} rows with null patient_id"

    def test_source_file_provenance(self, unified_df):
        """Verify source_file column is populated and valid."""
        # Check no nulls in source_file
        null_count = unified_df['source_file'].isnull().sum()
        assert null_count == 0, f"Found {null_count} rows with null source_file"

        # Check that all source files end with .csv
        invalid_sources = unified_df[~unified_df['source_file'].str.endswith('.csv')]
        assert len(invalid_sources) == 0, \
            f"Found {len(invalid_sources)} rows with invalid source_file extensions"

        # Check that source files contain expected patterns
        valid_patterns = ['patient_df', 'stimulus_results']
        valid_count = unified_df['source_file'].apply(
            lambda x: any(pattern in x for pattern in valid_patterns)
        ).sum()

        # At least some files should match known patterns
        assert valid_count > 0, "No rows found with recognized source file patterns"


class TestDataIntegrityEdgeCases:
    """Edge case tests for data integrity validation."""

    def test_empty_sentences_valid(self):
        """Empty sentences list should be valid."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'date': ['2024-01-01'],
            'trial_type': ['lcmd'],
            'sentences': ['[]'],  # Raw string input from CSV
            'start_time': [1.0],
            'end_time': [2.0],
            'duration': [1.0]
        })
        # Should not raise - empty list is valid
        # Process the dataframe to test pipeline handling
        processed = process_stimulus_df(df, "test_source")
        # Sentences is expected to be an empty list (actual list check)
        # Note: 'left_command' having empty sentences is expected behavior for now
        assert isinstance(processed['sentences'].iloc[0], list)
        assert len(processed['sentences'].iloc[0]) == 0

    def test_multiple_events_in_sentences(self):
        """Multiple events in sentences should all be accessible."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'date': ['2024-01-01'],
            'trial_type': ['lang_11'],
            # Raw JSON string input simulating CSV read
            'sentences': [json.dumps([
                {'event': '11', 'onset_time': 1.0},
                {'event': '12', 'onset_time': 2.0},
                {'event': '13', 'onset_time': 3.0},
            ])],
            'start_time': [1.0],
            'end_time': [4.0],
            'duration': [3.0]
        })
        # Pass through pipeline
        processed = process_stimulus_df(df, "test_source")
        # Verify structure preserved
        processed_events = [s['event'] for s in processed['sentences'].iloc[0]]
        assert processed_events == ['11', '12', '13']

    def test_trial_type_values(self):
        """Known trial types should be in expected set."""
        expected_types = {
            'language', 'left_command', 'right_command',
            'oddball', 'loved_one_voice', 'unknown'
        }
        df = pd.DataFrame({
            'patient_id': ['P001'] * 5,
            'date': ['2024-01-01'] * 5,
            'trial_type': ['lang_11', 'lcmd', 'rcmd',
                           'odd', 'loved_one'],
            'sentences': ['[]'] * 5,  # Raw strings
            'start_time': [1.0] * 5
        })
        
        # Run through pipeline to verify normalization
        processed = process_stimulus_df(df, "test_source")
        processed_types = set(processed['trial_type'].unique())
        
        # All actual types should be in expected set
        unexpected = processed_types - expected_types
        assert unexpected == set(), f"Unexpected trial types: {unexpected}"
