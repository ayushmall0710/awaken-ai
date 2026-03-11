import json

import numpy as np
import pandas as pd

from src.data_processing.normalization import (
    convert_new_format_to_canonical,
    is_new_format,
)
from src.data_processing.pipeline import process_stimulus_df


class TestDataIntegrity:
    """Integration tests for unified stimulus data integrity."""

    def test_schema_columns(self, unified_df):
        """Verify all required columns are present."""
        required_cols = [
            "patient_id",
            "date",
            "trial_type",
            "sentences",
            "start_time",
            "end_time",
            "duration",
            "source_file",
        ]
        missing = [c for c in required_cols if c not in unified_df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_trial_types_normalized(self, unified_df):
        """Verify trial types are normalized and no lang_XX leaked."""
        actual_types = set(unified_df["trial_type"].unique())

        # Check for lang_XX leakage
        lang_leakage = [t for t in actual_types if "lang_" in str(t)]
        assert lang_leakage == [], f"Found un-normalized trial types: {lang_leakage}"

    def test_sentences_structure(self, unified_df):
        """Verify sentences are stored as lists/arrays."""
        if len(unified_df) > 0:
            sample = unified_df["sentences"].iloc[0]
            assert isinstance(sample, (np.ndarray, list)), f"Sentences should be list/array, got {type(sample)}"

    def test_data_rescue_logic(self, unified_df):
        """Verify that lang_XX data was correctly moved to sentences events."""
        # Look for specific known events like '11' or '70' that come from lang_XX data
        has_rescued = False
        for sentences in unified_df["sentences"]:
            if len(sentences) > 0:
                events = [s.get("event") for s in sentences if isinstance(s, dict)]
                if "11" in events or "70" in events:
                    has_rescued = True
                    break

        assert has_rescued, "Failed to find rescued lang_XX data (events '11' or '70') in sentences column."

    def test_no_critical_nulls(self, unified_df):
        """Verify patient_id is never null."""
        null_count = unified_df["patient_id"].isnull().sum()
        assert null_count == 0, f"Found {null_count} rows with null patient_id"

    def test_source_file_provenance(self, unified_df):
        """Verify source_file column is populated and valid."""
        # Check no nulls in source_file
        null_count = unified_df["source_file"].isnull().sum()
        assert null_count == 0, f"Found {null_count} rows with null source_file"

        # Check that all source files end with .csv
        invalid_sources = unified_df[~unified_df["source_file"].str.endswith(".csv")]
        assert len(invalid_sources) == 0, f"Found {len(invalid_sources)} rows with invalid source_file extensions"

        # Check that source files contain expected patterns
        valid_patterns = ["patient_df", "stimulus_results"]
        valid_count = unified_df["source_file"].apply(lambda x: any(pattern in x for pattern in valid_patterns)).sum()

        # At least some files should match known patterns
        assert valid_count > 0, "No rows found with recognized source file patterns"


class TestDataIntegrityEdgeCases:
    """Edge case tests for data integrity validation."""

    def test_empty_sentences_valid(self):
        """Empty sentences list should be valid."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lcmd"],
                "sentences": ["[]"],  # Raw string input from CSV
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
            }
        )
        # Should not raise - empty list is valid
        # Process the dataframe to test pipeline handling
        processed = process_stimulus_df(df, "test_source")
        # Sentences is expected to be an empty list (actual list check)
        # Note: 'left_command' having empty sentences is expected behavior for now
        assert isinstance(processed["sentences"].iloc[0], list)
        assert len(processed["sentences"].iloc[0]) == 0

    def test_multiple_events_in_sentences(self):
        """Multiple events in sentences should all be accessible."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lang_11"],
                # Raw JSON string input simulating CSV read
                "sentences": [
                    json.dumps(
                        [
                            {"event": "11", "onset_time": 1.0},
                            {"event": "12", "onset_time": 2.0},
                            {"event": "13", "onset_time": 3.0},
                        ]
                    )
                ],
                "start_time": [1.0],
                "end_time": [4.0],
                "duration": [3.0],
            }
        )
        # Pass through pipeline
        processed = process_stimulus_df(df, "test_source")
        # Verify structure preserved
        processed_events = [s["event"] for s in processed["sentences"].iloc[0]]
        assert processed_events == ["11", "12", "13"]

    def test_trial_type_values(self):
        """Known trial types should be in expected set."""
        expected_types = {
            "language",
            "left_command",
            "right_command",
            "oddball",
            "loved_one_voice",
            "unknown",
        }
        df = pd.DataFrame(
            {
                "patient_id": ["P001"] * 5,
                "date": ["2024-01-01"] * 5,
                "trial_type": ["lang_11", "lcmd", "rcmd", "odd", "loved_one"],
                "sentences": ["[]"] * 5,  # Raw strings
                "start_time": [1.0] * 5,
            }
        )

        # Run through pipeline to verify normalization
        processed = process_stimulus_df(df, "test_source")
        processed_types = set(processed["trial_type"].unique())

        # All actual types should be in expected set
        unexpected = processed_types - expected_types
        assert unexpected == set(), f"Unexpected trial types: {unexpected}"


def _make_new_format_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal new-format DataFrame from a list of row dicts."""
    base = {
        "patient_id": "CON010",
        "date": "2026-03-06",
        "start_time": 1000.0,
        "end_time": 1010.0,
    }
    records = [{**base, **r} for r in rows]
    return pd.DataFrame(records)


class TestNewCON010Format:
    """Unit tests for new CON010 (Mar 2026) CSV format detection and conversion."""

    def test_format_detection_new(self):
        """is_new_format returns True for new-format DataFrame."""
        df = _make_new_format_df([{"stim_type": "familiar", "notes": ""}])
        assert is_new_format(df) is True

    def test_format_detection_old(self):
        """is_new_format returns False for old-format DataFrame."""
        df = pd.DataFrame({"patient_id": ["CON008"], "trial_type": ["language"], "sentences": ["[]"]})
        assert is_new_format(df) is False

    def test_familiar_maps_to_loved_one_voice(self):
        """stim_type='familiar' converts to trial_type='loved_one_voice'."""
        df = _make_new_format_df([{"stim_type": "familiar", "notes": "file: CON010_femalevoice.wav"}])
        result = convert_new_format_to_canonical(df)
        assert result["trial_type"].iloc[0] == "loved_one_voice"

    def test_unfamiliar_maps_to_control(self):
        """stim_type='unfamiliar' converts to trial_type='control'."""
        df = _make_new_format_df([{"stim_type": "unfamiliar", "notes": "speaker: Hannah"}])
        result = convert_new_format_to_canonical(df)
        assert result["trial_type"].iloc[0] == "control"

    def test_sync_pulse_preserved(self):
        """manual_sync_pulse rows survive conversion with trial_type='manual_sync_pulse'."""
        df = _make_new_format_df([{"stim_type": "manual_sync_pulse", "notes": "Manual sync pulse at 11:07:21"}])
        result = convert_new_format_to_canonical(df)
        assert "manual_sync_pulse" in result["trial_type"].values

    def test_language_sentences_parsed(self):
        """language notes are parsed into a list of event dicts."""
        notes = "Sentences: ['8', '16', '14']"
        df = _make_new_format_df([{"stim_type": "language", "notes": notes}])
        result = convert_new_format_to_canonical(df)
        sentences = result["sentences"].iloc[0]
        assert isinstance(sentences, list)
        assert len(sentences) == 3
        events = [s["event"] for s in sentences]
        assert events == ["8", "16", "14"]
        assert all(s["onset_time"] is None for s in sentences)

    def test_oddball_aggregated_into_blocks(self):
        """Contiguous oddball+p rows are aggregated into a single trial row."""
        rows = [
            {"stim_type": "oddball+p", "notes": "standard_tone", "start_time": 100.0, "end_time": 100.03},
            {"stim_type": "oddball+p", "notes": "standard_tone", "start_time": 101.0, "end_time": 101.03},
            {"stim_type": "oddball+p", "notes": "rare_tone", "start_time": 102.0, "end_time": 102.03},
        ]
        df = _make_new_format_df(rows)
        result = convert_new_format_to_canonical(df)
        oddball_rows = result[result["trial_type"] == "oddball"]
        # All 3 tones should collapse into 1 block row
        assert len(oddball_rows) == 1
        sentences = oddball_rows["sentences"].iloc[0]
        assert len(sentences) == 3
        assert sentences[0]["event"] == "standard"
        assert sentences[2]["event"] == "rare"

    def test_duration_computed(self):
        """duration is computed as end_time - start_time when column absent."""
        df = _make_new_format_df([{"stim_type": "familiar", "notes": "", "start_time": 200.0, "end_time": 210.5}])
        result = convert_new_format_to_canonical(df)
        assert abs(result["duration"].iloc[0] - 10.5) < 1e-6

    def test_full_pipeline_new_format(self):
        """process_stimulus_df produces the canonical schema for new-format input."""
        rows = [
            {"stim_type": "manual_sync_pulse", "notes": "sync", "start_time": 50.0, "end_time": 51.0},
            {"stim_type": "familiar", "notes": "file: CON010_femalevoice.wav", "start_time": 100.0, "end_time": 110.0},
            {"stim_type": "unfamiliar", "notes": "speaker: Hannah", "start_time": 112.0, "end_time": 122.0},
            {"stim_type": "language", "notes": "Sentences: ['3', '7']", "start_time": 130.0, "end_time": 146.0},
            {"stim_type": "oddball+p", "notes": "standard_tone", "start_time": 200.0, "end_time": 200.03},
            {"stim_type": "oddball+p", "notes": "rare_tone", "start_time": 201.0, "end_time": 201.03},
        ]
        df = _make_new_format_df(rows)
        result = process_stimulus_df(df, "CON010_2026-03-06_stimulus_results.csv")

        # Canonical schema present
        required = [
            "patient_id",
            "date",
            "trial_type",
            "sentences",
            "start_time",
            "end_time",
            "duration",
            "source_file",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

        # Trial types are normalised correctly
        types = set(result["trial_type"].unique())
        assert "loved_one_voice" in types
        assert "control" in types
        assert "language" in types
        assert "oddball" in types
        assert "manual_sync_pulse" in types
        # stim_type / notes should not appear
        assert "stim_type" not in result.columns
        assert "notes" not in result.columns
