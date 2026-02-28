import pandas as pd

from src.data_processing.pipeline import (
    REQUIRED_COLS,
    generate_session_ids,
    generate_trial_ids,
    process_stimulus_df,
)


class TestProcessStimulusDf:
    """Tests for the process_stimulus_df transformation function."""

    def test_lang_xx_rescue(self, raw_stimulus_df):
        """lang_XX trial types with empty sentences should have event ID rescued."""
        result = process_stimulus_df(raw_stimulus_df, source_name="test.csv")
        # lang_11 row should have sentences with event '11'
        assert result.iloc[0]["sentences"] == [{"event": "11", "onset_time": None}]

    def test_trial_normalization(self, raw_stimulus_df):
        """Trial types should be normalized correctly."""
        result = process_stimulus_df(raw_stimulus_df, source_name="test.csv")
        assert result.iloc[0]["trial_type"] == "language"
        assert result.iloc[1]["trial_type"] == "left_command"

    def test_schema_reindex(self):
        """Output should contain only the required columns."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lcmd"],
                "sentences": ["[]"],
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
                "extra_column": ["should_be_dropped"],
                "another_extra": [999],
            }
        )
        result = process_stimulus_df(df, source_name="test.csv")
        # process_stimulus_df outputs base cols (sans session_id, trial_id)
        base_cols = [c for c in REQUIRED_COLS if c not in ("session_id", "trial_id")]
        assert list(result.columns) == base_cols
        assert "extra_column" not in result.columns
        assert "another_extra" not in result.columns

    def test_missing_trial_type_column(self):
        """Missing trial_type column should default to 'unknown'."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "sentences": ["[1, 2]"],
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
            }
        )
        result = process_stimulus_df(df, source_name="test.csv")
        assert result.iloc[0]["trial_type"] == "unknown"

    def test_missing_sentences_column(self):
        """Missing sentences column should default to empty list."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lcmd"],
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
            }
        )
        result = process_stimulus_df(df, source_name="test.csv")
        assert result.iloc[0]["sentences"] == []

    def test_source_provenance(self):
        """source_file column should be set to the provided source_name."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001", "P002"],
                "date": ["2024-01-01", "2024-01-02"],
                "trial_type": ["lcmd", "rcmd"],
                "sentences": ["[]", "[]"],
                "start_time": [1.0, 2.0],
                "end_time": [2.0, 3.0],
                "duration": [1.0, 1.0],
            }
        )
        result = process_stimulus_df(df, source_name="test_file.csv")
        assert all(result["source_file"] == "test_file.csv")

    def test_does_not_modify_input(self):
        """Original DataFrame should not be modified."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lang_11"],
                "sentences": [""],
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
            }
        )
        original_trial_type = df.iloc[0]["trial_type"]
        original_sentences = df.iloc[0]["sentences"]

        process_stimulus_df(df, source_name="test.csv")

        assert df.iloc[0]["trial_type"] == original_trial_type
        assert df.iloc[0]["sentences"] == original_sentences

    def test_lang_rescue_only_when_sentences_empty(self):
        """lang_XX rescue should not overwrite existing sentence data."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001"],
                "date": ["2024-01-01"],
                "trial_type": ["lang_99"],
                "sentences": ["[1, 2, 3]"],  # Not empty
                "start_time": [1.0],
                "end_time": [2.0],
                "duration": [1.0],
            }
        )
        result = process_stimulus_df(df, source_name="test.csv")
        # Should parse existing sentences, not rescue
        assert result.iloc[0]["sentences"] == [
            {"event": "1", "onset_time": None},
            {"event": "2", "onset_time": None},
            {"event": "3", "onset_time": None},
        ]

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame with required columns."""
        df = pd.DataFrame(
            columns=[
                "patient_id",
                "date",
                "trial_type",
                "sentences",
                "start_time",
                "end_time",
                "duration",
            ]
        )
        result = process_stimulus_df(df, source_name="test.csv")
        assert len(result) == 0
        base_cols = [c for c in REQUIRED_COLS if c not in ("session_id", "trial_id")]
        assert list(result.columns) == base_cols

    def test_sentences_normalization_in_pipeline(self):
        """Sentences should be properly normalized to list of dicts."""
        df = pd.DataFrame(
            {
                "patient_id": ["P001", "P002", "P003"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "trial_type": ["lcmd", "rcmd", "odd"],
                "sentences": ["[1, 2]", "['a', 'b']", '[{"event": "x"}]'],
                "start_time": [1.0, 2.0, 3.0],
                "end_time": [2.0, 3.0, 4.0],
                "duration": [1.0, 1.0, 1.0],
            }
        )
        result = process_stimulus_df(df, source_name="test.csv")

        assert result.iloc[0]["sentences"] == [
            {"event": "1", "onset_time": None},
            {"event": "2", "onset_time": None},
        ]
        assert result.iloc[1]["sentences"] == [
            {"event": "a", "onset_time": None},
            {"event": "b", "onset_time": None},
        ]
        assert result.iloc[2]["sentences"] == [{"event": "x"}]


class TestGenerateSessionIds:
    """Tests for vectorized session_id generation."""

    def test_basic_format(self):
        df = pd.DataFrame(
            {
                "patient_id": ["CON008", "CON008"],
                "date": ["2025-01-10", "2025-01-10"],
            }
        )
        result = generate_session_ids(df)
        assert result.iloc[0] == "s_CON008_202501100000"
        assert result.iloc[0] == result.iloc[1]  # same session

    def test_different_dates_different_ids(self):
        df = pd.DataFrame(
            {
                "patient_id": ["CON008", "CON008"],
                "date": ["2025-01-10", "2025-01-11"],
            }
        )
        result = generate_session_ids(df)
        assert result.iloc[0] != result.iloc[1]

    def test_malformed_date_fallback(self):
        df = pd.DataFrame(
            {
                "patient_id": ["CON008"],
                "date": ["bad-date_123!"],
            }
        )
        result = generate_session_ids(df)
        # Should strip non-alphanumeric chars: "bad-date_123!" -> "baddate_123"
        assert result.iloc[0] == "s_CON008_baddate_123"


class TestGenerateTrialIds:
    """Tests for vectorized trial_id generation."""

    def test_sequential_ids(self):
        df = pd.DataFrame(
            {
                "session_id": ["s_CON008_202501101430"] * 3,
                "trial_type": ["language", "language", "oddball"],
                "start_time": [100.0, 200.0, 300.0],
            }
        )
        result = generate_trial_ids(df)
        assert list(result) == ["lt1", "lt2", "obt1"]

    def test_unknown_type_fallback(self):
        df = pd.DataFrame(
            {
                "session_id": ["s_CON008_202501101430"],
                "trial_type": ["mystery_type"],
                "start_time": [100.0],
            }
        )
        result = generate_trial_ids(df)
        assert result.iloc[0] == "unk1"
