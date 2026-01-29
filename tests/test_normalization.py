import pytest
import pandas as pd

from data_processing.normalization import normalize_trial_type, normalize_sentences


class TestNormalizeTrialType:
    """Unit tests for normalize_trial_type function."""

    @pytest.mark.parametrize("input_val", [pd.NA, None])
    def test_nan_handling(self, input_val):
        """NaN/None values should return 'unknown'."""
        assert normalize_trial_type(input_val) == "unknown"

    @pytest.mark.parametrize("input_val,expected", [
        ("lang_11", "language"),
        ("lang_70", "language"),
        ("lang_1", "language"),
        ("lang_999", "language"),
    ])
    def test_lang_xx_pattern(self, input_val, expected):
        """lang_XX patterns should normalize to 'language'."""
        assert normalize_trial_type(input_val) == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("lcmd", "left_command"),
        ("rcmd", "right_command"),
        ("lang", "language"),
        ("odd", "oddball"),
        ("loved_one", "loved_one_voice"),
        ("language_11", "language"),
    ])
    def test_trial_type_map(self, input_val, expected):
        """Known trial types should map to their normalized values."""
        assert normalize_trial_type(input_val) == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("lcmd+p", "left_command"),
        ("rcmd+p", "right_command"),
        ("odd+p", "oddball"),
    ])
    def test_plus_p_suffix_removal(self, input_val, expected):
        """+p suffix should be removed before normalization."""
        assert normalize_trial_type(input_val) == expected

    @pytest.mark.parametrize("input_val,expected", [
        ("LCMD", "left_command"),
        ("Lcmd", "left_command"),
        ("LANG_11", "language"),
        ("Lang_70", "language"),
    ])
    def test_case_insensitivity(self, input_val, expected):
        """Trial type normalization should be case-insensitive."""
        assert normalize_trial_type(input_val) == expected

    @pytest.mark.parametrize("input_val,expected", [
        (" lcmd ", "left_command"),
        ("  rcmd  ", "right_command"),
        ("\tlcmd\n", "left_command"),
    ])
    def test_whitespace_stripping(self, input_val, expected):
        """Whitespace should be stripped before normalization."""
        assert normalize_trial_type(input_val) == expected

    def test_unknown_trial_type_passthrough(self):
        """Unknown trial types should pass through as lowercase."""
        assert normalize_trial_type("custom_type") == "custom_type"
        assert normalize_trial_type("CUSTOM") == "custom"


class TestNormalizeSentences:
    """Unit tests for normalize_sentences function."""

    @pytest.mark.parametrize("input_val", [pd.NA, None])
    def test_na_handling(self, input_val):
        """NA/None values should return empty list."""
        assert normalize_sentences(input_val) == []

    @pytest.mark.parametrize("input_val", ["", "[]"])
    def test_empty_string_handling(self, input_val):
        """Empty strings and empty list strings should return empty list."""
        assert normalize_sentences(input_val) == []

    def test_json_list_parsing(self):
        """JSON string lists should be parsed and converted to dicts."""
        result = normalize_sentences("[1, 2, 3]")
        expected = [
            {'event': '1', 'onset_time': None},
            {'event': '2', 'onset_time': None},
            {'event': '3', 'onset_time': None},
        ]
        assert result == expected

    def test_ast_list_parsing(self):
        """AST-parseable strings should be converted to dicts."""
        result = normalize_sentences("['a', 'b']")
        expected = [
            {'event': 'a', 'onset_time': None},
            {'event': 'b', 'onset_time': None},
        ]
        assert result == expected

    def test_raw_string_handling(self):
        """Unparseable raw strings should become single-item list."""
        result = normalize_sentences("raw text event")
        expected = [{'event': 'raw text event', 'onset_time': None}]
        assert result == expected

    def test_mixed_types_in_list_as_string(self):
        """JSON string with mixed types should all be normalized to dicts."""
        # In practice, data comes from CSV as strings
        input_str = '[10, "hello", {"event": "x", "onset_time": 1.5}]'
        result = normalize_sentences(input_str)
        expected = [
            {'event': '10', 'onset_time': None},
            {'event': 'hello', 'onset_time': None},
            {'event': 'x', 'onset_time': 1.5},
        ]
        assert result == expected

    def test_dict_preservation(self):
        """Existing dicts should be preserved."""
        input_list = [{'event': 'test', 'onset_time': 2.5, 'extra': 'data'}]
        result = normalize_sentences(input_list)
        assert result == [{'event': 'test', 'onset_time': 2.5, 'extra': 'data'}]

    def test_numeric_event_codes_as_string(self):
        """Numeric event codes in JSON string should be converted to string events."""
        # In practice, data comes from CSV as strings
        result = normalize_sentences("[11, 70]")
        expected = [
            {'event': '11', 'onset_time': None},
            {'event': '70', 'onset_time': None},
        ]
        assert result == expected

    def test_single_numeric_string(self):
        """Single numeric string should be parsed as JSON."""
        result = normalize_sentences("11")
        expected = [{'event': '11', 'onset_time': None}]
        assert result == expected

    def test_nested_json_string(self):
        """JSON string with dict should be parsed."""
        result = normalize_sentences('[{"event": "test", "onset_time": 1.0}]')
        expected = [{'event': 'test', 'onset_time': 1.0}]
        assert result == expected
