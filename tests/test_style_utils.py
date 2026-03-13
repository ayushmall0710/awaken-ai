import pytest

from src.reports import style_utils


def test_significance_constants_exist():
    assert hasattr(style_utils, "ICON_SIG_1")
    assert hasattr(style_utils, "ICON_SIG_2")
    assert hasattr(style_utils, "ICON_SIG_3")
    assert hasattr(style_utils, "ICON_SIG_NONE")

    assert style_utils.ICON_SIG_1 == "*"
    assert style_utils.ICON_SIG_2 == "**"
    assert style_utils.ICON_SIG_3 == "***"
    assert style_utils.ICON_SIG_NONE == ""


@pytest.mark.parametrize(
    "value, p_value, expected",
    [
        (0.5, 0.049, "0.5000*"),
        (0.5, 0.009, "0.5000**"),
        (0.5, 0.0009, "0.5000***"),
        (0.5, 0.05, "0.5000"),
        (0.5, 0.1, "0.5000"),
        (0.123456, 0.009, "0.1235**"),  # testing precision/rounding
    ],
)
def test_format_with_significance(value, p_value, expected):
    assert style_utils.format_with_significance(value, p_value) == expected


def test_format_with_significance_custom_precision():
    assert style_utils.format_with_significance(0.5, 0.009, precision=2) == "0.50**"
