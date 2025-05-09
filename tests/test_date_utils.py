"""
Test suite for the `parse_and_format_dates` function
"""

from datetime import datetime

import pandas as pd
import pytest

from eda_toolkit.date_utils import (
    create_new_date_columns,
    parse_and_format_dates,
)


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("2023-12-25", "2023-12-25"),
        ("25-12-2023", "2023-12-25"),
        ("12/25/2023", "2023-12-25"),
        ("25 Dec 2023", "2023-12-25"),
        ("December 25, 2023", "2023-12-25"),
    ],
)
def test_valid_formats_string_output(input_str, expected):
    """
    Test that the `parse_and_format_dates` function correctly formats
    date strings in various valid input formats to the standard format.

    This test uses parameterized inputs to verify that the function can
    handle multiple date formats and return the expected formatted date
    string.

    Args:
        input_str: A date string in one of the valid formats.
        expected: The expected output in the standard format.

    Asserts:
        - The parsed and formatted date matches the expected output.
    """

    assert parse_and_format_dates(input_str) == expected


@pytest.mark.parametrize(
    "input_str",
    [
        "2023-12-25",
        "25-12-2023",
    ],
)
def test_valid_formats_datetime_output(input_str):
    """
    Test that the `parse_and_format_dates` function correctly converts
    date strings in various valid input formats to datetime objects.

    This test uses parameterized inputs to verify that the function can
    handle multiple date formats and return datetime objects.

    Args:
        input_str: A date string in one of the valid formats.

    Asserts:
        - The parsed date matches the expected datetime object.
    """

    expected = datetime(2023, 12, 25)
    assert (
        parse_and_format_dates(input_str, return_type="datetime") == expected
    )


def test_custom_output_format():
    """
    Test that the `parse_and_format_dates` function correctly formats
    date strings with a custom output format.

    Args:
        input_str: A date string in one of the valid formats.
        standard_format: A custom output format string.

    Asserts:
        - The parsed date matches the expected output string in the custom
          format.
    """
    result = parse_and_format_dates("25-12-2023", standard_format="%d/%m/%Y")
    assert result == "25/12/2023"


@pytest.mark.parametrize(
    "input_str",
    [
        "25/12/23",
        "2023.12.25",
        "random text",
    ],
)
def test_invalid_formats(input_str):
    """
    Test that the `parse_and_format_dates` function correctly handles
    invalid date strings.

    This test uses parameterized inputs to verify that the function can
    handle multiple invalid date formats and return None.

    Args:
        input_str: An invalid date string.

    Asserts:
        - The parsed date matches None.
    """
    assert parse_and_format_dates(input_str) is None


@pytest.mark.parametrize(
    "bad_input",
    [
        12345,
        None,
        12.5,
        [],
    ],
)
def test_invalid_input_type(bad_input):
    """
    Test that the `parse_and_format_dates` function correctly handles
    invalid input types.

    This test uses parameterized inputs to verify that the function can
    handle multiple invalid input types and return None.

    Args:
        bad_input: An invalid input type.

    Asserts:
        - The parsed date matches None.
    """
    assert parse_and_format_dates(bad_input) is None


def test_invalid_return_type(capfd):
    """
    Test that the `parse_and_format_dates` function correctly handles
    invalid return types.

    This test verifies that the function can handle multiple invalid
    return types and return the default output type (string).

    Args:
        capfd: pytest fixture that captures the output of the function.

    Asserts:
        - The function prints a message indicating that the return type
          is not valid.
        - The parsed date matches the default output type (string).
    """
    result = parse_and_format_dates("2023-12-25", return_type="invalid_type")
    captured = capfd.readouterr()
    assert "not a valid return type" in captured.out
    assert result == "2023-12-25"


def test_basic_extraction(sample_df):
    """
    Test that the `create_new_date_columns` function correctly extracts the
    date components (year, month, day, weekday) from a column of dates.

    This test uses a sample DataFrame with a single column of dates and
    verifies that the extracted components are correct.

    Args:
        sample_df: A sample DataFrame containing a column of dates, provided
        as a pytest fixture.

    Asserts:
        - The function correctly extracts the date components.
    """
    df = create_new_date_columns(
        sample_df.copy(), ["date_col"], calculate_difference=False
    )
    for suffix in ("_year", "_month", "_day", "_weekday"):
        assert f"date_col{suffix}" in df.columns
    assert list(df["date_col_year"]) == [2020, 2020]
    assert list(df["date_col_month"]) == [1, 1]
    assert list(df["date_col_day"]) == [1, 2]
    assert list(df["date_col_weekday"]) == ["Wednesday", "Thursday"]


def test_difference_with_valid_reference(sample_df):
    """
    Test that the `create_new_date_columns` function correctly calculates
    the difference in days between the reference date and the dates in
    the given column.

    This test verifies that the function accurately computes the date
    differences when a valid reference date is provided.

    Args:
        sample_df: A sample DataFrame containing a column of dates,
            provided as a pytest fixture.

    Asserts:
        - The calculated differences in days are correct.
        - The differences are represented as `pd.Timedelta` objects.
    """

    ref = "2020-01-05"
    df = create_new_date_columns(
        sample_df.copy(),
        ["date_col"],
        calculate_difference=True,
        reference_date=ref,
    )
    assert list(df["actual_date - date_col in days"]) == [4, 3]
    assert all(
        isinstance(x, pd.Timedelta) for x in df["actual_date - date_col"]
    )


def test_no_difference_columns_when_disabled(sample_df):
    """
    Test that the `create_new_date_columns` function does not create
    the date difference columns when `calculate_difference` is False.

    This test verifies that the function does not create the columns
    when `calculate_difference` is False.

    Args:
        sample_df: A sample DataFrame containing a column of dates,
            provided as a pytest fixture.

    Asserts:
        - The columns "actual_date - date_col" and
          "actual_date - date_col in days" are not created.
    """
    df = create_new_date_columns(
        sample_df.copy(), ["date_col"], calculate_difference=False
    )
    assert "actual_date - date_col" not in df.columns
    assert "actual_date - date_col in days" not in df.columns


def test_mixed_valid_and_invalid_dates():
    # Mix of valid and invalid
    df_input = pd.DataFrame(
        {
            "mix_dates": ["2020-01-01", "not a date"],
        }
    )
    df = create_new_date_columns(
        df_input.copy(), ["mix_dates"], calculate_difference=False
    )

    # Columns must exist
    assert "mix_dates_year" in df.columns
    # Valid row gets year, invalid gets NaN
    assert df.loc[0, "mix_dates_year"] == 2020
    assert pd.isna(
        df.loc[1, "mix_dates_year"]
    ), "Expected NaN for invalid date"
