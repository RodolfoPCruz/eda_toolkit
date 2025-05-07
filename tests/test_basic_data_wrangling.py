"""
Test suite for the `basic_wrangling` function.
"""

import numpy as np
import pandas as pd

from eda_toolkit.basic_data_wrangling import basic_wrangling


def test_drop_high_nan_columns():
    """
    Test the removal of columns with high proportion of NaN values.

    This test verifies that the `basic_wrangling` function correctly
    removes columns where the proportion of NaN values exceeds the
    specified threshold.

    Asserts:
        - Column 'A' is removed because it has 75% NaN values, which
          is above the 50% threshold.
        - Column 'B' remains since it has no NaN values.
    """

    df = pd.DataFrame({"A": [1, np.nan, np.nan, np.nan], "B": [1, 2, 3, 4]})
    result = basic_wrangling(df, proportion_nan_thresh=0.5)
    assert "A" not in result.columns
    assert "B" in result.columns


def test_drop_single_unique_value_column():
    """
    Test the removal of columns with a single unique value.

    This test verifies that the `basic_wrangling` function correctly
    removes columns that contain only one unique value.

    Asserts:
        - Column 'A' is removed because it contains only the value 1.
        - Column 'B' remains since it contains multiple unique values.
    """

    df = pd.DataFrame({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})

    result = basic_wrangling(df)
    assert "A" not in result.columns
    assert "B" in result.columns


def test_drop_high_uniqueness_categorical_column():
    """
    Test that the `basic_wrangling` function removes categorical columns
    with a high proportion of unique values.

    This test verifies that the function correctly removes categorical
    columns where the proportion of unique values exceeds the specified
    threshold.

    Asserts:
        - Column 'A' is removed because it has 100% unique values, which
          is above the 50% threshold.
        - Column 'B' remains since it has 75% unique values.
    """

    df = pd.DataFrame({"A": ["a", "b", "c", "d"], "B": ["x", "y", "x", "x"]})
    result = basic_wrangling(df, proportion_unique_thresh=0.9)
    assert "A" not in result.columns
    assert "B" in result.columns


def test_ignore_numeric_high_uniqueness():
    """
    Test that the `basic_wrangling` function ignores numeric columns with
    high proportion of unique values.

    This test verifies that the function correctly leaves numeric columns
    untouched, even if they have a high proportion of unique values.

    Asserts:
        - Column 'A' is not removed because it is numeric, even though it
          has 100% unique values.
        - Column 'B' is removed because it is categorical and has a high
          proportion of unique values.
    """
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],  # high uniqueness, but numeric
            "B": ["x", "y", "x", "x"],
        }
    )
    result = basic_wrangling(df, proportion_unique_thresh=0.9)
    assert "A" in result.columns
    assert "B" in result.columns


def test_only_selected_columns_wrangled():
    """
    Test that the `basic_wrangling` function only wrangles the specified
    columns.

    This test verifies that the function correctly removes columns specified
    in the `columns` argument, while leaving the other columns untouched.

    Asserts:
        - Column 'A' is removed because it has a high proportion of NaN values.
        - Column 'B' remains since it was not specified in the `columns`
                    argument.
        - Column 'C' is removed because it has only one unique value
    """
    df = pd.DataFrame(
        {
            "A": [np.nan, np.nan, np.nan, np.nan],
            "B": ["a", "b", "c", "d"],
            "C": [1, 1, 1, 1],
        }
    )
    result = basic_wrangling(df, columns=["A", "C"])
    assert "A" not in result.columns
    assert "B" in result.columns  # untouched
    assert "C" not in result.columns


def test_handle_nonexistent_column():
    """
    Test that the `basic_wrangling` function handles nonexistent columns
    correctly.

    This test verifies that the function does not fail when a nonexistent
    column is specified in the `columns` argument.

    Asserts:
        - Column 'A' is not removed since it exists in the dataframe.
    """
    df = pd.DataFrame({"A": [1, 2, 3]})
    result = basic_wrangling(df, columns=["A", "Z"])  # 'Z' does not exist
    assert "A" in result.columns
