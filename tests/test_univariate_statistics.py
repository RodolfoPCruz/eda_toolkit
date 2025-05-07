"""
Test suite for the `univariate_statistics` function.

This module tests:
- Numerical and categorical feature handling
- NaN/missing value computation
- Correctness of statistics like mean, median, mode, etc.

Edge cases covered:
- Columns with all NaNs
- Mixed-type data columns
"""

import numpy as np
import pandas as pd

from eda_toolkit.univariate_statistics import univariate_statistics


def test_numeric_categorical_features():
    """
    Test the univariate statistics for both numerical and categorical features.

    This test verifies that the `univariate_statistics` function correctly calculates
    the univariate statistics for both numerical and categorical features in the
    dataset.

    Asserts:
        - The 'num' column has the correct type, count, missing, unique, and mean values.
        - The 'cat' column has the correct type, count, missing, unique, and mode values.
    """
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5], "cat": ["a", "a", "b", "b", "b"]})

    stats = univariate_statistics(df)

    assert stats.loc["num"]["type"] == "int64"
    assert stats.loc["num"]["count"] == 5
    assert stats.loc["num"]["missing"] == 0
    assert stats.loc["num"]["unique"] == 5
    assert stats.loc["num"]["mean"] == 3

    assert stats.loc["cat"]["type"] == "object"
    assert stats.loc["cat"]["count"] == 5
    assert stats.loc["cat"]["missing"] == 0
    assert stats.loc["cat"]["unique"] == 2
    assert stats.loc["cat"]["mode"] == "b"


def test_nan_handling():
    """
    Test the handling of NaN values in univariate statistics.

    This test verifies that the `univariate_statistics` function correctly calculates
    the number of missing and non-missing values for both numerical and categorical
    features in the presence of NaN values in the dataset.

    Asserts:
        - The 'num' column has 2 missing values and 3 non-missing values.
        - The 'cat' column has 1 missing value and 4 non-missing values.
    """

    df = pd.DataFrame(
        {"num": [1, np.nan, 3, np.nan, 5], "cat": ["a", "a", None, "b", "b"]}
    )

    stats = univariate_statistics(df)

    assert stats.loc["num"]["missing"] == 2
    assert stats.loc["num"]["count"] == 3

    assert stats.loc["cat"]["missing"] == 1
    assert stats.loc["cat"]["count"] == 4


def test_numerical_features():
    """
    Test that the univariate_statistics function correctly calculates
    statistics for numerical columns in the DataFrame.

    Asserts:
        - The 'feat_1' and 'feat_2' columns have the correct mean, min, and max values.
    """
    df = pd.DataFrame({"feat_1": [5, 10, 15, 20, 25], "feat_2": [10, 20, 30, 40, 50]})

    stats = univariate_statistics(df)

    assert stats.loc["feat_1"]["type"] == "int64"
    assert stats.loc["feat_1"]["mean"] == 15
    assert stats.loc["feat_1"]["min_value"] == 5
    assert stats.loc["feat_2"]["max_value"] == 50


def test_all_nan():
    """
    Test that the univariate_statistics function correctly handles columns with all NaN values.

    This test verifies that the 'missing' statistic is correctly calculated for a column
    with all NaN values.

    Asserts:
        - The 'feat_1' column has 5 missing values.
    """
    df = pd.DataFrame({"feat_1": [None, None, None, None, None]})

    stats = univariate_statistics(df)

    assert stats.loc["feat_1"]["missing"] == 5


def test_categorical_columns():
    """
    Test that the univariate_statistics function correctly handles categorical columns.

    This test verifies that the function correctly calculates the mode of a categorical
    column and sets all other statistics to '-' for categorical columns.

    Asserts:
        - The 'cat_feature' column has a mode of 'a' or 'b'.
        - All other statistics are '-'.
    """
    df = pd.DataFrame({"cat_feature": ["a", "a", "b", "c", "c"]})

    stats = univariate_statistics(df)

    assert stats.loc["cat_feature"]["mode"] in ["a", "b"]
    assert stats.loc["cat_feature"]["min_value"] == "-"
    assert stats.loc["cat_feature"]["max_value"] == "-"
    assert stats.loc["cat_feature"]["q_1"] == "-"
    assert stats.loc["cat_feature"]["q_3"] == "-"
    assert stats.loc["cat_feature"]["median"] == "-"
    assert stats.loc["cat_feature"]["mean"] == "-"
    assert stats.loc["cat_feature"]["std"] == "-"
    assert stats.loc["cat_feature"]["skew"] == "-"
    assert stats.loc["cat_feature"]["kurtosis"] == "-"


def test_constant_column():
    """
    Test that the univariate_statistics function correctly handles constant columns.

    This test verifies that the standard deviation, skewness, and kurtosis of a constant
    column are all zero, and that the number of unique values is one.

    Asserts:
        - The 'constant_feature' column has a standard deviation of 0.
        - The 'constant_feature' column has a skewness of 0.
        - The 'constant_feature' column has a kurtosis of 0.
        - The 'constant_feature' column has one unique value.
    """
    df = pd.DataFrame({"constant_feature": [1, 1, 1, 1, 1]})

    stats = univariate_statistics(df)

    assert stats.loc["constant_feature"]["std"] == 0
    assert stats.loc["constant_feature"]["skew"] == 0
    assert stats.loc["constant_feature"]["kurtosis"] == 0
    assert stats.loc["constant_feature"]["unique"] == 1


def test_empty_dataframe():
    """
    Test that the univariate_statistics function correctly handles an empty dataframe.

    This test verifies that the function returns an empty dataframe when given an
    empty dataframe as input.

    Asserts:
        - The output of the function is an empty dataframe.
    """
    df = pd.DataFrame()
    stats = univariate_statistics(df)
    assert stats.empty


def test_rounding():
    """
    Test that the univariate_statistics function correctly rounds the results of
    calculations to the specified decimal place.

    This test verifies that the mean of a column is rounded to the specified decimal place.

    Asserts:
        - The mean of the 'x' column is rounded to 2 decimal places.
        - The mean of the 'x' column is a float.
    """
    df = pd.DataFrame({"x": [0.123456, 0.234567, 0.345678]})

    stats = univariate_statistics(df, round_to=2)

    assert stats.loc["x"]["mean"] == round(df["x"].mean(), 2)
    assert isinstance(stats.loc["x"]["mean"], float)


def test_mode_with_multiple_modes():
    """
    Test that the univariate_statistics function correctly handles a column
    with multiple modes.

    This test verifies that the mode of a column with multiple modes is
    one of the modes in the column.

    Asserts:
        - The mode of the 'f' column is one of the modes in the column.
    """
    df = pd.DataFrame({"f": [1, 1, 2, 2, 3]})

    stats = univariate_statistics(df)
    assert stats.loc["f"]["mode"] in [1, 2]
