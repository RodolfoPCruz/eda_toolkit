"""
Disable logging during tests
"""

import logging

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def disable_logging():
    """
    Automatically disable logging for all tests.
    Re-enables it after each test.
    """
    logging.disable(logging.CRITICAL)  # Disable all logging
    yield
    logging.disable(logging.NOTSET)  # Re-enable logging


@pytest.fixture
def sample_df():
    """
    A sample DataFrame containing a column of dates for testing purposes.

    This fixture is used to test the `create_new_date_columns` function.

    Returns:
        pd.DataFrame: A DataFrame with a single column of dates.
    """
    return pd.DataFrame(
        {
            "date_col": ["2020-01-01", "2020-01-02"],
        }
    )


@pytest.fixture
def mixed_dataframe():
    """
    A sample DataFrame fixture with mixed data types for testing purposes.

    This fixture provides a DataFrame that includes numeric, binary
    categorical, and multi-class categorical columns, along with
    target columns for regression
    and classification tasks.

    Columns:
        - 'numeric1': Numeric values.
        - 'numeric2': Numeric values.
        - 'binary_cat': Binary categorical values ('A' or 'B').
        - 'multi_cat': Multi-class categorical values ('X', 'Y', or 'Z').
        - 'target_num': Numeric target for regression tasks.
        - 'target_bin': Binary target for classification tasks.

    Returns:
        pd.DataFrame: A DataFrame with various data types for testing.
    """

    return pd.DataFrame(
        {
            "numeric1": [1, 2, 3, 4, 5],
            "numeric2": [2, 4, 6, 8, 10],
            "binary_cat": ["A", "A", "B", "B", "B"],
            "multi_cat": ["X", "Y", "X", "Y", "Z"],
            "target_num": [5, 4, 3, 2, 1],
            "target_bin": ["yes", "no", "yes", "no", "yes"],
        }
    )


@pytest.fixture
def basic_numeric_df():
    """
    A basic DataFrame fixture with numeric columns for testing purposes.

    This fixture provides a DataFrame with a single feature column and a
    target column,
    both containing numeric values.

    Columns:
        - 'feature1': Numeric feature values.
        - 'target': Numeric target values.

    Returns:
        pd.DataFrame: A DataFrame with basic numeric columns for testing.
    """
    return pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "target": [2, 4, 6, 8, 10]}
    )


@pytest.fixture
def mixed_type_df():
    """
    A sample DataFrame fixture with mixed data types for testing purposes.

    This fixture provides a DataFrame with a numeric feature column
    containing a missing value, a categorical feature column, and a
    numeric target column.

    Columns:
        - 'feature1': Numeric feature values with one missing value.
        - 'feature2': Categorical feature values ('a', 'b', 'c').
        - 'target': Numeric target values.

    Returns:
        pd.DataFrame: A DataFrame with mixed data types for testing.
    """
    return pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 4, 5],
            "feature2": ["a", "a", "b", "b", "c"],
            "target": [2, 4, 6, 8, 10],
        }
    )


@pytest.fixture
def categorical_target_df():
    """
    A sample DataFrame fixture with categorical target for testing purposes.

    This fixture provides a DataFrame with a numeric feature column and a
    categorical target column.

    Columns:
        - 'feature1': Numeric feature values.
        - 'target': Categorical target values ('yes' or 'no').

    Returns:
        pd.DataFrame: A DataFrame with a categorical target for testing.
    """
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "target": ["yes", "yes", "no", "no", "yes"],
        }
    )


@pytest.fixture
def edge_case_df():
    """
    A sample DataFrame fixture for testing edge cases.

    This fixture provides a DataFrame with several edge cases:

    - A constant feature ('feature1').
    - A feature with all unique values ('feature2').
    - A feature with two classes each containing two values ('feature3').
    - A target column with all unique integer values.

    Returns:
        pd.DataFrame: A DataFrame with edge cases for testing.
    """

    return pd.DataFrame(
        {
            "feature1": [1, 1, 1, 1, 1],  # Constant
            "feature2": [1, 2, 3, 4, 5],  # All unique
            "feature3": ["a", "a", "b", "b", "c"],
            "target": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def sample_data_two_groups():
    """
    A sample DataFrame fixture with two categorical groups for testing.

    This fixture provides a DataFrame with a categorical 'category' column
    and a numeric 'value' column, containing two groups ('A' and 'B') with
    associated values for testing purposes.

    Columns:
        - 'category': Categorical feature with two groups ('A', 'B').
        - 'value': Numeric values associated with each category.

    Returns:
        pd.DataFrame: A DataFrame with two groups for testing.
    """

    return pd.DataFrame(
        {"category": ["A", "A", "B", "B"], "value": [1.0, 2.0, 2.0, 3.0]}
    )


@pytest.fixture
def sample_data_three_groups():
    """
    A sample DataFrame fixture with three categorical groups for testing.

    This fixture provides a DataFrame with a categorical 'category' column
    and a numeric 'value' column, containing three groups ('A', 'B', 'C')
    with associated values for testing purposes.

    Columns:
        - 'category': Categorical feature with three groups ('A', 'B', 'C').
        - 'value': Numeric values associated with each category.

    Returns:
        pd.DataFrame: A DataFrame with three groups for testing.
    """
    return pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def insufficient_data():
    """
    A sample DataFrame fixture with insufficient data for two-sample
    comparison.

    This fixture provides a DataFrame with a categorical 'category' column
    and a numeric 'value' column, containing only two groups ('A' and 'B')
    with two values in the 'B' group. This is insufficient for two-sample
    comparison.

    Columns:
        - 'category': Categorical feature with two groups ('A', 'B').
        - 'value': Numeric values associated with each category.

    Returns:
        pd.DataFrame: A DataFrame with insufficient data for two-sample
        comparison.
    """
    return pd.DataFrame(
        {"category": ["A", "B", "B"], "value": [1.0, 2.0, 3.0]}
    )


@pytest.fixture
def test_df():
    """
    A sample DataFrame fixture with binary categorical data for testing.

    This fixture provides a DataFrame with two columns: 'gender' with two
    categories ('M' and 'F'), and 'response' with two categories ('Yes' and
    'No'). The responses are evenly distributed between the two gender
    categories.

    Columns:
        - 'gender': Binary categorical feature with two categories
         ('M', 'F').
        - 'response': Binary categorical feature with two categories
        ('Yes', 'No').

    Returns:
        pd.DataFrame: A DataFrame with binary categorical data for testing.
    """
    return pd.DataFrame(
        {
            "gender": ["M", "F", "M", "F", "M", "F"],
            "response": ["Yes", "No", "Yes", "Yes", "No", "No"],
        }
    )
