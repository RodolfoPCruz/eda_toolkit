"""
Disable logging during tests
"""

import logging

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
