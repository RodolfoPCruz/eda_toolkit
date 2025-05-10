"""
Test suite for testing the `visualization_bivariate_statistics` module
"""
# pylint: disable=import-error
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from eda_toolkit.visualization_bivariate_statistics import (
    compute_regression_statistics,
    generate_regression_plot,
)


def test_compute_regression_statistics_output_keys():
    """
    Tests that the output dictionary of compute_regression_statistics contains
    the expected keys
    """
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    stats = compute_regression_statistics(df, "x", "y")
    expected_keys = {
        "slope",
        "intercept",
        "pearson_r",
        "pearson_p",
        "spearman",
        "spearman_p",
        "kendall",
        "kendall_p",
        "skew_1",
        "skew_2",
    }
    assert set(stats.keys()) == expected_keys


def test_compute_regression_statistics_values_are_numeric():
    """
    Tests that the output of compute_regression_statistics is a
    dictionary where
    all the values are numeric (int or float).
    """
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    stats = compute_regression_statistics(df, "x", "y")
    for value in stats.values():
        assert isinstance(value, (int, float))


def test_generate_regression_plot_with_valid_data(monkeypatch):
    # Avoid displaying the plot during test
    """
    Tests that generate_regression_plot does not crash when given
    valid data.
    """
    monkeypatch.setattr(plt, "show", lambda: None)

    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 5, 4, 5]})
    # Should not raise
    generate_regression_plot(df, "x", "y")


def test_generate_regression_plot_raises_with_non_numeric(monkeypatch):
    """
    Tests that generate_regression_plot raises a TypeError when given
    non-numeric data.
    """
    monkeypatch.setattr(plt, "show", lambda: None)

    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    with pytest.raises(TypeError, match="Both columns must be numeric"):
        generate_regression_plot(df, "x", "y")


def test_generate_regression_plot_handles_missing(monkeypatch):
    """
    Tests that generate_regression_plot handles missing data correctly.

    The function should drop NA without error and plot the available data.
    """
    monkeypatch.setattr(plt, "show", lambda: None)

    df = pd.DataFrame({"x": [1, 2, None, 4], "y": [1, 2, 3, None]})
    # Should handle dropping NA without error
    generate_regression_plot(df, "x", "y")
