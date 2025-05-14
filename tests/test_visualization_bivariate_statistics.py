"""
Test suite for testing the `visualization_bivariate_statistics` module
"""

# pylint: disable=import-error
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from eda_toolkit.visualization_bivariate_statistics import (
    compute_regression_statistics,
    generate_bar_plot,
    generate_heat_map,
    generate_regression_plot,
)


def test_compute_regression_statistics_output_keys():
    """
    Tests that the output dictionary of compute_regression_statistics
    contains the expected keys
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


def test_bar_plot_with_two_groups(monkeypatch, sample_data_two_groups):
    """
    Test that generate_bar_plot correctly creates a bar plot
    for a numerical and a categorical feature with two groups.

    This test verifies that the function completes without error
    and that plt.show is called, indicating that a plot was generated.

    Parameters:
        monkeypatch: A pytest fixture for safely modifying objects
                     during testing.
        sample_data_two_groups: A fixture providing a sample DataFrame
                                with two categorical groups and numerical
                                values for testing.

    Asserts:
        - The plt.show function is called, confirming plot generation.
    """

    called = {"show": False}

    def fake_show():

        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)
    generate_bar_plot(sample_data_two_groups, "value", "category")
    assert called["show"]


def test_bar_plot_with_three_groups(monkeypatch, sample_data_three_groups):
    """
    Test that generate_bar_plot correctly creates a bar plot
    for a numerical and a categorical feature with three groups.

    This test verifies that the function completes without error
    and that plt.show is called, indicating that a plot was generated.

    Parameters:
        monkeypatch: A pytest fixture for safely modifying objects
                     during testing.
        sample_data_three_groups: A fixture providing a sample DataFrame
                                  with three categorical groups and numerical
                                  values for testing.

    Asserts:
        - The plt.show function is called, confirming plot generation.
    """
    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)
    generate_bar_plot(sample_data_three_groups, "value", "category")
    assert called["show"]


def test_bar_plot_with_insufficient_data(monkeypatch, insufficient_data):
    """
    Test that generate_bar_plot correctly creates a bar plot
    for a numerical and a categorical feature with insufficient data.

    This test verifies that the function completes without error
    and that plt.show is called, indicating that a plot was generated.

    Parameters:
        monkeypatch: A pytest fixture for safely modifying objects
                     during testing.
        insufficient_data: A fixture providing a sample DataFrame with
                           categorical and numerical data, but with
                           insufficient data to generate a meaningful plot.

    Asserts:
        - The plt.show function is called, confirming plot generation.
    """
    called = {"show": False}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr(plt, "show", fake_show)
    generate_bar_plot(insufficient_data, "value", "category")
    assert called["show"]


def test_bar_plot_text_and_bars(monkeypatch, sample_data_two_groups):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, ax1, ax2 = generate_bar_plot(
        sample_data_two_groups, "value", "category"
    )

    # Assert bar labels on x-axis
    x_labels = [tick.get_text() for tick in ax1.get_xticklabels()]
    assert set(x_labels) == {"A", "B"}

    # Assert correct number of bars
    bars = ax1.patches
    assert len(bars) == 2

    # Assert text content in ax2 (summary stats)
    texts = [t.get_text() for t in ax2.texts]
    assert any("t-test" in t for t in texts)

    plt.close(fig)  # Always clean up


@pytest.mark.parametrize("axis_sum", [None, 0, 1])
def test_heatmap_outputs(monkeypatch, test_df, axis_sum):
    """
    Tests that generate_heat_map correctly renders a heatmap and
    accompanying chi-squared statistics for a given DataFrame and
    categorical feature pair.

    Parameters:
        monkeypatch: A pytest fixture for safely modifying objects
                     during testing.
        sample_df: A fixture providing a sample DataFrame with categorical
                   and numerical data for testing.
        axis_sum: An integer indicating which axis to sum over to calculate
                  proportions in the heatmap. One of: None (global sum),
                  0 (column-wise sum), or 1 (row-wise sum).

    Asserts:
        - The heatmap is correctly rendered in ax1.
        - The chi-squared statistics are correctly rendered in ax2.
    """
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, ax1, ax2 = generate_heat_map(
        test_df, "gender", "response", axis_sum=axis_sum
    )

    # Test ax1: heatmap structure
    assert len(ax1.collections) > 0, "Heatmap not rendered"
    assert ax1.get_title() == "Contingency Heatmap"

    # Test ax2: text exists and contains chi-squared info
    assert ax2.texts, "Statistics text not found"
    stats_text = ax2.texts[0].get_text()
    assert "Chi-squared" in stats_text
    assert "p-value" in stats_text

    plt.close(fig)  # Clean up


def test_invalid_axis_sum(monkeypatch, test_df):
    """
    Tests that generate_heat_map raises a ValueError for an invalid
    axis_sum value.

    Parameters:
        monkeypatch: A pytest fixture for safely modifying objects during
        testing.
        sample_df: A fixture providing a sample DataFrame with categorical
                   and numerical data for testing.

    Asserts:
        - A ValueError is raised with the correct message when axis_sum
        is not one of the valid options: None, 0, or 1.
    """

    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(
        ValueError, match="axis_sum must be one of: None, 0, or 1"
    ):
        generate_heat_map(test_df, "gender", "response", axis_sum=99)
