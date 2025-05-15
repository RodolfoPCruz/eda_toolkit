"""
Tests for the outliers module
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from eda_toolkit.outliers import (
    calculate_outlier_threshold,
    clean_outliers,
    clean_outliers_using_dbscan,
    search_eps_dbscan,
)


def test_calculate_outlier_threshold_normal_distribution(sample_data_outliers):
    """
    Test calculate_outlier_threshold function for normal distribution data.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing a normal distribution.

    Returns
    -------
    None

    Notes
    -----
    The test assert that the min threshold is less than the max threshold.
    """
    min_t, max_t = calculate_outlier_threshold(
        sample_data_outliers, "normal_dist"
    )
    assert isinstance(min_t, float)
    assert isinstance(max_t, float)
    assert min_t < max_t


def test_calculate_outlier_threshold_skewed_distribution(sample_data_outliers):
    """
    Test calculate_outlier_threshold function for skewed distribution data.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing a skewed distribution.

    Returns
    -------
    None

    Notes
    -----
    The test assert that the min threshold is less than the max threshold.
    """
    min_t, max_t = calculate_outlier_threshold(
        sample_data_outliers, "skewed_dist"
    )
    assert isinstance(min_t, float)
    assert isinstance(max_t, float)
    assert min_t < max_t


def test_clean_outliers_no_treatment(sample_data_outliers):
    """
    Test clean_outliers function with no outlier treatment.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified feature without applying any outlier treatment.
    """

    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["normal_dist"],
        outlier_treatment=None,
    )
    assert isinstance(df_cleaned, pd.DataFrame)
    assert "normal_dist" in df_cleaned.columns


def test_clean_outliers_remove(sample_data_outliers):
    """
    Test clean_outliers function with remove outlier treatment.

    Parameters
    ----------
    sample_data_outliers : pd.DataFrame
        A pandas DataFrame containing sample data with outliers.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified feature without outliers.
    """
    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["normal_dist"],
        outlier_treatment="remove",
    )
    assert (
        df_cleaned["normal_dist"]
        .between(
            *calculate_outlier_threshold(sample_data_outliers, "normal_dist")
        )
        .all()
    )


def test_clean_outliers_replace(sample_data_outliers):
    """
    Test clean_outliers function with replace outlier treatment.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data with outliers.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified feature with outliers replaced by the outlier thresholds.
    """
    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["normal_dist"],
        outlier_treatment="replace",
    )
    min_t, max_t = calculate_outlier_threshold(
        sample_data_outliers, "normal_dist"
    )
    assert df_cleaned["normal_dist"].max() <= max_t
    assert df_cleaned["normal_dist"].min() >= min_t


def test_clean_outliers_ignore_non_numeric_and_constant(sample_data_outliers):
    """
    Test that clean_outliers function ignores non numeric and constant
    features.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data.

    caplog : pytest fixture
        A pytest fixture to capture log messages.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified features without applying any outlier treatment.
    """
    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["non_numeric", "constant"],
        outlier_treatment="remove",
    )
    assert "non_numeric" in df_cleaned.columns
    assert "constant" in df_cleaned.columns
    assert df_cleaned.equals(sample_data_outliers.dropna())


def test_clean_outliers_ignore_binary(sample_data_outliers):
    """
    Test that clean_outliers function ignores binary features.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data.

    caplog : pytest fixture
        A pytest fixture to capture log messages.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified binary feature without altering its values when 'replace'
    outlier treatment is used.
    """

    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["binary"],
        outlier_treatment="replace",
    )
    assert set(df_cleaned["binary"].unique()).issubset({0, 1})


def test_invalid_outlier_treatment(sample_data_outliers):
    """
    Test that clean_outliers function will still process the dataframe even
    if the outlier_treatment method is invalid.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data.

    caplog : pytest fixture
        A pytest fixture to capture log messages.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified feature without applying any outlier treatment.
    """

    df_cleaned = clean_outliers(
        sample_data_outliers,
        features_list=["normal_dist"],
        outlier_treatment="invalid_method",
    )
    assert "normal_dist" in df_cleaned.columns


def test_clean_outliers_impute(sample_data_outliers):
    # Inject artificial outliers to ensure they get imputed
    """
    Test that clean_outliers function replaces outliers with imputed values
    when 'impute' outlier treatment is used.

    Parameters
    ----------
    sample_data : pd.DataFrame
        A pandas DataFrame containing sample data.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame containing the
    specified feature after imputing outliers with median values.
    """
    df = sample_data_outliers.copy()
    df.loc[0:4, "normal_dist"] = 1e6  # extreme high values
    df.loc[5:9, "normal_dist"] = -1e6  # extreme low values

    df_cleaned = clean_outliers(
        df, features_list=["normal_dist"], outlier_treatment="impute"
    )

    # Check that there are no extremely high or low values after imputation
    assert df_cleaned["normal_dist"].max() < 1e5
    assert df_cleaned["normal_dist"].min() > -1e5

    # Check that the column still exists and is float
    assert "normal_dist" in df_cleaned.columns
    assert pd.api.types.is_float_dtype(df_cleaned["normal_dist"])

    # Check that no NaN values are left in the column
    assert df_cleaned["normal_dist"].isna().sum() == 0


def test_returns_expected_output(sample_df_eps_search):
    """
    Test that search_eps_dbscan returns expected output.

    Parameters
    ----------
    sample_df : pd.DataFrame
        A pandas DataFrame containing sample data.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a DataFrame and a float as
    expected.
    """
    results, best_eps = search_eps_dbscan(
        sample_df_eps_search, plot=False, verbose=False
    )

    assert isinstance(results, pd.DataFrame)
    assert "eps" in results.columns
    assert "percentage_outliers(%)" in results.columns
    assert isinstance(best_eps, float)


def test_custom_desired_outlier_percentage(sample_df_eps_search):
    """
    Test that search_eps_dbscan returns a valid eps value when a
    custom outlier percentage is specified.

    Parameters
    ----------
    sample_df : pd.DataFrame
        A pandas DataFrame containing sample data.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns a valid eps value when a custom
    outlier percentage is specified.
    """
    _, best_eps = search_eps_dbscan(
        sample_df_eps_search,
        desired_percentage_outliers=0.1,
        plot=False,
        verbose=False,
    )
    assert best_eps > 0


def test_min_samples_effect(sample_df):
    """
    Test that search_eps_dbscan returns different best eps values when the
    min_samples parameter is changed.

    Parameters
    ----------
    sample_df : pd.DataFrame
        A pandas DataFrame containing sample data.

    Returns
    -------
    None

    Notes
    -----
    The test checks that the function returns different best eps values when
    the min_samples parameter is changed. This test is useful for
    understanding how the min_samples parameter influences the best eps
    value.
    """
    _, eps_low = search_eps_dbscan(
        sample_df, min_samples=2, plot=False, verbose=False
    )
    _, eps_high = search_eps_dbscan(
        sample_df, min_samples=10, plot=False, verbose=False
    )
    assert eps_low != eps_high


def test_empty_dataframe_raises():
    """
    Test that search_eps_dbscan raises a ValueError when given an empty
    DataFrame.

    The test checks that the function raises a ValueError with an appropriate
    error message when given an empty DataFrame as input. This ensures that
    the function does not crash silently when given invalid input.
    """
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        search_eps_dbscan(df_empty, plot=False, verbose=False)


def test_all_nan_columns_removed():
    """
    Test that search_eps_dbscan removes columns with all NaN values.

    This test verifies that columns containing only NaN values are
    removed from the DataFrame before DBSCAN is applied. It ensures that
    the results contain the expected columns and that a valid eps value
    is returned.

    Asserts:
        - The 'results' DataFrame contains the 'eps' column.
        - The 'best_eps' value is greater than 0.
    """

    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan], "c": [4, 5, 6]}
    )
    results, best_eps = search_eps_dbscan(df, plot=False, verbose=False)
    assert "eps" in results.columns
    assert best_eps > 0


def test_works_with_categorical_features():
    """
    Test that search_eps_dbscan works with DataFrames containing categorical
    features.

    This test verifies that the function works as expected when given a
    DataFrame containing categorical features. It checks that a valid eps
    value is returned and that the results contain the 'eps' column.

    Asserts:
        - The 'results' DataFrame contains the 'eps' column.
        - The 'best_eps' value is greater than 0.
    """

    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], size=100),
        }
    )
    results, best_eps = search_eps_dbscan(df, plot=False, verbose=False)
    assert isinstance(results, pd.DataFrame)
    assert best_eps > 0


def test_no_outliers_removed_on_dense_data():
    """Dense clusters shouldn't have outliers with large enough eps"""
    X, _ = make_blobs(
        n_samples=100, centers=3, cluster_std=0.5, random_state=42
    )
    df = pd.DataFrame(X, columns=["feature1", "feature2"])

    cleaned = clean_outliers_using_dbscan(
        df, eps=1.5, min_samples=5, distance_metric="euclidean"
    )
    assert len(cleaned) == len(df)  # No rows should be removed


def test_outliers_are_removed():
    """Should remove obvious outliers"""
    # Add outliers manually
    X, _ = make_blobs(
        n_samples=100, centers=1, cluster_std=0.3, random_state=42
    )
    df = pd.DataFrame(X, columns=["feature1", "feature2"])
    df.loc[100] = [10, 10]  # extreme outlier
    df.loc[101] = [15, -10]  # another outlier

    cleaned = clean_outliers_using_dbscan(
        df, eps=0.5, min_samples=5, distance_metric="euclidean"
    )
    assert len(cleaned) < len(df)
    assert not cleaned.isin([[10, 10], [15, -10]]).any().any()


def test_with_categorical_data():
    """Should handle categorical variables"""
    df = pd.DataFrame(
        {"feature1": [1, 2, 3, 100], "feature2": ["A", "A", "B", "B"]}
    )
    cleaned = clean_outliers_using_dbscan(
        df, eps=0.5, min_samples=2, distance_metric="manhattan"
    )
    assert len(cleaned) < len(df)


def test_verbose_output(capsys):
    """Test that verbose mode prints output"""
    X, _ = make_blobs(
        n_samples=50, centers=1, cluster_std=0.3, random_state=42
    )
    df = pd.DataFrame(X, columns=["feature1", "feature2"])
    df.loc[50] = [10, 10]
    clean_outliers_using_dbscan(
        df, eps=0.5, min_samples=5, distance_metric="euclidean", verbose=True
    )
    captured = capsys.readouterr()
    assert "Removing" in captured.out
