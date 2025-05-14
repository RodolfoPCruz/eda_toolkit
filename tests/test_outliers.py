"""
Tests for the outliers module
"""

import pandas as pd
from eda_toolkit.outliers import calculate_outlier_threshold, clean_outliers


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


def test_clean_outliers_ignore_non_numeric_and_constant(
    sample_data_outliers, caplog
):
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
