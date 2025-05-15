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
    sample_data_outliers
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

    df_cleaned = clean_outliers(sample_data_outliers, features_list=['binary'], 
                                outlier_treatment='replace')
    assert set(df_cleaned['binary'].unique()).issubset({0, 1})

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
    
    df_cleaned = clean_outliers(sample_data_outliers, 
                                features_list=['normal_dist'], 
                                outlier_treatment='invalid_method')
    assert 'normal_dist' in df_cleaned.columns 

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
    df.loc[0:4, 'normal_dist'] = 1e6  # extreme high values
    df.loc[5:9, 'normal_dist'] = -1e6  # extreme low values

    df_cleaned = clean_outliers(df, features_list=['normal_dist'], 
                                outlier_treatment='impute')

    # Check that there are no extremely high or low values after imputation
    assert df_cleaned['normal_dist'].max() < 1e5
    assert df_cleaned['normal_dist'].min() > -1e5

    # Check that the column still exists and is float
    assert 'normal_dist' in df_cleaned.columns
    assert pd.api.types.is_float_dtype(df_cleaned['normal_dist'])

    # Check that no NaN values are left in the column
    assert df_cleaned['normal_dist'].isna().sum() == 0
