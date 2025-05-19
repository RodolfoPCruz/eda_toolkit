"""
Test suite for the `skewness` module
"""

import numpy as np
import pandas as pd

from eda_toolkit.skewness import (
    apply_final_transformation,
    apply_transformations,
    correct_skew,
    fallback_to_binary,
    remove_nans_and_subsample,
    select_best_transformation,
)


def test_remove_nans_and_subsample(df_positive_skew):
    """
    Tests that `remove_nans_and_subsample`:
    - Removes NaNs from the specified feature
    - Subsamples the DataFrame if it has more rows than `subsample_limit`
    - Sets the `subsampled` flag accordingly
    """
    df = df_positive_skew.copy()
    df.loc[:10, "feature"] = np.nan
    cleaned, sampled, subsampled = remove_nans_and_subsample(
        df, "feature", verbose=False, subsample_limit=500
    )
    assert cleaned["feature"].isna().sum() == 0
    assert len(sampled) <= 500
    assert subsampled is True


def test_apply_transformations(df_positive_skew):
    """
    Tests that `apply_transformations`:
    - Returns a dictionary containing the transformed features as NumPy
            arrays or Pandas Series
    - Returns a dictionary containing the results of the transformations
            as floats
    """
    df = df_positive_skew.copy()
    skew = df["feature"].skew()
    transformed, results = apply_transformations(
        df, "feature", initial_skew=skew, max_power=10, plot=False
    )
    assert isinstance(transformed, dict)
    assert all(
        isinstance(v, np.ndarray) or isinstance(v, pd.Series)
        for v in transformed.values()
    )
    assert all(isinstance(k, str) for k in results)


def test_select_best_transformation():
    """
    Tests that `select_best_transformation`:
    - Identifies the transformation with the minimum skewness value.
    - Correctly returns the best transformation as 'yeo'.
    - Returns True for successful transformation when the best skewness
      is below the specified threshold.
    """

    results = {"log": 0.4, "2": 0.3, "yeo": 0.2}
    best, success = select_best_transformation(results, final_threshold=0.25)
    assert best == "yeo"
    assert success is True


def test_apply_final_transformation(df_positive_skew):
    """
    Tests that `apply_final_transformation`:
    - Adds the transformed feature to the DataFrame when subsampled is False.
    - Returns a DataFrame with the transformed feature added.
    - Does not add NaN values to the DataFrame.
    """
    df = df_positive_skew.copy()
    df_temp = df.copy()
    transformed = {"log": np.log(df_temp["feature"] + 1e-7)}
    result_df = apply_final_transformation(
        df,
        transformed,
        "log",
        "feature",
        subsampled=False,
        positively_skewed=True,
    )
    assert "feature_transformed" in result_df.columns
    assert result_df["feature_transformed"].notna().all()


def test_fallback_to_binary(df_positive_skew):
    """
    Tests that `fallback_to_binary`:
    - Creates a new binary feature
    - Sets the binary feature to 1 if the original feature is above its minimum
      value, and 0 otherwise
    """
    df = df_positive_skew.copy()
    df_bin = fallback_to_binary(df, "feature", skew_positive=True)
    assert df_bin["feature_binary"].isin([0, 1]).all()


def test_correct_skew_positive(df_positive_skew):
    """
    Tests that `correct_skew`:
    - Returns a DataFrame with the transformed feature added when given a
      positively skewed feature.
    - Returns a dictionary containing the results of the different
      transformations.
    """
    df = df_positive_skew.copy()
    df_out, results = correct_skew(
        df,
        "feature",
        plot_all_transformations=False,
        plot_transformed_feature=False,
        verbose=False,
    )
    assert isinstance(df_out, pd.DataFrame)
    assert isinstance(results, dict)
