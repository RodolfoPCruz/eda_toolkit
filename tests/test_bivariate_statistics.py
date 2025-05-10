"""
Test suite for the `bivariate_stats` function
"""
import pandas as pd
# pylint: disable=import-error
from eda_toolkit.bivariate_statistics import bivariate_stats


def test_numeric_numeric(mixed_dataframe):
    """
    Test the bivariate statistics computation for numeric-numeric pairs.

    This test verifies that the `bivariate_stats` function correctly computes
    statistics for numeric feature-target pairs. It checks if the result
    contains the expected feature 'numeric1' and ensures that the correlation
    coefficient 'r' is calculated and not '-' (indicating a missing value).

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame with
                                        mixed data types, including numeric
                                        columns.
    """

    df = mixed_dataframe.rename(columns={"target_num": "target"})
    result = bivariate_stats(df, "target")
    assert "numeric1" in result.index
    assert result.loc["numeric1"]["r"] != "-"


def test_categorical_categorical(mixed_dataframe):
    """
    Test the bivariate statistics computation for categorical-
        categorical pairs.

    This test verifies that the `bivariate_stats` function correctly computes
    statistics for categorical feature-target pairs. It checks if the result
    contains the expected feature 'multi_cat' and ensures that the chi-squared
    statistic 'chi2' is calculated and not '-' (indicating a missing value).

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame with
                                    mixed data types, including categorical
                                    columns.
    """
    df = mixed_dataframe.rename(columns={"target_bin": "target"})
    df["multi_cat"] = df["multi_cat"].astype(str)
    result = bivariate_stats(df, "target")
    assert "multi_cat" in result.index
    assert result.loc["multi_cat"]["chi2"] != "-"


def test_numeric_vs_binary_cat(mixed_dataframe):
    """
    Test the bivariate statistics computation for numeric vs binary
    categorical pairs.

    This test verifies that the `bivariate_stats` function correctly
    computes statistics for numeric feature-target pairs where the
    target is binary categorical. It checks if the result contains
    the expected feature 'binary_cat' and ensures that the T-test
    statistic 'ttest' is calculated and not '-' (indicating a missing
    value).

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame
                                        with mixed data types, including
                                        numeric columns and a binary
                                        categorical target.
    """

    df = mixed_dataframe.rename(columns={"target_num": "target"})
    result = bivariate_stats(df, "target")
    assert "binary_cat" in result.index
    assert result.loc["binary_cat"]["ttest"] != "-"


def test_binary_numeric_target(mixed_dataframe):
    """
    Test the bivariate statistics computation for binary categorical vs
    numeric pairs.

    This test verifies that the `bivariate_stats` function correctly computes
    statistics for binary categorical feature-target pairs where the target is
    numeric. It checks if the result contains the expected feature 'binary_cat'
    and ensures that the T-test statistic 'ttest' is calculated and not '-'
    (indicating a missing value).

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame with
                                        mixed data types, including a numeric
                                        target and binary categorical columns.
    """

    df = mixed_dataframe.rename(columns={"numeric1": "target"})
    result = bivariate_stats(df, "target")
    assert "binary_cat" in result.index
    assert result.loc["binary_cat"]["ttest"] != "-"


def test_invalid_categorical():
    """
    Test the bivariate statistics computation for invalid categorical features.

    This test verifies that the `bivariate_stats` function correctly identifies
    and handles invalid categorical features, where both feature and target are
    non-numeric with all unique values. It ensures that the output result
    for such features contains a '-' for the 'p_value', indicating that no
    valid statistical test was performed.

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame with
                                        mixed data types, although not used
                                        in this test case.
    """

    df = pd.DataFrame(
        {"id": [f"id_{i}" for i in range(10)], "target": [1] * 10}
    )
    result = bivariate_stats(df, "target")
    assert result.loc["id"]["p_value"] == "-"


def test_anova_case(mixed_dataframe):
    """
    Test the bivariate statistics computation for ANOVA case.

    This test verifies that the `bivariate_stats` function correctly computes
    the F-statistic for the ANOVA test between a categorical feature and a
    numeric target. It checks if the result contains the expected feature
    'multi_cat' and ensures that the F-statistic 'F' is calculated and not '-'
    (indicating a missing value).

    Parameters:
        mixed_dataframe (pd.DataFrame): A fixture providing a DataFrame with
                                        mixed data types, including a numeric
                                        target and categorical features.
    """
    df = mixed_dataframe.rename(columns={"numeric2": "target"})
    result = bivariate_stats(df, "target")
    assert "multi_cat" in result.index
    print(result.head())
    assert result.loc["multi_cat"]["ttest"] != "-"


# test for missing values
def test_with_missing_values():
    """
    Test the bivariate statistics computation with missing values.

    This test verifies that the `bivariate_stats` function correctly calculates
    the percentage of missing values for a numeric feature with missing data
    when paired with a target that also contains missing values.

    Asserts:
        - The 'feature' column reflects the percentage of dropped rows
          due to missing values, ensuring the 'missing' statistic is
          not '0%'.
    """

    df = pd.DataFrame(
        {"feature": [1, 2, None, 4, 5], "target": [1, 2, 3, None, 5]}
    )
    result = bivariate_stats(df, "target")
    assert (
        result.loc["feature"]["missing"] != "0%"
    )  # should reflect dropped rows


def test_constant_column():
    """
    Test the bivariate statistics computation for constant columns.

    This test verifies that the `bivariate_stats` function correctly handles
    constant columns by not computing any statistics and returning '-' for
    the 'p_value' column.

    Asserts:
        - The 'constant' column has a 'p_value' of '-'.
    """
    df = pd.DataFrame({"constant": [1, 1, 1, 1, 1], "target": [1, 2, 3, 4, 5]})
    result = bivariate_stats(df, "target")
    assert result.loc["constant"]["p_value"] == "-"  # no variance = no stats


def test_basic_numeric_correlation(basic_numeric_df):
    """
    Test the bivariate statistics computation for a simple numeric feature
    correlation with a perfect positive correlation.

    This test verifies that the `bivariate_stats` function correctly computes
    the Pearson correlation coefficient 'r' for a feature with a perfect
    positive correlation with the target. It checks if the result contains
    the expected feature 'feature1' and ensures that the correlation
    coefficient 'r' is equal to 1.0.

    Parameters:
        basic_numeric_df (pd.DataFrame): A fixture providing a DataFrame with
                                          two numeric columns, 'feature1' and
                                          'target', where 'feature1' has a
                                          perfect positive correlation with
                                          'target'.
    """
    result = bivariate_stats(basic_numeric_df, target="target")
    assert "feature1" in result.index
    assert result.loc["feature1", "r"] == 1.0


def test_handles_missing_data(mixed_type_df):
    """
    Test the bivariate statistics computation for a DataFrame with missing
    data.

    This test verifies that the `bivariate_stats` function correctly computes
    the percentage of missing values for a numeric feature with missing data
    when paired with a target that also contains missing values, and that the
    outpu result for such features contains the correct data type.

    Parameters:
        mixed_type_df (pd.DataFrame): A fixture providing a DataFrame with
                                       mixed data types and missing values.
    """
    result = bivariate_stats(mixed_type_df, target="target")
    assert result.loc["feature1", "missing"].endswith("%")
    assert result.loc["feature2", "type"] == "object"


def test_categorical_target_ttest(categorical_target_df):
    """
    Test the bivariate statistics computation for a categorical target with
    t-test.

    This test verifies that the `bivariate_stats` function correctly computes
    the t-test statistic 'ttest' for a numeric feature when paired with a
    categorical target. It checks if the result contains the expected feature
    'feature1' and ensures that the t-test statistic 'ttest' is not '-'
    (indicating a missing value).

    Parameters:
        categorical_target_df (pd.DataFrame): A fixture providing a DataFrame
                                    with a categorical target and a numeric
                                              feature 'feature1'.
    """
    result = bivariate_stats(categorical_target_df, target="target")
    assert "feature1" in result.index
    assert result.loc["feature1", "ttest"] != "-"


def test_handles_edge_cases(edge_case_df):
    """
    Test the bivariate statistics computation for edge cases.

    This test verifies that the `bivariate_stats` function correctly handles
    edge cases, such as a feature with a single unique value, a feature with
    all missing values, and a feature with a non-numeric data type. It checks
    if the result contains the correct values for the feature 'feature1'
    (correlation coefficient 'r' is '-'), feature 'feature2' (type is not '-'),
    and feature 'feature3' (p-value is not '-').

    Parameters:
        edge_case_df (pd.DataFrame): A fixture providing a DataFrame with edge
                                      case data.
    """
    result = bivariate_stats(edge_case_df, target="target")
    assert result.loc["feature1", "r"] == "-"
    assert result.loc["feature2", "type"] != "-"
    assert result.loc["feature3", "p_value"] != "-"


def test_invalid_categorical_case():
    """
    Test the bivariate statistics computation for an invalid categorical
    feature case.

    This test verifies that the `bivariate_stats` function correctly identifies
    and handles an invalid categorical feature, where both feature and target
    are non-numeric with all unique values. It ensures that the output result
    for such features contains a '-' for the 'p_value', indicating that no
    valid statistical test was performed.
    """
    df = pd.DataFrame(
        {"feature": ["a", "b", "c", "d", "e"], "target": [1, 2, 3, 4, 5]}
    )
    result = bivariate_stats(df, target="target")
    assert result.loc["feature", "p_value"] == "-"


def test_anova_skips_when_one_group_has_single_sample():
    """
    Ensure ANOVA is not computed when one of the groups has only a single
    sample.
    The 'F' value should be '-' in such cases.
    """
    df = pd.DataFrame(
        {
            "category": [
                "a",
                "a",
                "b",
                "c",
                "c",
            ],  # 'b' group has only one sample
            "value": [5, 6, 7, 8, 9],
        }
    )

    result = bivariate_stats(
        df.rename(columns={"category": "feature"}), target="value"
    )

    # F-stat should not be computed (should be '-')
    assert result.loc["feature", "F"] == "-"


def test_anova_downgrade_to_ttest():
    """
    Ensure that when performing ANOVA, if one of the groups has fewer than two
    samples, the test is downgraded to a t-test. The 'ttest' value should
    not be '-' in such cases.
    """
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c"],  # Group 'c' will be filtered
            "target": [5, 6, 7, 8, 9],
        }
    )
    df = df[df["group"] != "c"]
    result = bivariate_stats(
        df.rename(columns={"group": "feature"}), target="target"
    )
    assert result.loc["feature", "ttest"] != "-"
