"""
Module to detect an treat outliers in a pandas dataframe
"""

import logging

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from sklearn import set_config

# from sklearn.cluster import DBSCAN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# from sklearn.preprocessing import MinMaxScaler

from eda_toolkit.utils.data_loader import load_csv_from_data
from eda_toolkit.utils.logger_utils import configure_logging

configure_logging(log_file_name="outliers.log")
logger = logging.getLogger(__name__)


def calculate_outlier_threshold(
    df: pd.DataFrame, feature: str, skew_thresold: float = 1
) -> tuple[float, float]:
    """
    Calculate the inferior and superior thresholds for outlier detection.
    Values above max_thresh and values below min_thresh are considered
    outliers

    Args:
        df (pd.DataFrame): The dataframe containing the feature.
        feature (str): The name of the feature to be processed.
        skew_thresold (float): The skewness threshold used to determine
        whether the distribution is normal or not.

    Returns:
        tuple: A tuple containing the inferior and superior thresholds
        for outlier detection.
    """
    skew = df[feature].skew()
    if skew > -1 * skew_thresold and skew < skew_thresold:
        # empirical rule to detect outliers in normal distributions
        min_thresh = df[feature].mean() - 3 * df[feature].std()
        max_thresh = df[feature].mean() + 3 * df[feature].std()
    else:
        # apply the Tukey rule to detect outliers in distributions
        # that are not normal
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1  # interquartile range
        max_thresh = q3 + 1.5 * (iqr)
        min_thresh = q1 - 1.5 * (iqr)

    return (min_thresh, max_thresh)


def clean_outliers(
    df: pd.DataFrame,
    features_list: list = None,
    outlier_treatment: str = None,
    output_column: str = None,
    skew_thresold: float = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect and treat outliers. Two criteria will be used to detect outliers:
        - empirical rule to detect outliers in normal distributions;
        - Tukey rule to detect outliers in distributions that are not normal.
    A distribution will be considered normal when its skewness
    is between -1 * skew_thresold and skew_thresold.One of the two criteria
    will be used to calculate max_thresh and min_thresh. Values above
    max_thresh and values below min_thresh are considered outliers.


    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be cleaned.
    features_list : list, optional
        The list of features to be cleaned, by default, is None. If None,
        the function will clean all features.
    outlier_treatment : str, optional
        The treatment to be applied to outliers, by default, None.
        Can be 'remove'; 'replace'; 'impute', or None:
            - remove: rows containing outliers are completely removed;
            - replace: outliers above max_thresh and outliers below
              min_thresh are replaced by max_thresh and min_thresh,
              respectively;
            - impute: outliers above max_thresh and outliers below min_thresh
              are imputed using the Iterative imputer method form
              scikit-learn;
            - None: no treatment is applied to outliers.
    output_column : str, optional
        The name of the output column, by default is None. The output column
        will not be used when calculating the values to be imputed.
        If None, it is considered that the output column is not in df.
    skew_thresold : float, optional
        The skewness threshold for a distribution to be considered normal,
        by default 1.

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.

    """
    set_config(transform_output="pandas")

    logging.info(
        "Detecting and treating outliers " "using traditional methods..."
    )

    df = df.copy()
    initial_number_rows = df.shape[0]
    df = df.dropna()
    final_number_rows = df.shape[0]
    removed_rows = initial_number_rows - final_number_rows
    logging.info(
        f"{removed_rows} rows were removed because "
        "they contained NaN values."
    )

    if features_list is None:
        features_list = df.columns.to_list()

    if outlier_treatment not in ["remove", "replace", "impute", None]:
        logging.info(
            f"The outlier_treatment {outlier_treatment} is not "
            "valid. No treatment will be applied to outliers."
        )
        outlier_treatment = None

    # remove output_column from features_list. The output column can not be
    # used when calculating the values to be imputed
    if output_column is not None and output_column in features_list:
        features_list_impute_method = features_list.copy()
        features_list_impute_method.remove(output_column)
    else:
        features_list_impute_method = features_list

    for feature in features_list:
        #the input features can not be used impute values to output feature   
        if feature == output_column and outlier_treatment == 'impute':
            continue
        # test whether the feature is in the dataframe
        if feature in df.columns:
            # only numeric columns will be cleaned
            if pd.api.types.is_numeric_dtype(df[feature]):

                # test whether the feature has only one unique value
                if df[feature].nunique() == 1:
                    logging.info(
                        f"The feature {feature} has only one "
                        "unique value and was therefore ignored."
                    )

                # test whether the feature is binary (only true or
                # false values)
                elif set(df[feature].dropna().unique()).issubset(
                    {0, 1}
                ) or set(df[feature].dropna().unique()).issubset(
                    {True, False}
                ):
                    logging.info(
                        f"The feature {feature} is binary "
                        "(only true or false values) and was "
                        "therefore ignored."
                    )

                # look for outliers
                else:
                    # calculate max_thresh and min_thresh

                    min_thresh, max_thresh = calculate_outlier_threshold(
                        df, feature, skew_thresold
                    )

                    # values above max_thresh and values below min_thresh
                    # are considered outliers
                    count_max_outlier = len(df.loc[df[feature] > max_thresh])
                    count_min_outlier = len(df.loc[df[feature] < min_thresh])
                    logging.info(
                        f"The feature {feature} has "
                        f"{count_max_outlier} "
                        f"values above {max_thresh}"
                    )

                    logging.info(
                        f"The feature {feature} "
                        f"has {count_min_outlier} "
                        f"values below {min_thresh}"
                    )

                    has_outliers = (
                        count_max_outlier > 0 or count_min_outlier > 0
                    )

                    if not has_outliers:
                        continue

                    if outlier_treatment == "remove":
                        df = df[
                            (df[feature] >= min_thresh)
                            & (df[feature] <= max_thresh)
                        ]
                        logging.info(f"{feature}: outliers removed")

                    elif outlier_treatment == "replace":
                        if not pd.api.types.is_float_dtype(df[feature]):
                            df[feature] = df[feature].astype(float)
                        logging.info(f"{feature}: outliers replaced")
                        df.loc[df[feature] > max_thresh, feature] = max_thresh
                        df.loc[df[feature] < min_thresh, feature] = min_thresh

                    elif outlier_treatment == "impute":
                        df_temp = df[features_list_impute_method].copy()
                        df_temp.loc[df_temp[feature] > max_thresh, feature] = (
                            np.nan
                        )
                        df_temp.loc[df_temp[feature] < min_thresh, feature] = (
                            np.nan
                        )

                        df_temp = pd.get_dummies(df_temp, drop_first=True)
                        imputer = IterativeImputer(
                            max_iter=10, random_state=random_state
                        )
                        df_temp = imputer.fit_transform(df_temp)
                        df[feature] = df_temp[feature]

            else:
                logging.info(
                    f"The feature {feature} is not "
                    "numeric and was therefore ignored"
                )

        else:
            logging.info(
                f"A {feature} não foi encontrada "
                "no dataframe e foi ignorada"
            )

    return df


if __name__ == "__main__":
    nba = load_csv_from_data("nba/nba_salaries.csv")
    nba_cleaned = clean_outliers(nba)
