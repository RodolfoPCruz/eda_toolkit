"""
Module to calculate and crete plots of
bivariate statistics of features in a pandas dataframe.
"""
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from eda_toolkit.utils.data_loader import load_csv_from_data


def compute_regression_statistics(
    df: pd.DataFrame, column_1: str, column_2: str, round_to: int = 3
) -> dict:
    """
    Compute regression and correlation statistics between two numeric columns.

    Returns:
        dict: Dictionary containing regression slope, intercept, p-values,
        correlations, and skewness.
    """
    result_lin_regression = stats.linregress(df[column_1], df[column_2])
    res_spearman = stats.spearmanr(df[column_1], df[column_2])
    res_kendall = stats.kendalltau(df[column_1], df[column_2])

    return {
        "slope": round(result_lin_regression.slope, round_to),
        "intercept": round(result_lin_regression.intercept, round_to),
        "pearson_r": round(result_lin_regression.rvalue, round_to),
        "pearson_p": round(result_lin_regression.pvalue, round_to),
        "spearman": round(res_spearman.statistic, round_to),
        "spearman_p": round(res_spearman.pvalue, round_to),
        "kendall": round(res_kendall.statistic, round_to),
        "kendall_p": round(res_kendall.pvalue, round_to),
        "skew_1": round(df[column_1].skew(), round_to),
        "skew_2": round(df[column_2].skew(), round_to),
    }


def generate_regression_plot(
    df: pd.DataFrame, column_1: str, column_2: str, round_to: int = 3
):
    """
    Generate a linear regression plot between two numeric columns
        in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input data.
        column_1 (str): Column name for the X-axis.
        column_2 (str): Column name for the Y-axis.
        round_to (int): Decimal precision for statistical results.

    Returns:
        None
    """
    df = df[[column_1, column_2]].dropna()

    if not (
        pd.api.types.is_numeric_dtype(df[column_1])
        and pd.api.types.is_numeric_dtype(df[column_2])
    ):
        raise TypeError("Both columns must be numeric.")

    stats_summary = compute_regression_statistics(
        df, column_1, column_2, round_to
    )

    # Plot
    sns.regplot(
        data=df, x=column_1, y=column_2, line_kws={"color": "darkorange"}
    )
    plt.xlabel(column_1)
    plt.ylabel(column_2)
    plt.title(f"Regression plot: {column_1} vs {column_2}")

    # Text summary
    text_str = (
        f"y = {stats_summary['slope']}x + {stats_summary['intercept']}\n"
        f"Pearson r = {stats_summary['pearson_r']}, "
        f"p = {stats_summary['pearson_p']}\n"
        f"Spearman ρ = {stats_summary['spearman']}, "
        f"p = {stats_summary['spearman_p']}\n"
        f"Kendall τ = {stats_summary['kendall']}, "
        f"p = {stats_summary['kendall_p']}\n"
        f"Skewness {column_1} = {stats_summary['skew_1']}\n"
        f"Skewness {column_2} = {stats_summary['skew_2']}"
    )

    plt.gca().text(
        0.95,
        0.9,
        text_str,
        fontsize=10,
        ha="right",
        va="center",
        transform=plt.gca().transAxes,
        bbox={'boxstyle':"round,pad=0.3",
               'facecolor':"white", 
               'edgecolor':"gray"})
    # plt.text(0.95, 0.3, text_str, fontsize = 12,
    #         transform = plt.gcf().transFigure)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    insurance = load_csv_from_data("insurance/insurance.csv")
    generate_regression_plot(insurance, "bmi", "charges")
