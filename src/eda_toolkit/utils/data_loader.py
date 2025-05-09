"""
Load files from the data directory
"""

from pathlib import Path

import pandas as pd


def load_csv_from_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the `data/` directory at the project root.

    Args:
        filename (str): Name of the CSV file (e.g., 'raw/my_data.csv').

    Returns:
        pd.DataFrame: The loaded data.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]  # Adjust as needed
    data_path = project_root / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    return pd.read_csv(data_path)
