# EDA Toolkit

EDA Toolkit is a Python-based series of tools and utilities to assist data scientists and analysts understand and visualize datasets efficiently.

## Features

- **Automated Data Profiling**: Generate comprehensive reports summarizing dataset statistics, missing values, and data distributions.
- **Visualization Tools**: Create a variety of plots (histograms, regression plots, etc) to visualize data distributions and relationships.
- **Data Cleaning Utilities**: Identify and handle outliers, skewed features.
- **Correlation Analysis**: Assess relationships between variables using correlation matrices and heatmaps.
- **Modular Design**: Easily integrate specific components into existing data analysis workflows.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RodolfoPCruz/eda_toolkit.git
cd eda_toolkit
```

2. Set up a virtual environment with your preferred tool (`venv`, `conda`, `pyenv`, etc.) and activate it.

Make sure you're using **Python 3.9 or higher**.

3. Install dependencies:

```
pip install -r requirements.txt
```

## 🧪 Usage Example

## 📁 Project Structure

```
eda_toolkit/
│
├── src/eda_toolkit/          # Core modules
├── notebooks/                # Example notebooks
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
└── setup.py                  # Install config

```

## 📚 Dependencies

This project relies on the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `joblib`

These are listed in `requirements.txt`.

## ✅ Running Tests

Make sure your virtual environment is active, then run:

`pytest tests/`