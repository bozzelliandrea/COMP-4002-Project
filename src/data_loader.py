"""
data_loader.py
--------------
Convenience helpers for loading datasets into pandas DataFrames.
"""

import pandas as pd
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits,
)


def load_sklearn_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load a built-in scikit-learn dataset.

    Parameters
    ----------
    name : str
        One of ``"iris"``, ``"breast_cancer"``, ``"wine"``, or ``"digits"``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    class_names : list[str]
        Human-readable class names.
    """
    loaders = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(loaders)}")

    bunch = loaders[name](as_frame=True)
    X: pd.DataFrame = bunch.frame.drop(columns=["target"])
    y: pd.Series = bunch.target
    class_names: list[str] = list(bunch.target_names)
    return X, y, class_names


def load_csv(path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV file and split it into features and target.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    target_column : str
        Name of the column to use as the prediction target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except the target).
    y : pd.Series
        Target series.
    """
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in {path}")
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y
