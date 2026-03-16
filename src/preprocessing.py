"""
preprocessing.py
----------------
Common preprocessing helpers used across experiments.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Split data into train, validation, and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float
        Fraction reserved for the held-out test set.
    val_size : float
        Fraction of the *remaining* data reserved for validation.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on the training set and apply it to all splits.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to integers.

    Returns
    -------
    y_encoded : np.ndarray
    encoder : LabelEncoder
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder
