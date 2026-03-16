"""
evaluation.py
-------------
Helpers for evaluating and comparing classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
) -> dict:
    """Compute a standard set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities (required for AUC).
    average : str
        Averaging strategy for multi-class metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc (if y_prob provided).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["auc"] = roc_auc_score(y_true, prob)
        else:
            metrics["auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=average
            )
    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of {model_name: metrics_dict}.

    Parameters
    ----------
    results : dict
        ``{"ModelName": {"accuracy": 0.95, "f1": 0.94, ...}, ...}``

    Returns
    -------
    pd.DataFrame sorted by F1 score descending.
    """
    df = pd.DataFrame(results).T
    df = df.sort_values("f1", ascending=False)
    return df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels.
    class_names : list[str]
        Labels for each class, in order.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if ``None``.

    Returns
    -------
    ax : matplotlib Axes
    """
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Print sklearn's full classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))
