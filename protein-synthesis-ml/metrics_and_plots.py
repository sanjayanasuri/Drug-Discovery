"""
metrics_and_plots.py

Helpers for classification/regression metrics and plots.
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    classification_report,
    r2_score,
    mean_squared_error,
)


def evaluate_classifier(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    threshold: float = 0.5, 
    model_name: str = ""
) -> np.ndarray:
    """
    Print ROC-AUC, classification report, and return y_pred.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    threshold : float
        Decision threshold (default: 0.5)
    model_name : str
        Name of the model for display
        
    Returns
    -------
    np.ndarray
        Binary predictions at the given threshold
    """
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n=== {model_name} (threshold={threshold:.2f}) ===")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return y_pred


def plot_roc_pr(y_true: np.ndarray, y_proba: np.ndarray, title_prefix: str = ""):
    """
    Plot ROC and Precision-Recall curves.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    title_prefix : str
        Prefix for plot titles
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend()
    plt.show()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} Precision-Recall Curve")
    plt.legend()
    plt.show()


def print_threshold_table(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Print precision/recall at different decision thresholds.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    """
    thresholds = np.linspace(0.1, 0.9, 9)
    print("\nThreshold  Precision  Recall")
    for thr in thresholds:
        y_hat = (y_proba >= thr).astype(int)
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        print(f"{thr:8.2f}  {p:9.3f}  {r:6.3f}")


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = ""):
    """
    Print regression metrics: R^2 and RMSE.
    
    Parameters
    ----------
    y_true : np.ndarray
        True continuous values
    y_pred : np.ndarray
        Predicted continuous values
    model_name : str
        Name of the model for display
    """
    r2 = r2_score(y_true, y_pred)
    # Calculate RMSE manually (sqrt of MSE) for compatibility
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n=== {model_name} Regression ===")
    print(f"R^2:   {r2:.4f}")
    print(f"RMSE:  {rmse:.4f}")
