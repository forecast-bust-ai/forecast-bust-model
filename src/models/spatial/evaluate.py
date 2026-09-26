"""
Evaluation & Baseline Comparison Module for Spatial CNN Model.
Computes ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score, and Confusion Matrix
on spatial grid outputs and prints model comparison table.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)


def evaluate_spatial_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics on flattened spatial grid arrays.
    """
    y_true_flat = np.asarray(y_true).flatten().astype(int)
    y_prob_flat = np.asarray(y_pred_proba).flatten().astype(float)
    y_bin_flat = (y_prob_flat >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat)) if len(np.unique(y_true_flat)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true_flat, y_prob_flat)) if len(np.unique(y_true_flat)) > 1 else 0.0

    precision = float(precision_score(y_true_flat, y_bin_flat, zero_division=0))
    recall = float(recall_score(y_true_flat, y_bin_flat, zero_division=0))
    f1 = float(f1_score(y_true_flat, y_bin_flat, zero_division=0))

    brier = float(brier_score_loss(y_true_flat, y_prob_flat))

    cm = confusion_matrix(y_true_flat, y_bin_flat)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "decision_threshold": threshold,
    }


def print_model_comparison_table(xgb_metrics: dict, cnn_metrics: dict):
    """Format and print Markdown comparison table for XGBoost vs Spatial CNN."""
    print("\n==================================================================")
    print("      MODEL COMPARISON TABLE: XGBOOST BASELINE vs SPATIAL CNN     ")
    print("==================================================================")
    print(f"{'Model':<15} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'F1-Score':<9} | {'Brier Score':<11} | {'Recall':<8}")
    print("-" * 72)
    print(
        f"{'XGBoost':<15} | "
        f"{xgb_metrics.get('roc_auc', 0.0):<9.4f} | "
        f"{xgb_metrics.get('pr_auc', 0.0):<9.4f} | "
        f"{xgb_metrics.get('f1_score', 0.0):<9.4f} | "
        f"{xgb_metrics.get('brier_score', 0.0):<11.4f} | "
        f"{xgb_metrics.get('recall', 0.0):<8.4f}"
    )
    print(
        f"{'Spatial CNN':<15} | "
        f"{cnn_metrics.get('roc_auc', 0.0):<9.4f} | "
        f"{cnn_metrics.get('pr_auc', 0.0):<9.4f} | "
        f"{cnn_metrics.get('f1_score', 0.0):<9.4f} | "
        f"{cnn_metrics.get('brier_score', 0.0):<11.4f} | "
        f"{cnn_metrics.get('recall', 0.0):<8.4f}"
    )
    print("==================================================================")
