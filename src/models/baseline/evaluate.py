"""
Model Evaluation Module for Baseline Forecast Bust Classifier.
Computes ROC-AUC, PR-AUC, Precision, Recall, F1-Score, Brier Score, and Confusion Matrix.
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


def evaluate_bust_predictions(
    y_true: np.ndarray | list,
    y_pred_proba: np.ndarray | list,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive classification metrics for forecast bust predictions.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0

    precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    brier = float(brier_score_loss(y_true, y_pred_proba))

    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    metrics = {
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

    return metrics


def print_evaluation_summary(metrics: dict, dataset_name: str = "Test"):
    """Format and print clean evaluation metrics table."""
    print(f"\n==================================================")
    print(f"EVALUATION METRICS SUMMARY: {dataset_name.upper()} SET")
    print(f"==================================================")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC (Avg Prec):  {metrics['pr_auc']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1_score']:.4f}")
    print(f"  Brier Score:        {metrics['brier_score']:.4f} (Lower is better probability calibration)")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN: {cm['true_negatives']:,}  |  FP: {cm['false_positives']:,}")
    print(f"    FN: {cm['false_negatives']:,}  |  TP: {cm['true_positives']:,}")


if __name__ == "__main__":
    y_test = [0, 0, 0, 1, 1, 0, 1, 0]
    y_probs = [0.1, 0.2, 0.3, 0.8, 0.9, 0.4, 0.7, 0.2]
    m = evaluate_bust_predictions(y_test, y_probs)
    print_evaluation_summary(m)
