"""
evaluation/metrics.py
----------------------
Classification metrics for evaluating fake news detection performance.

Metrics computed (§5.2):
    - Accuracy     : (TP + TN) / (TP + TN + FP + FN)
    - Precision    : TP / (TP + FP)
    - Recall       : TP / (TP + FN)
    - F1-Score     : 2 · Precision · Recall / (Precision + Recall)
    - ROC-AUC      : Area under the ROC curve (requires probability scores)
    - Macro F1     : Mean F1 across all classes (for multi-class extension)

Paper's reported results:
    Accuracy = 95.08%, Precision = 94.2%, Recall = 95.6%, F1 = 94.8%
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_prob:  Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute the full set of classification metrics.

    Args:
        y_true  : (N,) ground-truth binary labels {0, 1}.
        y_pred  : (N,) predicted binary labels {0, 1}.
        y_prob  : (N,) predicted probabilities for the Fake class (class 1).
                  Required for ROC-AUC computation.
        average : Averaging strategy for multi-class metrics ('binary', 'macro').

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc (if y_prob given).
    """
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError as e:
            logger.warning("ROC-AUC computation failed: %s", e)
            metrics["roc_auc"] = float("nan")

    # Confusion matrix components
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            "true_positive":  int(tp),
            "true_negative":  int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        })

    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dict as a human-readable string for logging."""
    parts = []
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if key in metrics:
            parts.append(f"{prefix}{key}={metrics[key]:.4f}")
    return " | ".join(parts)


class MetricsTracker:
    """
    Tracks classification metrics across multiple training rounds.

    Useful for logging per-round performance curves (Figure 4 in the paper).
    """

    def __init__(self):
        self._history: List[Dict[str, float]] = []

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        round_num: int = -1,
    ) -> Dict[str, float]:
        """Compute and record metrics for one evaluation point."""
        metrics           = compute_metrics(y_true, y_pred, y_prob)
        metrics["round"]  = round_num
        self._history.append(metrics)
        return metrics

    @property
    def best_accuracy(self) -> float:
        """Return the highest accuracy recorded across all rounds."""
        if not self._history:
            return 0.0
        return max(m["accuracy"] for m in self._history)

    @property
    def best_f1(self) -> float:
        if not self._history:
            return 0.0
        return max(m["f1"] for m in self._history)

    def get_history(self, key: str) -> List[float]:
        """Return time series of a specific metric."""
        return [m.get(key, float("nan")) for m in self._history]

    def summary(self) -> Dict[str, float]:
        """Return final-round metrics."""
        if not self._history:
            return {}
        return {k: v for k, v in self._history[-1].items() if isinstance(v, float)}
