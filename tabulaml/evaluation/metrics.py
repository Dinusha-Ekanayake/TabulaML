"""Metric definitions shared by cross-validation (scorers) and test-set evaluation.

All reported values are "display oriented": error metrics (MAE, RMSE, ...) are positive and
lower is better; see ``LOWER_IS_BETTER`` in ``core.types``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from tabulaml.core.types import LOWER_IS_BETTER, ProblemType

BINARY_F1_IMBALANCE = 0.35

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC (avg. precision)",
    "mcc": "MCC",
    "log_loss": "Log-loss",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "f1_weighted": "F1 (weighted)",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "medae": "Median AE",
    "mape": "MAPE",
}


def greater_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def minority_label(y: Any) -> int:
    counts = np.bincount(np.asarray(y, dtype=int))
    return int(np.argmin(np.where(counts > 0, counts, np.iinfo(np.int64).max)))


def default_primary_metric(problem_type: ProblemType, y: Any) -> str:
    if problem_type == ProblemType.REGRESSION:
        return "rmse"
    counts = np.bincount(np.asarray(y, dtype=int))
    if len(counts[counts > 0]) > 2:
        return "f1_macro"
    return "f1" if counts.min() / counts.sum() < BINARY_F1_IMBALANCE else "roc_auc"


def _no_zero_targets(y: Any) -> bool:
    return bool(np.all(np.asarray(y, dtype=float) != 0))


def scoring_map(problem_type: ProblemType, y: Any) -> dict[str, Any]:
    """sklearn scorers keyed by metric name; error metrics are negated (sklearn convention)."""
    if problem_type == ProblemType.REGRESSION:
        scorers = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "medae": "neg_median_absolute_error",
        }
        if _no_zero_targets(y):
            scorers["mape"] = "neg_mean_absolute_percentage_error"
        return scorers
    n_classes = len(np.unique(y))
    common = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "mcc": make_scorer(matthews_corrcoef),
        "log_loss": "neg_log_loss",
    }
    if n_classes > 2:
        return common | {
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted",
            "roc_auc": "roc_auc_ovr",
        }
    pos = minority_label(y)
    return common | {
        "precision": make_scorer(precision_score, pos_label=pos, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=pos, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=pos, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(
            average_precision_score, response_method="predict_proba", pos_label=pos
        ),
    }


def display_value(metric: str, raw: float) -> float:
    """Convert a scorer value to display orientation (positive errors)."""
    return -raw if metric in LOWER_IS_BETTER else raw


def _regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
    }
    if _no_zero_targets(y_true):
        out["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    return out


def _probability_metrics(y_true: np.ndarray, proba: np.ndarray, pos: int) -> dict[str, float]:
    labels = np.arange(proba.shape[1])
    proba = proba / proba.sum(axis=1, keepdims=True)  # float32 boosters drift off 1.0
    out = {"log_loss": log_loss(y_true, proba, labels=labels)}
    if len(np.unique(y_true)) < 2:
        return out
    if proba.shape[1] == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        out["pr_auc"] = average_precision_score(y_true == pos, proba[:, pos])
    else:
        out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=labels)
    return out


def _classification(
    y_true: np.ndarray, y_pred: np.ndarray, binary: bool, pos: int
) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if binary:
        kw = {"pos_label": pos, "zero_division": 0}
        out |= {
            "precision": precision_score(y_true, y_pred, **kw),
            "recall": recall_score(y_true, y_pred, **kw),
            "f1": f1_score(y_true, y_pred, **kw),
        }
    else:
        out |= {
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    return out


def compute_metrics(
    problem_type: ProblemType,
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    y_train: Any = None,
) -> dict[str, float]:
    """Test-set metrics. ``y_train`` fixes the positive class (minority of the training data)."""
    y_true = np.asarray(y_true)
    if problem_type == ProblemType.REGRESSION:
        return {
            k: float(v)
            for k, v in _regression(y_true.astype(float), np.asarray(y_pred, float)).items()
        }
    reference = y_train if y_train is not None else y_true
    n_classes = int(np.asarray(reference).max()) + 1
    pos = minority_label(reference)
    out = _classification(y_true, np.asarray(y_pred), n_classes == 2, pos)
    if y_proba is not None:
        out |= _probability_metrics(y_true, np.asarray(y_proba, dtype=float), pos)
    return {k: float(v) for k, v in out.items()}
