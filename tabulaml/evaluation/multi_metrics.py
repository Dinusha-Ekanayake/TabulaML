"""Metrics for several targets at once (multi-label / multi-output).

Every target is scored on its own and the scores are AVERAGED over targets, so the usual
names (accuracy, F1 macro, RMSE, R², ...) keep their meaning: "the average target". Two
extra classification metrics describe the targets together:

* ``subset_accuracy``: share of rows where EVERY target is right (strict);
* ``hamming_loss``: share of individual target values that are wrong (lower is better).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from tabulaml.core.types import ProblemType


def _columns(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    return arr.reshape(len(arr), -1)


def subset_accuracy(y_true: Any, y_pred: Any) -> float:
    t, p = _columns(y_true), _columns(y_pred)
    return float(np.mean(np.all(t == p, axis=1)))


def hamming_loss(y_true: Any, y_pred: Any) -> float:
    t, p = _columns(y_true), _columns(y_pred)
    return float(np.mean(t != p))


def _mean_over_targets(fn: Any, y_true: Any, y_pred: Any, **kwargs: Any) -> float:
    t, p = _columns(y_true), _columns(y_pred)
    return float(np.mean([fn(t[:, j], p[:, j], **kwargs) for j in range(t.shape[1])]))


def mean_accuracy(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(accuracy_score, y_true, y_pred)


def mean_balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(balanced_accuracy_score, y_true, y_pred)


def mean_f1_macro(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(f1_score, y_true, y_pred, average="macro", zero_division=0)


def mean_rmse(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(root_mean_squared_error, y_true, y_pred)


def mean_mae(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(mean_absolute_error, y_true, y_pred)


def mean_r2(y_true: Any, y_pred: Any) -> float:
    return _mean_over_targets(r2_score, y_true, y_pred)


_CLASSIFICATION = {
    "f1_macro": mean_f1_macro,
    "accuracy": mean_accuracy,
    "balanced_accuracy": mean_balanced_accuracy,
    "subset_accuracy": subset_accuracy,
    "hamming_loss": hamming_loss,
}
_REGRESSION = {"rmse": mean_rmse, "mae": mean_mae, "r2": mean_r2}
_LOWER = {"hamming_loss", "rmse", "mae"}


def multi_metric_functions(problem_type: ProblemType) -> dict[str, Any]:
    return dict(_REGRESSION if problem_type == ProblemType.REGRESSION else _CLASSIFICATION)


def multi_output_metrics(problem_type: ProblemType, y_true: Any, y_pred: Any) -> dict[str, float]:
    """Averaged metrics for one evaluation set with several targets."""
    out = {}
    for name, fn in multi_metric_functions(problem_type).items():
        try:
            out[name] = float(fn(y_true, y_pred))
        except ValueError:  # e.g. a fold where one target has a single class
            out[name] = float("nan")
    return out


def multi_scoring_map(problem_type: ProblemType) -> dict[str, Any]:
    """sklearn scorers (errors negated, the sklearn convention) for tuning and importance."""
    return {
        name: make_scorer(fn, greater_is_better=name not in _LOWER)
        for name, fn in multi_metric_functions(problem_type).items()
    }


def per_target_metrics(
    problem_type: ProblemType, y_true: Any, y_pred: Any, targets: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Plain metrics for each target on its own (shown in a per-target table)."""
    t, p = _columns(y_true), _columns(y_pred)
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(targets):
        if problem_type == ProblemType.REGRESSION:
            out[name] = {
                "rmse": float(root_mean_squared_error(t[:, j], p[:, j])),
                "mae": float(mean_absolute_error(t[:, j], p[:, j])),
                "r2": float(r2_score(t[:, j], p[:, j])),
            }
        else:
            out[name] = {
                "accuracy": float(accuracy_score(t[:, j], p[:, j])),
                "balanced_accuracy": float(balanced_accuracy_score(t[:, j], p[:, j])),
                "f1_macro": float(f1_score(t[:, j], p[:, j], average="macro", zero_division=0)),
            }
    return out
