"""Rough training-time estimate shown before training starts.

The numbers are deliberately simple: a relative cost per model (seconds per fit on 1,000
rows with about 20 features, measured on a laptop), multiplied by the number of fits that
cross-validation, tuning and the final fit need. The estimate is a RANGE, because real
times depend on the machine and the data; its job is to warn about slow combinations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tabulaml.config import CLOUD_MODE, SVM_MAX_ROWS

# Seconds per fit for 1,000 rows and ~20 features (relative costs, rounded).
FIT_COST = {
    "logreg": 0.05,
    "linreg": 0.01,
    "ridge": 0.01,
    "lasso": 0.02,
    "elasticnet": 0.02,
    "dt": 0.02,
    "rf": 0.3,
    "extratrees": 0.2,
    "bagging": 0.3,
    "adaboost": 0.2,
    "gb": 0.4,
    "histgb": 0.15,
    "xgb": 0.15,
    "lgbm": 0.1,
    "catboost": 0.6,
    "knn": 0.05,
    "mlp": 0.8,
    "svm": 0.3,
    "hybrid_rf_ann": 1.2,
    "naive": 0.0,
    "seasonal_naive": 0.0,
}
DEFAULT_COST = 0.3
PREPROCESS_COST = 0.03  # cleaner + column transformer, per fit per 1k rows
STACKING_INNER_FOLDS = 5
KNN_IMPUTE_SLOW_ROWS = 20_000
QUICK_SECONDS, MODERATE_SECONDS = 30, 180
OVERHEAD_SECONDS = 2.0
CLOUD_FACTOR = 2.0


@dataclass(frozen=True)
class TrainingEstimate:
    seconds_low: float
    seconds_high: float
    level: str  # quick | moderate | long
    per_model: dict[str, float]  # central estimate per model key (seconds)
    warnings: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{_fmt(self.seconds_low)} to {_fmt(self.seconds_high)}"


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min" if seconds >= 120 else f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _fit_cost(key: str, rows: int, size: float, bases: Mapping[str, Sequence[str]]) -> float:
    """Seconds for ONE fit of model ``key`` on ``rows`` rows (``size`` = feature factor)."""
    per_k = rows / 1000
    if key in ("voting", "stacking"):
        inner = sum(_fit_cost(b, rows, size, bases) for b in bases.get(key, ("rf", "xgb", "lgbm")))
        return inner * (STACKING_INNER_FOLDS + 1 if key == "stacking" else 1)
    if key == "svm":
        return FIT_COST["svm"] * per_k**2 * size  # kernel SVM grows roughly with rows squared
    return (FIT_COST.get(key, DEFAULT_COST) + PREPROCESS_COST) * per_k * size


def estimate_training(
    n_rows: int,
    n_features: int,
    model_keys: Sequence[str],
    cv_folds: int,
    tuning: Mapping[str, int] | None = None,
    ensemble_bases: Mapping[str, Sequence[str]] | None = None,
    knn_imputation: bool = False,
    n_targets: int = 1,
) -> TrainingEstimate:
    """Estimated wall time for training ``model_keys`` with the given settings."""
    tuning, bases = tuning or {}, ensemble_bases or {}
    size = max(1.0, n_features / 20) ** 0.7 * max(1, n_targets)
    fold_rows = int(n_rows * (cv_folds - 1) / cv_folds)
    per_model: dict[str, float] = {}
    for key in model_keys:
        fits = (tuning.get(key, 0) + 1) * cv_folds
        per_model[key] = fits * _fit_cost(key, fold_rows, size, bases) + _fit_cost(
            key, n_rows, size, bases
        )
    factor = CLOUD_FACTOR if CLOUD_MODE else 1.0
    central = (OVERHEAD_SECONDS + sum(per_model.values())) * factor
    level = (
        "quick" if central < QUICK_SECONDS else "moderate" if central < MODERATE_SECONDS else "long"
    )
    notes = _warnings(n_rows, model_keys, tuning, knn_imputation, per_model, central)
    return TrainingEstimate(central * 0.5, central * 2.0, level, per_model, tuple(notes))


def _warnings(
    n_rows: int,
    keys: Sequence[str],
    tuning: Mapping[str, int],
    knn_imputation: bool,
    per_model: Mapping[str, float],
    central: float,
) -> list[str]:
    out = []
    if "stacking" in keys:
        out.append(
            "Stacking refits all its base models inside every fold, so it is the slowest choice."
        )
    if tuning:
        extra = sum(tuning.values())
        out.append(f"Hyperparameter search tries {extra} extra combinations, each with full CV.")
    if "svm" in keys and n_rows > SVM_MAX_ROWS:
        out.append(f"SVM will be skipped: it is limited to {SVM_MAX_ROWS:,} training rows.")
    elif "svm" in keys and n_rows > SVM_MAX_ROWS // 2:
        out.append("SVM gets much slower as rows grow (roughly rows squared).")
    if knn_imputation and n_rows > KNN_IMPUTE_SLOW_ROWS:
        out.append("KNN imputation is slow on large data; median imputation is much faster.")
    if CLOUD_MODE:
        out.append("Running on Streamlit Cloud: shared CPUs make training about twice as slow.")
    if central >= MODERATE_SECONDS and per_model:
        slowest = max(per_model, key=per_model.get)
        out.append(
            f"Slowest model: '{slowest}'. To speed up, deselect it, lower the folds, or set a "
            "row cap in step 3 (Evaluation settings)."
        )
    return out
