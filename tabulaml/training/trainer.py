"""Leakage-safe training loop: split -> cross-validate -> fit on train -> evaluate on test.

Rows with a missing target and exact duplicate rows are dropped (stateless, safe before the
split). Every learned step lives inside each model's pipeline, so cross-validation refits
all preprocessing inside every fold and the held-out test set is touched exactly once.
"""

from __future__ import annotations

import logging
import platform
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from tabulaml.core.types import ModelResult, PreprocessingConfig, ProblemType, TrainingRun
from tabulaml.evaluation.metrics import (
    compute_metrics,
    default_primary_metric,
    display_value,
    scoring_map,
)
from tabulaml.models.registry import SVM_MAX_ROWS, get_model
from tabulaml.preprocessing.cleaning import clean_target
from tabulaml.training.pipeline_factory import build_model_pipeline

logger = logging.getLogger(__name__)
Progress = Callable[[str, int, int], None]
_PACKAGES = (
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced-learn",
    "pandas",
    "numpy",
)
MAX_ERROR_CHARS = 300


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def prepare_data(
    df: pd.DataFrame, config: PreprocessingConfig
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...] | None, list[str]]:
    """Return (X, y, class_labels, notes). y is label-encoded for classification."""
    regression = config.problem_type == ProblemType.REGRESSION
    features = [c.name for c in config.included() if c.name != config.target]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}")
    y_all = clean_target(df[config.target], regression=regression)
    notes: list[str] = []
    keep = y_all.notna()
    if (~keep).sum():
        notes.append(f"Dropped {int((~keep).sum())} rows with a missing target.")
    data = df.loc[keep, features].assign(__y=y_all[keep])
    duplicated = data.astype(str).duplicated()
    if duplicated.sum():
        notes.append(f"Dropped {int(duplicated.sum())} exact duplicate rows before splitting.")
    data = data[~duplicated]
    X, y_raw = data[features], data["__y"]
    if regression:
        return X, y_raw.to_numpy(dtype=float), None, notes
    encoder = LabelEncoder().fit(y_raw.astype(str))
    labels = tuple(str(c) for c in encoder.classes_)
    return X, encoder.transform(y_raw.astype(str)), labels, notes


def _cv(config: PreprocessingConfig) -> Any:
    if config.problem_type == ProblemType.CLASSIFICATION:
        return StratifiedKFold(config.cv_folds, shuffle=True, random_state=config.random_state)
    return KFold(config.cv_folds, shuffle=True, random_state=config.random_state)


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


def _failed(key: str, message: str) -> ModelResult:
    return ModelResult(key, get_model(key).name, "failed", {}, {}, 0.0, 0.0, error=message)


def _cross_validate(est: Any, X: pd.DataFrame, y: np.ndarray, config: PreprocessingConfig) -> dict:
    scorers = scoring_map(config.problem_type, y)
    res = cross_validate(est, X, y, cv=_cv(config), scoring=scorers, n_jobs=1, error_score=np.nan)
    return {m: tuple(float(display_value(m, v)) for v in res[f"test_{m}"]) for m in scorers}


def _train_one(
    key: str, split: tuple, config: PreprocessingConfig, bases: Sequence[str] | None
) -> ModelResult:
    X_tr, X_te, y_tr, y_te = split
    if key == "svm" and len(X_tr) > SVM_MAX_ROWS:
        return _failed(key, f"Skipped: SVM is too slow above {SVM_MAX_ROWS:,} training rows.")
    try:
        est = build_model_pipeline(key, config, y=y_tr, base_keys=bases)
        cv_scores = _cross_validate(est, X_tr, y_tr, config)
        model = clone(est)
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        predict_s = time.perf_counter() - start
        metrics = compute_metrics(config.problem_type, y_te, y_pred, y_proba, y_train=y_tr)
    except Exception as exc:  # noqa: BLE001 - one failing model must never abort the run
        logger.exception("Model %s failed", key)
        return _failed(key, _short_error(exc))
    return ModelResult(
        key,
        get_model(key).name,
        "ok",
        cv_scores,
        metrics,
        fit_s,
        predict_s,
        pipeline=model,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def train_models(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    model_keys: Sequence[str],
    primary_metric: str | None = None,
    progress: Progress | None = None,
    ensemble_bases: Mapping[str, Sequence[str]] | None = None,
) -> TrainingRun:
    """Train and evaluate every requested model on the same split."""
    if not model_keys:
        raise ValueError("Select at least one model.")
    X, y, labels, notes = prepare_data(df, config)
    stratify = y if config.problem_type == ProblemType.CLASSIFICATION else None
    split = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )
    notes.append(f"Train rows: {len(split[0]):,}; test rows: {len(split[1]):,}.")
    primary = primary_metric or default_primary_metric(config.problem_type, split[2])
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
        for i, key in enumerate(model_keys):
            if progress:
                progress(key, i, len(model_keys))
            bases = (ensemble_bases or {}).get(key)
            results.append(_train_one(key, split, config, bases))
    if progress:
        progress("done", len(model_keys), len(model_keys))
    return TrainingRun(
        config.problem_type,
        config,
        primary,
        tuple(results),
        labels,
        X_test=split[1],
        y_test=split[3],
        feature_columns=tuple(X.columns),
        versions=_versions(),
        notes=tuple(notes),
    )
