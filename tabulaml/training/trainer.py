"""Leakage-safe training loop.

    prepare -> holdout split (random / grouped / time-ordered) -> CV on dev rows only
    (scores + out-of-fold predictions) -> optional tuning on dev -> fit on all dev rows
    -> score ONCE on the held-out test rows.

Every learned step lives inside each model's pipeline, so every fold refits all
preprocessing. A model whose primary metric failed in any fold is marked ``incomplete``
and is never chosen as the best model.
"""

from __future__ import annotations

import logging
import platform
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from tabulaml.core.types import (
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
)
from tabulaml.evaluation.metrics import compute_metrics, default_primary_metric, minority_label
from tabulaml.evaluation.multi_metrics import per_target_metrics
from tabulaml.models.registry import SVM_MAX_ROWS, get_model
from tabulaml.training.pipeline_factory import (
    build_model_pipeline,
    effective_imbalance,
    model_params,
)
from tabulaml.training.prepare import Prepared, prepare, prepare_data
from tabulaml.training.progress import Progress, StopCheck, TrainingCancelled, _Tracker
from tabulaml.training.splits import Split, describe, make_cv, make_holdout, order_key
from tabulaml.training.tuning import tune

__all__ = [
    "Prepared",
    "TrainingCancelled",
    "build_context",
    "positive_index",
    "prepare",
    "prepare_data",
    "train_models",
]

logger = logging.getLogger(__name__)
_PACKAGES = ("scikit-learn", "xgboost", "lightgbm", "catboost", "imbalanced-learn", "pandas")
MAX_ERROR_CHARS = 300


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def positive_index(
    config: PreprocessingConfig, labels: tuple[str, ...] | None, y: np.ndarray
) -> int | None:
    """Encoded positive class for binary tasks: the user's choice, else the minority class."""
    if config.problem_type != ProblemType.CLASSIFICATION or not labels or len(labels) != 2:
        return None
    if config.positive_class in labels:
        return labels.index(config.positive_class)
    return minority_label(y)


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]


def _failed(key: str, message: str) -> ModelResult:
    return ModelResult(key, get_model(key).name, "failed", {}, {}, 0.0, 0.0, error=message)


def _full_proba(model: Any, X: pd.DataFrame, n_classes: int) -> np.ndarray | None:
    """predict_proba aligned to ALL classes (a fold may miss a rare class in training)."""
    if not hasattr(model, "predict_proba"):
        return None
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1])), dtype=int)
    full = np.zeros((len(X), n_classes))
    full[:, classes] = proba
    return full


@dataclass(frozen=True)
class _Ctx:
    config: PreprocessingConfig
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    groups_dev: np.ndarray | None
    X_test: pd.DataFrame
    y_test: np.ndarray
    n_classes: int
    positive: int | None
    primary: str


def _cv_loop(
    est: Any, ctx: _Ctx, tracker: _Tracker | None = None, name: str = ""
) -> tuple[dict, np.ndarray, Any, np.ndarray, int]:
    """Fold scores (display-oriented) plus out-of-fold predictions on the dev rows."""
    n = len(ctx.X_dev)
    oof_pred = np.full(np.shape(ctx.y_dev), np.nan)
    oof_proba = np.full((n, ctx.n_classes), np.nan) if ctx.n_classes else None
    mask = np.zeros(n, dtype=bool)
    folds: list[dict[str, float]] = []
    failed = 0
    cv = make_cv(ctx.config, ctx.y_dev)
    for fold, (tr, te) in enumerate(cv.split(ctx.X_dev, ctx.y_dev, ctx.groups_dev), start=1):
        if tracker:
            tracker.check()
            tracker.report(f"{name}: cross-validation fold {fold}/{ctx.config.cv_folds}")
        try:
            model = clone(est).fit(ctx.X_dev.iloc[tr], ctx.y_dev[tr])
            pred = _predictions(model, ctx.X_dev.iloc[te], ctx.y_dev)
            proba = _full_proba(model, ctx.X_dev.iloc[te], ctx.n_classes) if ctx.n_classes else None
            folds.append(
                compute_metrics(
                    ctx.config.problem_type,
                    ctx.y_dev[te],
                    pred,
                    proba,
                    y_train=ctx.y_dev[tr],
                    positive=ctx.positive,
                )
            )
            oof_pred[te], mask[te] = pred, True
            if proba is not None:
                oof_proba[te] = proba
        except Exception as exc:  # noqa: BLE001 - a failing fold is recorded, not fatal
            logger.warning("CV fold failed: %s", _short_error(exc))
            failed += 1
            folds.append({})
        if tracker:
            tracker.step(f"{name}: fold {fold}/{ctx.config.cv_folds} done")
    names = sorted({m for f in folds for m in f})
    scores = {m: tuple(float(f.get(m, np.nan)) for f in folds) for m in names}
    return scores, oof_pred, oof_proba, mask, failed


def _predictions(model: Any, X: pd.DataFrame, y_like: np.ndarray) -> np.ndarray:
    """1-D predictions for one target (CatBoost returns (n, 1)); (n, k) for k targets."""
    pred = np.asarray(model.predict(X))
    return pred.reshape(len(X), -1) if np.ndim(y_like) == 2 else pred.ravel()


def _status(scores: dict, failed: int, primary: str) -> str:
    primary_scores = np.asarray(scores.get(primary, ()), dtype=float)
    if failed or not primary_scores.size or np.isnan(primary_scores).any():
        return "incomplete"
    return "ok"


def _train_one(
    key: str, ctx: _Ctx, bases: Sequence[str] | None, tracker: _Tracker | None = None
) -> ModelResult:
    config = ctx.config
    name = get_model(key).name
    if key == "svm" and len(ctx.X_dev) > SVM_MAX_ROWS:
        return _failed(key, f"Skipped: SVM is too slow above {SVM_MAX_ROWS:,} training rows.")
    try:
        est = build_model_pipeline(key, config, y=ctx.y_dev, base_keys=bases)
        history = None
        if config.tuning.get(key):
            if tracker:
                tracker.report(f"{name}: auto-tuning ({config.tuning[key]} combinations)")
            est, history = tune(
                est,
                key,
                ctx.X_dev,
                ctx.y_dev,
                ctx.groups_dev,
                config,
                ctx.primary,
                n_iter=config.tuning[key],
                positive=ctx.positive,
            )
        scores, oof_pred, oof_proba, mask, n_failed = _cv_loop(est, ctx, tracker, name)
        if tracker:
            tracker.check()
            tracker.report(f"{name}: final fit on all training rows + test")
        start = time.perf_counter()
        model = clone(est).fit(ctx.X_dev, ctx.y_dev)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = _predictions(model, ctx.X_test, ctx.y_dev)
        y_proba = _full_proba(model, ctx.X_test, ctx.n_classes) if ctx.n_classes else None
        predict_s = time.perf_counter() - start
        metrics = compute_metrics(
            config.problem_type,
            ctx.y_test,
            y_pred,
            y_proba,
            y_train=ctx.y_dev,
            positive=ctx.positive,
        )
        per_target = (
            per_target_metrics(config.problem_type, ctx.y_test, y_pred, config.targets)
            if config.is_multi_output
            else {}
        )
    except TrainingCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - one failing model must never abort the run
        logger.exception("Model %s failed", key)
        return _failed(key, _short_error(exc))
    status = _status(scores, n_failed, ctx.primary)
    error = f"{n_failed} CV fold(s) failed or gave no {ctx.primary}." if status != "ok" else None
    return ModelResult(
        key,
        name,
        status,
        scores,
        metrics,
        fit_s,
        predict_s,
        error=error,
        params=model_params(est, key),
        n_failed_folds=n_failed,
        effective_imbalance=effective_imbalance(key, config, as_base=False),
        pipeline=model,
        y_pred=y_pred,
        y_proba=y_proba,
        oof_pred=oof_pred,
        oof_proba=oof_proba,
        oof_mask=mask,
        tuning_history=history,
        per_target=per_target,
    )


def build_context(
    df: pd.DataFrame, config: PreprocessingConfig, primary_metric: str | None = None
) -> tuple[_Ctx, Prepared, Split]:
    prep = prepare(df, config)
    split = make_holdout(prep.X, prep.y, config, prep.extra)
    y_dev = prep.y[split.dev]
    n_classes = len(prep.class_labels) if prep.class_labels else 0  # 0: no probabilities
    positive = positive_index(config, prep.class_labels, y_dev)
    primary = primary_metric or default_primary_metric(config.problem_type, y_dev)
    groups_dev = split.groups[split.dev] if split.groups is not None else None
    ctx = _Ctx(
        config,
        prep.X.iloc[split.dev].reset_index(drop=True),
        y_dev,
        groups_dev,
        prep.X.iloc[split.test].reset_index(drop=True),
        prep.y[split.test],
        n_classes,
        positive,
        primary,
    )
    return ctx, prep, split


def train_models(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    model_keys: Sequence[str],
    primary_metric: str | None = None,
    progress: Progress | None = None,
    ensemble_bases: Mapping[str, Sequence[str]] | None = None,
    should_stop: StopCheck | None = None,
) -> TrainingRun:
    """Train and evaluate every requested model on the same split.

    ``progress(label, done, total)`` is called at fold level (total = models x (folds + 1)).
    When ``should_stop()`` becomes true, training stops at the next fold boundary and the
    finished models are returned (``run.cancelled``); with none finished, TrainingCancelled.
    """
    if not model_keys:
        raise ValueError("Select at least one model.")
    ctx, prep, split = build_context(df, config, primary_metric)
    notes = [
        *prep.notes,
        describe(config),
        f"Dev rows (training, CV, tuning): {len(split.dev):,}; "
        f"held-out test rows: {len(split.test):,}.",
    ]
    per_model = config.cv_folds + 1
    tracker = _Tracker(len(model_keys) * per_model, progress, should_stop)
    results, cancelled = _run_models(model_keys, ctx, ensemble_bases or {}, tracker, per_model)
    if cancelled:
        if not results:
            raise TrainingCancelled("Training was cancelled before any model finished.")
        notes.append(
            f"Cancelled by the user after {len(results)} of {len(model_keys)} models; "
            "only the finished models are shown."
        )
    tracker.report("done")
    return TrainingRun(
        config.problem_type,
        config,
        ctx.primary,
        tuple(results),
        prep.class_labels,
        X_test=ctx.X_test,
        y_test=ctx.y_test,
        feature_columns=tuple(prep.X.columns),
        versions=_versions(),
        notes=tuple(notes),
        y_dev=ctx.y_dev,
        positive_index=ctx.positive,
        cancelled=cancelled,
        target_labels=prep.target_labels,
        test_order=_test_order(config, prep, split),
    )


def _test_order(config: PreprocessingConfig, prep: Prepared, split: Split) -> np.ndarray | None:
    """Times of the held-out rows (time-ordered validation), for over-time plots."""
    if not config.order_column or config.order_column not in prep.extra.columns:
        return None
    return order_key(prep.extra[config.order_column]).to_numpy()[split.test]


def _run_models(
    keys: Sequence[str],
    ctx: _Ctx,
    bases: Mapping[str, Sequence[str]],
    tracker: _Tracker,
    per_model: int,
) -> tuple[list[ModelResult], bool]:
    results: list[ModelResult] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
        for i, key in enumerate(keys):
            try:
                tracker.check()
                tracker.report(f"{get_model(key).name}: starting ({i + 1}/{len(keys)})")
                results.append(_train_one(key, ctx, bases.get(key), tracker))
            except TrainingCancelled:
                return results, True
            tracker.done = (i + 1) * per_model  # skipped / failed models still advance
    return results, False
