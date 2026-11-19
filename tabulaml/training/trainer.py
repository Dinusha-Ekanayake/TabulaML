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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder

from tabulaml.core.types import (
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
)
from tabulaml.evaluation.metrics import compute_metrics, default_primary_metric, minority_label
from tabulaml.models.registry import SVM_MAX_ROWS, get_model
from tabulaml.preprocessing.cleaning import clean_target
from tabulaml.training.pipeline_factory import (
    build_model_pipeline,
    effective_imbalance,
    model_params,
)
from tabulaml.training.splits import Split, describe, make_cv, make_holdout
from tabulaml.training.tuning import tune

logger = logging.getLogger(__name__)
Progress = Callable[[str, int, int], None]
_PACKAGES = ("scikit-learn", "xgboost", "lightgbm", "catboost", "imbalanced-learn", "pandas")
MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class Prepared:
    X: pd.DataFrame  # model features only
    y: np.ndarray  # label-encoded for classification
    class_labels: tuple[str, ...] | None
    notes: tuple[str, ...]
    extra: pd.DataFrame  # group / order columns (not features), aligned with X
    index: np.ndarray  # original df index of every prepared row


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _role_columns(config: PreprocessingConfig) -> list[str]:
    return [c for c in (config.group_column, config.order_column) if c]


def prepare(df: pd.DataFrame, config: PreprocessingConfig) -> Prepared:
    """Select features, clean the target, drop missing-target rows and exact duplicates."""
    regression = config.problem_type == ProblemType.REGRESSION
    roles = _role_columns(config)
    features = [
        c.name for c in config.included() if c.name != config.target and c.name not in roles
    ]
    missing = [c for c in features + roles if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}")
    y_all = clean_target(df[config.target], regression=regression)
    notes: list[str] = []
    keep = y_all.notna()
    if (~keep).sum():
        notes.append(f"Dropped {int((~keep).sum())} rows with a missing target.")
    # Duplicates are judged on WHOLE rows, so the row set (and therefore the holdout split)
    # does not change when the user includes or excludes feature columns.
    duplicated = df.loc[keep].astype(str).duplicated()
    if duplicated.sum():
        notes.append(f"Dropped {int(duplicated.sum())} exact duplicate rows before splitting.")
    rows = df.loc[keep].index[~duplicated.to_numpy()]
    data = df.loc[rows, features + roles].assign(__y=y_all[rows]).reset_index(drop=True)
    X, y_raw, extra = data[features], data["__y"], data[roles]
    index = np.asarray(rows)
    if regression:
        return Prepared(X, y_raw.to_numpy(dtype=float), None, tuple(notes), extra, index)
    encoder = LabelEncoder().fit(y_raw.astype(str))
    labels = tuple(str(c) for c in encoder.classes_)
    return Prepared(X, encoder.transform(y_raw.astype(str)), labels, tuple(notes), extra, index)


def prepare_data(df: pd.DataFrame, config: PreprocessingConfig):  # noqa: ANN201 - compat
    """Backwards-compatible (X, y, class_labels, notes)."""
    p = prepare(df, config)
    return p.X, p.y, p.class_labels, list(p.notes)


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


def _cv_loop(est: Any, ctx: _Ctx) -> tuple[dict, np.ndarray, Any, np.ndarray, int]:
    """Fold scores (display-oriented) plus out-of-fold predictions on the dev rows."""
    n = len(ctx.X_dev)
    oof_pred = np.full(n, np.nan)
    oof_proba = np.full((n, ctx.n_classes), np.nan) if ctx.n_classes else None
    mask = np.zeros(n, dtype=bool)
    folds: list[dict[str, float]] = []
    failed = 0
    cv = make_cv(ctx.config, ctx.y_dev)
    for tr, te in cv.split(ctx.X_dev, ctx.y_dev, ctx.groups_dev):
        try:
            model = clone(est).fit(ctx.X_dev.iloc[tr], ctx.y_dev[tr])
            pred = np.asarray(model.predict(ctx.X_dev.iloc[te])).ravel()  # CatBoost: (n, 1)
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
    names = sorted({m for f in folds for m in f})
    scores = {m: tuple(float(f.get(m, np.nan)) for f in folds) for m in names}
    return scores, oof_pred, oof_proba, mask, failed


def _status(scores: dict, failed: int, primary: str) -> str:
    primary_scores = np.asarray(scores.get(primary, ()), dtype=float)
    if failed or not primary_scores.size or np.isnan(primary_scores).any():
        return "incomplete"
    return "ok"


def _train_one(key: str, ctx: _Ctx, bases: Sequence[str] | None) -> ModelResult:
    config = ctx.config
    if key == "svm" and len(ctx.X_dev) > SVM_MAX_ROWS:
        return _failed(key, f"Skipped: SVM is too slow above {SVM_MAX_ROWS:,} training rows.")
    try:
        est = build_model_pipeline(key, config, y=ctx.y_dev, base_keys=bases)
        history = None
        if config.tuning.get(key):
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
        scores, oof_pred, oof_proba, mask, n_failed = _cv_loop(est, ctx)
        start = time.perf_counter()
        model = clone(est).fit(ctx.X_dev, ctx.y_dev)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = np.asarray(model.predict(ctx.X_test)).ravel()
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
    except Exception as exc:  # noqa: BLE001 - one failing model must never abort the run
        logger.exception("Model %s failed", key)
        return _failed(key, _short_error(exc))
    status = _status(scores, n_failed, ctx.primary)
    error = f"{n_failed} CV fold(s) failed or gave no {ctx.primary}." if status != "ok" else None
    return ModelResult(
        key,
        get_model(key).name,
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
    )


def build_context(
    df: pd.DataFrame, config: PreprocessingConfig, primary_metric: str | None = None
) -> tuple[_Ctx, Prepared, Split]:
    prep = prepare(df, config)
    split = make_holdout(prep.X, prep.y, config, prep.extra)
    y_dev = prep.y[split.dev]
    n_classes = len(prep.class_labels) if prep.class_labels else 0
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
) -> TrainingRun:
    """Train and evaluate every requested model on the same split."""
    if not model_keys:
        raise ValueError("Select at least one model.")
    ctx, prep, split = build_context(df, config, primary_metric)
    notes = [
        *prep.notes,
        describe(config),
        f"Dev rows (training, CV, tuning): {len(split.dev):,}; "
        f"held-out test rows: {len(split.test):,}.",
    ]
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
        for i, key in enumerate(model_keys):
            if progress:
                progress(key, i, len(model_keys))
            results.append(_train_one(key, ctx, (ensemble_bases or {}).get(key)))
    if progress:
        progress("done", len(model_keys), len(model_keys))
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
    )
