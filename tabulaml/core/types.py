"""Core immutable data contracts shared by every layer (see docs/02_ARCHITECTURE.md §4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

LOWER_IS_BETTER = frozenset({"rmse", "mae", "medae", "mape", "log_loss", "hamming_loss"})


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"


class ModelProfile(str, Enum):
    TREE = "tree"
    SCALE_SENSITIVE = "scale_sensitive"


class ImbalanceStrategy(str, Enum):
    NONE = "none"
    CLASS_WEIGHT = "class_weight"
    RANDOM_OVERSAMPLE = "random_oversample"
    SMOTE = "smote"  # becomes SMOTENC when categorical features are present
    UNDERSAMPLE = "undersample"


class ValidationScheme(str, Enum):
    RANDOM = "random"  # independent rows: (stratified) random split and K-fold
    GROUPED = "grouped"  # repeated entities (customers, sales...): group-aware split and CV
    TIME_ORDERED = "time_ordered"  # chronological: last share is the test set, forward CV


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    raw_dtype: str
    inferred_type: FeatureType
    missing_ratio: float
    n_unique: int
    unique_ratio: float
    sample_values: tuple[str, ...]
    skewness: float | None = None
    outlier_ratio: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSchema:
    columns: tuple[ColumnProfile, ...]
    n_rows: int
    n_duplicate_rows: int

    def get(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def names_of(self, feature_type: FeatureType) -> list[str]:
        return [c.name for c in self.columns if c.inferred_type == feature_type]


@dataclass(frozen=True)
class ColumnConfig:
    name: str
    feature_type: FeatureType
    include: bool = True
    impute: str = "auto"  # auto|median|mean|most_frequent|constant|knn
    encoding: str = "auto"  # auto|onehot|ordinal|target
    scaling: str = "auto"  # auto|standard|robust|minmax|none
    transform: str = "auto"  # auto|yeo_johnson|none
    outliers: str = "auto"  # auto|iqr_clip|none
    reason: str = ""


@dataclass(frozen=True)
class ForecastConfig:
    """Forecasting = regression on features built from the PAST of the series.

    Every lag is at least ``horizon`` periods back, so all inputs are known when the
    forecast is made (direct multi-step strategy)."""

    horizon: int = 1  # forecast this many periods ahead
    n_lags: int = 3  # target lags horizon ... horizon + n_lags - 1
    seasonal_lag: int | None = None  # e.g. 52 for weekly data (must be >= horizon)
    windows: tuple[int, ...] = (4,)  # rolling means of the target, shifted by horizon
    series_column: str | None = None  # several series in one file (e.g. one per store)
    exog: str = "lagged"  # other columns: "lagged" by horizon (safe) | "none"

    def lags(self) -> tuple[int, ...]:
        regular = range(self.horizon, self.horizon + max(self.n_lags, 1))
        extra = (self.seasonal_lag,) if self.seasonal_lag else ()
        return tuple(sorted({*regular, *(k for k in extra if k >= self.horizon)}))


@dataclass(frozen=True)
class PreprocessingConfig:
    problem_type: ProblemType
    target: str
    mode: str  # "auto" | "custom"
    columns: tuple[ColumnConfig, ...]
    missing_indicator: bool = True
    rare_category_min_frequency: float = 0.01
    high_cardinality_threshold: int = 20
    normalize_category_case: bool = True
    imbalance: ImbalanceStrategy = ImbalanceStrategy.NONE
    target_transform: str = "none"  # none|log1p|yeo_johnson
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    explanations: tuple[str, ...] = ()
    validation: ValidationScheme = ValidationScheme.RANDOM
    group_column: str | None = None
    order_column: str | None = None
    positive_class: str | None = None  # binary classification: the class of interest
    decimal: str = "."  # number format for text numbers, e.g. "," for "1,25"
    thousands: str | None = None
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)  # manual params
    tuning: dict[str, int] = field(default_factory=dict)  # model key -> search candidates
    max_rows: int | None = None  # keep at most this many rows (sampled before the split)
    extra_targets: tuple[str, ...] = ()  # several targets: multi-label / multi-output
    forecast: ForecastConfig | None = None  # time-series forecasting (regression)

    @property
    def targets(self) -> tuple[str, ...]:
        """Every target column, the main ``target`` first."""
        return (self.target, *self.extra_targets)

    @property
    def is_multi_output(self) -> bool:
        return bool(self.extra_targets)

    def included(self) -> list[ColumnConfig]:
        return [c for c in self.columns if c.include]

    def with_column(self, cfg: ColumnConfig) -> PreprocessingConfig:
        """Return a NEW config with the column of the same name replaced (or appended)."""
        if any(c.name == cfg.name for c in self.columns):
            cols = tuple(cfg if c.name == cfg.name else c for c in self.columns)
        else:
            cols = (*self.columns, cfg)
        return replace(self, columns=cols)


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # error|warning|info
    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class ModelResult:
    key: str
    name: str
    status: str  # ok|incomplete (some CV folds failed)|failed
    cv_scores: dict[str, tuple[float, ...]]
    test_metrics: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    n_failed_folds: int = 0
    effective_imbalance: str = ""
    pipeline: Any = field(default=None, repr=False, compare=False)
    y_pred: Any = field(default=None, repr=False, compare=False)
    y_proba: Any = field(default=None, repr=False, compare=False)
    oof_pred: Any = field(default=None, repr=False, compare=False)  # dev out-of-fold
    oof_proba: Any = field(default=None, repr=False, compare=False)
    oof_mask: Any = field(default=None, repr=False, compare=False)  # dev rows with an OOF value
    tuning_history: Any = field(default=None, repr=False, compare=False)
    per_target: dict[str, dict[str, float]] = field(default_factory=dict)  # several targets


@dataclass(frozen=True)
class TrainingRun:
    problem_type: ProblemType
    config: PreprocessingConfig
    primary_metric: str
    results: tuple[ModelResult, ...]
    class_labels: tuple[str, ...] | None
    X_test: Any = field(repr=False, compare=False)
    y_test: Any = field(repr=False, compare=False)
    feature_columns: tuple[str, ...] = ()
    versions: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    y_dev: Any = field(default=None, repr=False, compare=False)
    positive_index: int | None = None  # encoded positive class (binary classification)
    cancelled: bool = False  # stopped by the user; only finished models are included
    target_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)  # multi classif.
    test_order: Any = field(default=None, repr=False, compare=False)  # forecasting: test times

    @property
    def is_multi_output(self) -> bool:
        return self.config.is_multi_output

    def leaderboard(self) -> pd.DataFrame:
        """One row per model, best first by CV mean of the primary metric; failed rows last."""
        rows = [self._row(r) for r in self.results]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        key = f"cv_{self.primary_metric}_mean"
        if key not in df.columns:  # e.g. every model failed, so no CV scores exist
            df[key] = np.nan
        ascending = self.primary_metric in LOWER_IS_BETTER
        df["_failed"] = df["status"] != "ok"
        df = df.sort_values(["_failed", key], ascending=[True, ascending], na_position="last")
        return df.drop(columns="_failed").reset_index(drop=True)

    def _row(self, r: ModelResult) -> dict[str, Any]:
        row: dict[str, Any] = {"model": r.name, "key": r.key, "status": r.status}
        for metric, scores in r.cv_scores.items():
            arr = np.asarray(scores, dtype=float)
            row[f"cv_{metric}_mean"] = float(arr.mean()) if arr.size else np.nan
            row[f"cv_{metric}_std"] = float(arr.std()) if arr.size else np.nan
        for metric, value in r.test_metrics.items():
            row[f"test_{metric}"] = value
        row["fit_s"] = r.fit_seconds
        row["predict_s"] = r.predict_seconds
        if r.effective_imbalance:
            row["imbalance"] = r.effective_imbalance
        if r.params:
            row["params"] = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items()))
        if r.n_failed_folds:
            row["failed_folds"] = r.n_failed_folds
        if r.error:
            row["error"] = r.error
        return row

    def result_for(self, key: str) -> ModelResult:
        for r in self.results:
            if r.key == key:
                return r
        raise KeyError(key)

    def best(self) -> ModelResult:
        board = self.leaderboard()
        ok = board[board["status"] == "ok"]
        if ok.empty:
            raise ValueError("No model trained successfully")
        best_key = ok.iloc[0]["key"]
        return next(r for r in self.results if r.key == best_key)
