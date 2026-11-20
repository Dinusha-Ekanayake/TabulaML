"""Automatic configuration that respects the holdout.

    1. prepare rows and create the holdout split FIRST (random / grouped / time-ordered);
    2. profile all rows for the data-quality report (information only, no target use);
    3. make every TARGET-DEPENDENT recommendation (leakage, imbalance, target transform)
       from the dev rows only, so test labels never influence the configuration;
    4. check that the split can support the chosen cross-validation.

Shared by the UI, the benchmark and the tests so all of them derive the same split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tabulaml.core.types import (
    ColumnConfig,
    DatasetSchema,
    FeatureType,
    ForecastConfig,
    ImbalanceStrategy,
    PreprocessingConfig,
    ProblemType,
    ValidationIssue,
    ValidationScheme,
)
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.validation import check_supported, detect_leakage
from tabulaml.forecasting.checks import forecast_issues
from tabulaml.forecasting.features import add_forecast_features
from tabulaml.preprocessing.cleaning import DataCleaner, clean_target, number_format_warning
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.splits import Split, describe, make_holdout
from tabulaml.training.trainer import prepare

_GROUP_NAME = re.compile(r"(?i)(^|_)(id|key|sale|session|patient|customer|user|store|group)(_|$)")


@dataclass(frozen=True)
class Roles:
    target: str
    problem_type: ProblemType
    validation: ValidationScheme = ValidationScheme.RANDOM
    group_column: str | None = None
    order_column: str | None = None
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    decimal: str = "."
    thousands: str | None = None
    positive_class: str | None = None
    max_rows: int | None = None  # optional row cap (applied before the holdout split)
    extra_targets: tuple[str, ...] = ()  # several targets (multi-label / multi-output)
    forecast: ForecastConfig | None = None  # time-series forecasting (regression)
    drop_duplicates: bool = True  # drop exact duplicate rows before the split
    missing_rows: str = "impute"  # impute | drop rows with a missing feature value


@dataclass(frozen=True)
class AutoResult:
    schema_all: DatasetSchema  # all rows: data-quality report only
    schema_dev: DatasetSchema  # dev rows: drives the recommendations
    leakage: tuple[ValidationIssue, ...]
    supported: tuple[ValidationIssue, ...]
    split_issues: tuple[ValidationIssue, ...]
    format_issues: tuple[ValidationIssue, ...]
    config: PreprocessingConfig
    n_dev: int
    n_test: int


def suggest_validation(df: pd.DataFrame, target: str) -> tuple[ValidationScheme, str | None, str]:
    """Suggest grouped validation when an ID-like column repeats across rows."""
    n = len(df)
    for col in df.columns:
        if col == target or not _GROUP_NAME.search(str(col)):
            continue
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= n / 2:
            reason = (
                f"'{col}' repeats across rows (about {n / n_unique:.0f} rows per value). "
                "Rows of the same group are related, so grouped validation keeps each "
                "group entirely in training or entirely in testing."
            )
            return ValidationScheme.GROUPED, str(col), reason
    return ValidationScheme.RANDOM, None, "Rows look independent, so a random split is used."


def _skeleton(roles: Roles) -> PreprocessingConfig:
    """Minimal config carrying only what the holdout split needs."""
    return PreprocessingConfig(
        problem_type=roles.problem_type,
        target=roles.target,
        mode="auto",
        columns=(),
        test_size=roles.test_size,
        cv_folds=roles.cv_folds,
        random_state=roles.random_state,
        validation=roles.validation,
        group_column=roles.group_column,
        order_column=roles.order_column,
        positive_class=roles.positive_class,
        decimal=roles.decimal,
        thousands=roles.thousands,
        max_rows=roles.max_rows,
        extra_targets=roles.extra_targets,
        forecast=roles.forecast,
        drop_duplicates=roles.drop_duplicates,
        missing_rows=roles.missing_rows,
    )


def _split_issues(
    config: PreprocessingConfig, y: np.ndarray, split: Split
) -> list[ValidationIssue]:
    """Can the dev rows support the chosen cross-validation? (re-checked on every change)"""
    k, issues = config.cv_folds, []
    y_dev = y[split.dev]
    if config.problem_type == ProblemType.CLASSIFICATION and config.is_multi_output:
        issues.extend(_multi_class_issues(config, y_dev))
    elif config.problem_type == ProblemType.CLASSIFICATION:
        counts = pd.Series(y_dev).value_counts()
        if counts.min() < k:
            issues.append(
                ValidationIssue(
                    "error",
                    "class_too_small_for_cv",
                    f"A class has only {counts.min()} training rows; "
                    f"{k}-fold CV needs at least {k}. "
                    "Lower the number of folds or merge rare classes.",
                )
            )
        missing = set(np.unique(y)) - set(np.unique(y[split.test]))
        if missing:
            issues.append(
                ValidationIssue(
                    "warning",
                    "class_missing_in_test",
                    f"{len(missing)} class(es) do not appear in the test set; their test metrics "
                    "will be missing.",
                )
            )
    if config.validation == ValidationScheme.GROUPED and split.groups is not None:
        n_groups = len(np.unique(split.groups[split.dev]))
        if n_groups < k:
            issues.append(
                ValidationIssue(
                    "error",
                    "too_few_groups",
                    f"Only {n_groups} groups in the training rows; {k}-fold group CV needs {k}.",
                )
            )
    issues.extend(_size_issues(len(split.dev), len(split.test)))
    if len(split.dev) <= k or len(split.test) == 0:
        issues.append(
            ValidationIssue(
                "error",
                "too_few_rows_for_cv",
                "Not enough rows for this test share and number of folds.",
            )
        )
    return issues


SMALL_DEV_ROWS, TINY_TEST_ROWS = 100, 20


def _size_issues(n_dev: int, n_test: int) -> list[ValidationIssue]:
    """Small data gives unstable scores: say so plainly instead of hiding it."""
    out = []
    if n_dev < SMALL_DEV_ROWS:
        out.append(
            ValidationIssue(
                "warning",
                "small_dataset",
                f"Only {n_dev} training rows: scores will change a lot between folds and "
                "seeds. Prefer simple models, use more folds, and treat small differences "
                "between models as ties.",
            )
        )
    if 0 < n_test < TINY_TEST_ROWS:
        out.append(
            ValidationIssue(
                "warning",
                "tiny_test_set",
                f"The test set has only {n_test} rows, so its score is a rough estimate. "
                "Raise the test share or add data.",
            )
        )
    return out


def _multi_class_issues(config: PreprocessingConfig, y_dev: np.ndarray) -> list[ValidationIssue]:
    out = []
    for j, name in enumerate(config.targets):
        counts = pd.Series(y_dev[:, j]).value_counts()
        if len(counts) < 2:
            out.append(
                ValidationIssue(
                    "error",
                    "target_single_class",
                    f"Target '{name}' has only one class in the training rows.",
                    name,
                )
            )
        elif counts.min() < config.cv_folds:
            out.append(
                ValidationIssue(
                    "warning",
                    "rare_class_multi",
                    f"Target '{name}': a class has only {counts.min()} training rows, so some "
                    "folds will not contain it and its scores will vary.",
                    name,
                )
            )
    return out


def _format_issues(df: pd.DataFrame, roles: Roles) -> list[ValidationIssue]:
    out = []
    for col in df.columns:
        if df[col].dtype == object:
            message = number_format_warning(df[col], roles.decimal)
            if message:
                out.append(
                    ValidationIssue(
                        "warning", "ambiguous_number_format", f"'{col}': {message}", str(col)
                    )
                )
    return out


def _multi_target_settings(
    config: PreprocessingConfig, roles: Roles, df: pd.DataFrame
) -> PreprocessingConfig:
    """Other targets are never features; imbalance handling needs a single target."""
    for name in roles.extra_targets:
        current = next((c for c in config.columns if c.name == name), None)
        ftype = current.feature_type if current else FeatureType.CATEGORICAL
        reason = "Excluded: this is one of the target columns."
        config = config.with_column(ColumnConfig(name, ftype, include=False, reason=reason))
    if not roles.extra_targets:
        return config
    transform = config.target_transform
    if transform == "log1p":
        lows = [clean_target(df[t], regression=True).min() for t in config.targets]
        transform = "log1p" if min(lows) > -1 else "none"
    note = (
        f"Several targets ({', '.join(config.targets)}): one model copy per target. Class "
        "weights and resampling need a single target, so they are off."
    )
    return replace(
        config,
        imbalance=ImbalanceStrategy.NONE,
        target_transform=transform,
        explanations=(*config.explanations, note),
    )


def _forecast_checks(df: pd.DataFrame, roles: Roles) -> list[ValidationIssue]:
    if roles.forecast is None:
        return []
    return forecast_issues(df, roles.target, roles.order_column, roles.forecast, roles.cv_folds)


def _forecast_columns(
    config: PreprocessingConfig, roles: Roles, raw: pd.DataFrame
) -> PreprocessingConfig:
    """Same-period values of other columns are not known when forecasting: exclude them
    (their lagged copies are used instead when 'other columns' is set to lagged)."""
    fc = roles.forecast
    if fc is None:
        return config
    keep = {fc.series_column} if fc.series_column else set()
    skip = {*config.targets, roles.order_column}
    used = "its lagged copy is used instead" if fc.exog == "lagged" else "other columns are off"
    for col in config.columns:
        if col.name in raw.columns and col.name not in skip | keep and col.include:
            reason = f"Excluded: same-period value, not known at forecast time ({used})."
            config = config.with_column(replace(col, include=False, reason=reason))
    note = (
        f"Forecasting {fc.horizon} period(s) ahead with target lags {list(fc.lags())}, "
        f"rolling means {list(fc.windows)} and calendar parts. Every feature uses values at "
        f"least {fc.horizon} period(s) old, so it is known when the forecast is made."
    )
    return replace(config, explanations=(*config.explanations, note))


def _exclude_roles(config: PreprocessingConfig, roles: Roles) -> PreprocessingConfig:
    for name, what in ((roles.group_column, "group"), (roles.order_column, "time-order")):
        if not name:
            continue
        current = next((c for c in config.columns if c.name == name), None)
        ftype = current.feature_type if current else FeatureType.CATEGORICAL
        reason = f"Excluded: used as the {what} column for validation, not as a feature."
        config = config.with_column(ColumnConfig(name, ftype, include=False, reason=reason))
    return config


def auto_configure(df: pd.DataFrame, roles: Roles) -> AutoResult:
    """Holdout first, then recommendations from the dev rows only (see module docstring)."""
    skeleton = _skeleton(roles)
    raw = df
    df = add_forecast_features(df, skeleton)  # forecasting: past-only features, else as-is
    prep = prepare(df, skeleton)
    split = make_holdout(prep.X, prep.y, skeleton, prep.extra)
    dev_rows = df.loc[prep.index[split.dev]]
    cleaner = DataCleaner(decimal=roles.decimal, thousands=roles.thousands)
    clean_all = cleaner.fit_transform(df)
    clean_dev = cleaner.transform(dev_rows)
    schema_dev = profile_dataset(clean_dev)
    others = [t for t in roles.extra_targets if t in clean_dev.columns]
    leakage = (
        ()  # forecast features are built from the past only, so they are known in advance
        if roles.forecast
        else tuple(detect_leakage(clean_dev.drop(columns=others), roles.target, roles.problem_type))
    )
    config = recommend_config(
        schema_dev, roles.target, roles.problem_type, leakage, y=dev_rows[roles.target]
    )
    config = replace(
        config,
        validation=roles.validation,
        group_column=roles.group_column,
        order_column=roles.order_column,
        test_size=roles.test_size,
        cv_folds=roles.cv_folds,
        random_state=roles.random_state,
        decimal=roles.decimal,
        thousands=roles.thousands,
        positive_class=roles.positive_class,
        max_rows=roles.max_rows,
        extra_targets=roles.extra_targets,
        forecast=roles.forecast,
        drop_duplicates=roles.drop_duplicates,
        missing_rows=roles.missing_rows,
    )
    config = _exclude_roles(config, roles)
    config = _multi_target_settings(config, roles, df)
    config = _forecast_columns(config, roles, raw)
    note = (
        "Recommendations use the training (dev) rows only; the test rows are held out "
        f"before any target-based decision. {describe(config)}"
    )
    config = replace(config, explanations=(*config.explanations, note))
    schema_all = profile_dataset(clean_all)
    return AutoResult(
        schema_all=schema_all,
        schema_dev=schema_dev,
        leakage=leakage,
        supported=tuple(check_supported(clean_all, schema_all, roles.target)),
        split_issues=tuple(_split_issues(config, prep.y, split) + _forecast_checks(raw, roles)),
        format_issues=tuple(_format_issues(raw, roles)),
        config=config,
        n_dev=len(split.dev),
        n_test=len(split.test),
    )
