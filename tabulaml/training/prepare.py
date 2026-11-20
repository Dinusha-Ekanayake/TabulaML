"""Turn a raw table into model inputs: features, encoded target(s) and role columns.

Missing targets and exact duplicate rows are dropped, an optional row cap is applied, and
in forecasting mode the past-only forecast features are built first. Nothing is fitted
here, so it is safe to run before the train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tabulaml.core.types import PreprocessingConfig, ProblemType, ValidationScheme
from tabulaml.forecasting.features import add_forecast_features
from tabulaml.preprocessing.cleaning import clean_target, normalize_missing
from tabulaml.training.splits import order_key


@dataclass(frozen=True)
class Prepared:
    X: pd.DataFrame  # model features only
    y: np.ndarray  # label-encoded for classification
    class_labels: tuple[str, ...] | None
    notes: tuple[str, ...]
    extra: pd.DataFrame  # group / order columns (not features), aligned with X
    index: np.ndarray  # original df index of every prepared row
    target_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)  # several targets


def _role_columns(config: PreprocessingConfig) -> list[str]:
    return [c for c in (config.group_column, config.order_column) if c]


def _cap_rows(
    df: pd.DataFrame, rows: pd.Index, config: PreprocessingConfig
) -> tuple[pd.Index, str | None]:
    """Keep at most ``config.max_rows`` rows: the LATEST rows for time-ordered data, whole
    random groups for grouped data, otherwise a seeded random sample."""
    limit = config.max_rows
    if not limit or len(rows) <= limit:
        return rows, None
    if config.validation == ValidationScheme.TIME_ORDERED and config.order_column in df:
        key = order_key(df.loc[rows, config.order_column]).reset_index(drop=True)
        keep = np.sort(key.sort_values(kind="stable").index.to_numpy()[-limit:])
        chosen, how = rows[keep], "the latest"
    elif config.validation == ValidationScheme.GROUPED and config.group_column in df:
        groups = df.loc[rows, config.group_column].astype(str)
        sizes = groups.value_counts(sort=False)
        sizes = sizes.sample(frac=1.0, random_state=config.random_state)
        within = sizes.cumsum() <= limit
        keep_groups = set(sizes.index[within]) or {sizes.index[0]}
        chosen, how = rows[groups.isin(keep_groups).to_numpy()], "whole random groups,"
    else:
        rng = np.random.default_rng(config.random_state)
        chosen = rows[np.sort(rng.choice(len(rows), limit, replace=False))]
        how = "a random sample of"
    note = f"Row cap: kept {how} {len(chosen):,} of {len(rows):,} rows to speed up training."
    return chosen, note


def _clean_targets(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    regression = config.problem_type == ProblemType.REGRESSION
    missing = [t for t in config.targets if t not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) not found in the data: {missing}")
    return pd.DataFrame({t: clean_target(df[t], regression=regression) for t in config.targets})


def _encode_targets(
    y_raw: pd.DataFrame, config: PreprocessingConfig
) -> tuple[np.ndarray, tuple[str, ...] | None, dict[str, tuple[str, ...]]]:
    """(y, class labels of a single target, labels per target when there are several)."""
    if config.problem_type == ProblemType.REGRESSION:
        y = y_raw.to_numpy(dtype=float)
        return (y if config.is_multi_output else y[:, 0]), None, {}
    columns, per_target = [], {}
    for name in config.targets:
        encoder = LabelEncoder().fit(y_raw[name].astype(str))
        per_target[name] = tuple(str(c) for c in encoder.classes_)
        columns.append(encoder.transform(y_raw[name].astype(str)))
    if config.is_multi_output:
        return np.column_stack(columns), None, per_target
    return columns[0], per_target[config.target], {}


def incomplete_rows(df: pd.DataFrame, config: PreprocessingConfig) -> pd.Series:
    """True for rows with a missing value in any column except the target(s) and roles.

    Judged on whole rows (like duplicates), so the row set and therefore the test split
    do not change when columns are included or excluded."""
    skip = {*config.targets, *_role_columns(config)}
    columns = [c for c in df.columns if c not in skip]
    if not columns:
        return pd.Series(False, index=df.index)
    return pd.DataFrame({c: normalize_missing(df[c]) for c in columns}).isna().any(axis=1)


def _complete_rows(df: pd.DataFrame, config: PreprocessingConfig, notes: list[str]) -> pd.Series:
    """missing_rows="drop": listwise deletion, the user's choice. Nothing is learned, so it
    is safe before the split; missing values in new data are still imputed."""
    if config.missing_rows != "drop" or config.forecast is not None:  # lags need every period
        return pd.Series(True, index=df.index)
    present = ~incomplete_rows(df, config)
    dropped = int((~present).sum())
    if dropped:
        notes.append(f"Dropped {dropped} rows with a missing feature value (your choice).")
    if not present.any():
        raise ValueError(
            "Every row has a missing value in some feature, so dropping them leaves nothing. "
            "Impute instead, or exclude the mostly-empty columns first."
        )
    return present


def prepare(df: pd.DataFrame, config: PreprocessingConfig) -> Prepared:
    """Select features, clean the target(s), drop missing-target rows and exact duplicates.

    With several targets, ``y`` is 2-D (one column per target) and the other targets are
    never used as features. In forecasting mode the lag / rolling / calendar features are
    built first (past values only, see ``tabulaml.forecasting.features``)."""
    df = add_forecast_features(df, config)
    roles = _role_columns(config)
    excluded = {*config.targets, *roles}
    features = [c.name for c in config.included() if c.name not in excluded]
    missing = [c for c in features + roles if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the data: {missing}")
    targets = _clean_targets(df, config)
    notes: list[str] = []
    keep = targets.notna().all(axis=1)
    if (~keep).sum():
        what = "missing a target value" if config.is_multi_output else "with a missing target"
        notes.append(f"Dropped {int((~keep).sum())} rows {what}.")
    # Duplicates are judged on WHOLE rows, so the row set (and therefore the holdout split)
    # does not change when the user includes or excludes feature columns.
    keep &= _complete_rows(df, config, notes)
    duplicated = df.loc[keep].astype(str).duplicated()
    if not config.drop_duplicates:
        duplicated[:] = False
        notes.append("Exact duplicate rows were kept (your choice).")
    elif duplicated.sum():
        notes.append(f"Dropped {int(duplicated.sum())} exact duplicate rows before splitting.")
    rows = df.loc[keep].index[~duplicated.to_numpy()]
    rows, cap_note = _cap_rows(df, rows, config)
    if cap_note:
        notes.append(cap_note)
    X = df.loc[rows, features].reset_index(drop=True)
    extra = df.loc[rows, roles].reset_index(drop=True)
    y, labels, per_target = _encode_targets(targets.loc[rows].reset_index(drop=True), config)
    return Prepared(X, y, labels, tuple(notes), extra, np.asarray(rows), per_target)


def prepare_data(df: pd.DataFrame, config: PreprocessingConfig):  # noqa: ANN201 - compat
    """Backwards-compatible (X, y, class_labels, notes)."""
    p = prepare(df, config)
    return p.X, p.y, p.class_labels, list(p.notes)
