"""Holdout and cross-validation splits for independent, grouped and time-ordered data.

| Scheme        | Holdout                                  | Cross-validation               |
|---------------|------------------------------------------|--------------------------------|
| random        | (stratified) random share                | (Stratified)KFold, shuffled     |
| grouped       | whole groups held out (GroupShuffleSplit)| (Stratified)GroupKFold          |
| time_ordered  | the LAST share in time order             | TimeSeriesSplit (forward)       |

Indices are positional (0..n-1) over the prepared frame, so the UI, the trainer and the
benchmark all derive exactly the same split from the same config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from tabulaml.config import MISSING_TOKENS
from tabulaml.core.types import PreprocessingConfig, ProblemType, ValidationScheme
from tabulaml.preprocessing.cleaning import coerce_numeric, parse_datetime


@dataclass(frozen=True)
class Split:
    dev: np.ndarray  # positional indices used for training / CV / tuning
    test: np.ndarray  # positional indices of the held-out test set
    groups: np.ndarray | None  # group labels aligned with the prepared frame (grouped only)


def order_key(values: pd.Series) -> pd.Series:
    """Sortable key for an order column: datetime if it parses, else numeric, else text."""
    as_dt = parse_datetime(values, MISSING_TOKENS)
    if as_dt.notna().mean() >= 0.9:
        return as_dt
    as_num = coerce_numeric(values)
    if as_num.notna().mean() >= 0.9:
        return as_num
    return values.astype(str)


def _stratified(config: PreprocessingConfig, y: Any) -> bool:
    """Stratify only single-target classification (2-D labels cannot be stratified)."""
    return config.problem_type == ProblemType.CLASSIFICATION and np.ndim(y) == 1


def _stratify(config: PreprocessingConfig, y: np.ndarray) -> np.ndarray | None:
    return y if _stratified(config, y) else None


def make_holdout(
    X: pd.DataFrame, y: np.ndarray, config: PreprocessingConfig, extra: pd.DataFrame | None = None
) -> Split:
    """Split positions into dev and test. ``extra`` holds the group/order columns, which are
    not model features but drive the split."""
    n = len(X)
    positions = np.arange(n)
    if config.validation == ValidationScheme.GROUPED:
        groups = _column(config.group_column, X, extra).astype(str).to_numpy()
        splitter = GroupShuffleSplit(
            1, test_size=config.test_size, random_state=config.random_state
        )
        dev, test = next(splitter.split(positions, y, groups))
        return Split(np.sort(dev), np.sort(test), groups)
    if config.validation == ValidationScheme.TIME_ORDERED:
        return _time_holdout(_column(config.order_column, X, extra), config)
    dev, test = train_test_split(
        positions,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=_stratify(config, y),
    )
    return Split(np.sort(dev), np.sort(test), None)


def horizon_gap(config: PreprocessingConfig) -> int:
    """Time points left out between training and evaluation when forecasting h periods
    ahead: training targets must not be later than the test rows' information cutoff."""
    return max(0, config.forecast.horizon - 1) if config.forecast is not None else 0


def time_ranks(values: pd.Series) -> np.ndarray:
    """Dense rank of each row's time point (rows sharing a timestamp share a rank)."""
    key = order_key(values).reset_index(drop=True)
    return key.rank(method="dense").fillna(0).astype("int64").to_numpy()


def _time_holdout(values: pd.Series, config: PreprocessingConfig) -> Split:
    """The LATEST time points form the test set. Splitting on time points (not rows) keeps
    every timestamp on one side, which matters when several series share timestamps."""
    ranks = time_ranks(values)
    times = np.unique(ranks)
    n_test = max(1, int(round(len(times) * config.test_size)))
    cut = times[len(times) - n_test]
    order = np.lexsort((np.arange(len(ranks)), ranks))  # time order, stable
    dev = order[ranks[order] < cut - horizon_gap(config)]
    test = order[ranks[order] >= cut]
    return Split(dev, test, ranks)


class TimeBlockSplit:
    """Forward cross-validation on time POINTS (``groups`` = time ranks), with an optional
    gap of ``gap`` time points between each training block and its validation block."""

    def __init__(self, n_splits: int = 5, gap: int = 0) -> None:
        self.n_splits, self.gap = n_splits, gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None):  # noqa: ANN201 - generator
        ranks = np.arange(len(X)) if groups is None else np.asarray(groups)
        times = np.unique(ranks)
        inner = TimeSeriesSplit(self.n_splits, gap=self.gap)
        for train_t, test_t in inner.split(times):
            yield (
                np.flatnonzero(np.isin(ranks, times[train_t])),
                np.flatnonzero(np.isin(ranks, times[test_t])),
            )


def _column(name: str | None, X: pd.DataFrame, extra: pd.DataFrame | None) -> pd.Series:
    if not name:
        raise ValueError("This validation scheme needs a group/order column.")
    for frame in (extra, X):
        if frame is not None and name in frame.columns:
            return frame[name].reset_index(drop=True)
    raise ValueError(f"Column '{name}' not found for the validation scheme.")


def make_cv(config: PreprocessingConfig, y_dev: np.ndarray) -> Any:
    """CV splitter for the dev rows (rows must already be in time order for time_ordered)."""
    k = config.cv_folds
    classification = _stratified(config, y_dev)
    if config.validation == ValidationScheme.GROUPED:
        if classification:
            return StratifiedGroupKFold(k, shuffle=True, random_state=config.random_state)
        return GroupKFold(k)
    if config.validation == ValidationScheme.TIME_ORDERED:
        return TimeBlockSplit(k, gap=horizon_gap(config))
    if classification:
        return StratifiedKFold(k, shuffle=True, random_state=config.random_state)
    return KFold(k, shuffle=True, random_state=config.random_state)


def describe(config: PreprocessingConfig) -> str:
    """One-line explanation of the validation design (shown in the UI and report)."""
    k, share = config.cv_folds, f"{config.test_size:.0%}"
    if config.validation == ValidationScheme.GROUPED:
        return (
            f"Grouped by '{config.group_column}': whole groups are held out ({share} test) "
            f"and {k}-fold group CV keeps every group in one fold only."
        )
    if config.validation == ValidationScheme.TIME_ORDERED:
        return (
            f"Time-ordered by '{config.order_column}': the latest {share} of time points is "
            f"the test set; {k}-split forward CV always trains on the past only"
            + (
                f", with a gap of {horizon_gap(config)} period(s) for the forecast horizon."
                if horizon_gap(config)
                else "."
            )
        )
    stratified = _stratified(config, [0]) and not config.is_multi_output
    return (
        f"Random {'stratified ' if stratified else ''}"
        f"split ({share} test) and {k}-fold CV, for independent rows."
    )
