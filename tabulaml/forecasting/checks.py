"""Checks that a table can be forecast honestly (shown in steps 2 and 3)."""

from __future__ import annotations

import pandas as pd

from tabulaml.core.types import ForecastConfig, ValidationIssue
from tabulaml.training.splits import order_key

MIN_PARSED = 0.9
HORIZON_SHARE_WARN = 0.25
GAP_FACTOR = 1.5


def _issue(level: str, code: str, message: str, column: str | None = None) -> ValidationIssue:
    return ValidationIssue(level, code, message, column)


def forecast_issues(
    df: pd.DataFrame, target: str, order_column: str | None, fc: ForecastConfig, cv_folds: int
) -> list[ValidationIssue]:
    """Errors block training; warnings explain risks."""
    if not order_column or order_column not in df.columns:
        return [_issue("error", "no_time_column", "Forecasting needs a date / time column.")]
    when = order_key(df[order_column])
    parsed = when.notna().mean() if when.dtype != object else 0.0
    if parsed < MIN_PARSED:
        return [
            _issue(
                "error",
                "time_not_parsed",
                f"'{order_column}' could not be read as dates or numbers for most rows.",
                order_column,
            )
        ]
    keys = [fc.series_column] if fc.series_column else []
    frame = pd.DataFrame({"when": when}).assign(**{k: df[k] for k in keys})
    issues = _duplicates(frame, keys, order_column)
    issues += _length(frame, keys, fc, cv_folds)
    issues += _gaps(frame, keys)
    return issues


def _duplicates(frame: pd.DataFrame, keys: list[str], order_column: str) -> list[ValidationIssue]:
    dup = int(frame.duplicated(subset=[*keys, "when"]).sum())
    if not dup:
        return []
    advice = (
        "Choose the column that identifies each series (for example a store or district)."
        if not keys
        else "Aggregate the data to one row per series and period first."
    )
    return [
        _issue(
            "error",
            "duplicate_times",
            f"{dup} rows share a time point within the same series. {advice}",
            order_column,
        )
    ]


def _length(
    frame: pd.DataFrame, keys: list[str], fc: ForecastConfig, folds: int
) -> list[ValidationIssue]:
    sizes = frame.groupby(keys).size() if keys else pd.Series([len(frame)])
    shortest = int(sizes.min())
    needed = max(fc.lags()) + fc.horizon + 2 * folds
    out = []
    if shortest < needed:
        out.append(
            _issue(
                "error",
                "series_too_short",
                f"The shortest series has {shortest} periods; these settings need at least "
                f"{needed} (largest lag + horizon + 2 x folds). Use fewer lags, a shorter "
                "horizon or fewer folds.",
            )
        )
    if fc.seasonal_lag and fc.seasonal_lag >= shortest / (folds + 1):
        out.append(
            _issue(
                "warning",
                "seasonal_lag_long",
                f"The seasonal lag ({fc.seasonal_lag}) is longer than the first "
                "cross-validation fold, so early folds cannot use it (it is filled in).",
            )
        )
    if fc.horizon > HORIZON_SHARE_WARN * shortest:
        out.append(
            _issue(
                "warning",
                "long_horizon",
                f"Forecasting {fc.horizon} periods ahead is long for a series of {shortest} "
                "periods; accuracy drops quickly with the horizon.",
            )
        )
    return out


def _gaps(frame: pd.DataFrame, keys: list[str]) -> list[ValidationIssue]:
    ordered = frame.sort_values([*keys, "when"])
    steps = ordered.groupby(keys)["when"].diff() if keys else ordered["when"].diff()
    steps = steps.dropna()
    if steps.empty:
        return []
    usual = steps.median()
    irregular = int((steps > usual * GAP_FACTOR).sum())
    if not irregular:
        return []
    return [
        _issue(
            "warning",
            "irregular_gaps",
            f"{irregular} gaps are longer than the usual period. Lags count ROWS, so a missing "
            "period makes some lags look further back than they are.",
        )
    ]
