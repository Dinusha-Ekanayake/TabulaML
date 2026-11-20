"""Turn a time series into a supervised-learning table (forecasting as regression).

For a horizon of h periods every generated feature looks at least h periods into the PAST:

* target lags   ``<target>__lag<k>``         for k = h, h+1, ... (plus a seasonal lag);
* rolling means ``<target>__mean<w>``        the mean of the w values ending h periods ago;
* calendar      ``__month``, ``__week``, ...  parts of the date (known in advance);
* other columns ``<col>__lag<h>``            their value h periods ago (optional).

So every value is known when the forecast is made, and nothing is fitted here (no learned
parameters), which is why this can safely run before the train/test split. Rows at the
start of each series without the h-step lag are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabulaml.core.types import ForecastConfig, PreprocessingConfig
from tabulaml.preprocessing.cleaning import clean_target
from tabulaml.training.splits import order_key

FUTURE_FLAG = "__future__"
CALENDAR_PARTS = ("month", "week", "dayofweek", "quarter")


def lag_name(target: str, k: int) -> str:
    return f"{target}__lag{k}"


def mean_name(target: str, w: int) -> str:
    return f"{target}__mean{w}"


def _calendar(when: pd.Series) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(when):
        return pd.DataFrame(index=when.index)
    parts = {
        "__month": when.dt.month,
        "__week": when.dt.isocalendar().week.astype("float"),
        "__dayofweek": when.dt.dayofweek,
        "__quarter": when.dt.quarter,
    }
    out = pd.DataFrame({k: v.astype(float) for k, v in parts.items()}, index=when.index)
    return out.loc[:, out.nunique(dropna=True) > 1]  # constant parts carry no information


def role_columns(config: PreprocessingConfig) -> set[str]:
    fc = config.forecast
    return {c for c in (config.order_column, fc.series_column if fc else None) if c}


def generated_columns(df: pd.DataFrame, config: PreprocessingConfig) -> list[str]:
    """Names of the columns ``add_forecast_features`` creates (for config and the UI)."""
    built = add_forecast_features(df, config, drop_incomplete=False)
    return [c for c in built.columns if c not in df.columns and c != FUTURE_FLAG]


def already_built(df: pd.DataFrame, config: PreprocessingConfig) -> bool:
    fc = config.forecast
    return fc is not None and lag_name(config.target, fc.horizon) in df.columns


def add_forecast_features(
    df: pd.DataFrame, config: PreprocessingConfig, drop_incomplete: bool = True
) -> pd.DataFrame:
    """Return a NEW frame (original index kept, time order) with the forecast features."""
    fc: ForecastConfig = config.forecast
    if fc is None or already_built(df, config):
        return df
    if not config.order_column or config.order_column not in df.columns:
        raise ValueError("Forecasting needs a date / time column.")
    when = order_key(df[config.order_column])
    keys = [fc.series_column] if fc.series_column else []
    frame = df.assign(__when=when).sort_values([*keys, "__when"], kind="stable")
    groups = frame.groupby(keys, sort=False) if keys else None
    y = clean_target(frame[config.target], regression=True)

    def past(values: pd.Series, k: int) -> pd.Series:
        return values.groupby(frame[fc.series_column]).shift(k) if groups else values.shift(k)

    new: dict[str, pd.Series] = {}
    for k in fc.lags():
        new[lag_name(config.target, k)] = past(y, k)
    shifted = past(y, fc.horizon)
    for w in fc.windows:
        rolled = (
            shifted.groupby(frame[fc.series_column]).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )
            if groups
            else shifted.rolling(w, min_periods=1).mean()
        )
        new[mean_name(config.target, w)] = rolled
    if fc.exog == "lagged":
        for col in _exog_columns(frame, config):
            new[f"{col}__lag{fc.horizon}"] = past(frame[col], fc.horizon)
    out = pd.concat(
        [frame, pd.DataFrame(new, index=frame.index), _calendar(frame["__when"])], axis=1
    )
    out = out.drop(columns="__when")
    if drop_incomplete:
        out = out[out[lag_name(config.target, fc.horizon)].notna()]
    return out


def _exog_columns(frame: pd.DataFrame, config: PreprocessingConfig) -> list[str]:
    skip = {*config.targets, *role_columns(config), "__when", FUTURE_FLAG}
    return [c for c in frame.columns if c not in skip]


def infer_step(when: pd.Series) -> object:
    """The usual step between periods: a pandas frequency, a Timedelta or a number."""
    values = pd.Series(when.dropna().unique()).sort_values()
    if len(values) < 2:
        raise ValueError("Need at least two time points to find the period length.")
    if pd.api.types.is_datetime64_any_dtype(values):
        freq = pd.infer_freq(pd.DatetimeIndex(values)) if len(values) >= 3 else None
        return pd.tseries.frequencies.to_offset(freq) if freq else values.diff().median()
    return float(np.median(np.diff(values.to_numpy(dtype=float))))


def suggest_seasonal_lag(when: pd.Series) -> int | None:
    """Seasonal lag for common period lengths (weekly 52, monthly 12, daily 7, ...)."""
    if not pd.api.types.is_datetime64_any_dtype(when):
        return None
    step = pd.Series(when.dropna().unique()).sort_values().diff().median()
    if pd.isna(step):
        return None
    days = step / pd.Timedelta(days=1)
    for low, high, lag in ((0.9, 1.1, 7), (6, 8, 52), (27, 32, 12), (88, 93, 4)):
        if low <= days <= high:
            return lag
    if days < 0.1:
        return 24
    return None
