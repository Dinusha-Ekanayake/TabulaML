"""Step 2 (forecasting mode): time column, series column, horizon, lags and checks."""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import ForecastConfig, ValidationScheme
from tabulaml.forecasting.checks import forecast_issues
from tabulaml.forecasting.features import suggest_seasonal_lag
from tabulaml.training.splits import order_key
from tabulaml.ui.components import issues_view
from tabulaml.ui.state import AppState, reset_from

NO_SERIES = "(one series)"
WINDOW_CHOICES = (3, 4, 6, 8, 12, 26, 52)
DEFAULT_FOLDS = 5


def _pick_time_column(state: AppState, columns: list[str]) -> str:
    likely = [c for c in columns if order_key(state.raw_df[c]).dtype.kind == "M"]
    options = likely + [c for c in columns if c not in likely]
    current = state.order_column if state.order_column in options else options[0]
    return st.selectbox(
        "Date / time column",
        options,
        index=options.index(current),
        key="fc_time",
        help="Rows are put in time order with this column. Dates work best.",
    )


def _settings(state: AppState, time_col: str, columns: list[str]) -> ForecastConfig:
    old = state.forecast or ForecastConfig()
    series_opts = [NO_SERIES] + [c for c in columns if c != time_col]
    c1, c2, c3 = st.columns(3)
    series = c1.selectbox(
        "Series column (optional)",
        series_opts,
        index=series_opts.index(old.series_column) if old.series_column in series_opts else 0,
        key="fc_series",
        help="Only if the file holds several series, e.g. one per district or shop.",
    )
    horizon = c2.number_input(
        "Forecast horizon (periods ahead)", 1, 52, old.horizon, key="fc_horizon"
    )
    n_lags = c3.number_input(
        "Past values to use (lags)",
        1,
        24,
        old.n_lags,
        key="fc_lags",
        help=f"Lags {int(horizon)} to {int(horizon) + 2} use the values that many periods ago.",
    )
    suggested = suggest_seasonal_lag(order_key(state.raw_df[time_col]))
    c4, c5, c6 = st.columns(3)
    seasonal = c4.number_input(
        "Seasonal lag (0 = none)",
        0,
        400,
        old.seasonal_lag if state.forecast else (suggested or 0),
        key="fc_season",
        help="One season back, e.g. 52 for weekly data, 12 for monthly, 7 for daily.",
    )
    windows = c5.multiselect(
        "Rolling means over",
        WINDOW_CHOICES,
        default=[w for w in old.windows if w in WINDOW_CHOICES],
        key="fc_windows",
        help="Average of the last N known values: smooths out noise.",
    )
    exog = c6.radio(
        "Other columns",
        ["lagged", "none"],
        index=["lagged", "none"].index(old.exog),
        format_func=lambda v: "Use their past values" if v == "lagged" else "Do not use",
        key="fc_exog",
        help="Same-period values are not known when forecasting, so other columns are "
        "used with the same delay as the horizon.",
    )
    return ForecastConfig(
        horizon=int(horizon),
        n_lags=int(n_lags),
        seasonal_lag=int(seasonal) or None,
        windows=tuple(sorted(windows)),
        series_column=None if series == NO_SERIES else series,
        exog=exog,
    )


def forecast_section(state: AppState) -> bool:
    """Forecast settings; returns False when the data cannot be forecast with them."""
    st.subheader("Forecast settings")
    st.caption(
        "Forecasting is regression on the series' own past: every input is at least "
        "'horizon' periods old, and the model is always tested on later periods."
    )
    columns = [str(c) for c in state.raw_df.columns if str(c) != state.target]
    if not columns:
        st.error("Forecasting needs a date / time column besides the target.")
        return False
    time_col = _pick_time_column(state, columns)
    fc = _settings(state, time_col, columns)
    changes = {
        "forecast": fc,
        "order_column": time_col,
        "validation": ValidationScheme.TIME_ORDERED,
        "group_column": None,
    }
    if any(getattr(state, k) != v for k, v in changes.items()):
        reset_from(3, **changes)
    issues = forecast_issues(state.raw_df, state.target, time_col, fc, DEFAULT_FOLDS)
    issues_view(issues)
    if not issues:
        st.success("The series is regular and long enough for these settings.", icon="✅")
    return not any(i.level == "error" for i in issues)
