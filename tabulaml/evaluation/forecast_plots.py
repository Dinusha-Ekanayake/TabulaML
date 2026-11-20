"""Over-time charts for forecasting: held-out test period and future forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tabulaml.evaluation.plots import PRIMARY, REFERENCE, SERIES, _layout

HISTORY_POINTS = 104  # show about the last two years of weekly data before the forecast


def _line(x: Any, y: Any, name: str, color: str, dash: str | None = None) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 2, "dash": dash},
        marker={"size": 5},
        hovertemplate=f"{name}<br>%{{x}}<br>%{{y:,.2f}}<extra></extra>",
    )


def test_period_fig(
    when: Any, actual: Any, predicted: Any, model: str, naive: Any = None
) -> go.Figure:
    """Actual vs forecast over the held-out test period (in time order)."""
    order = np.argsort(np.asarray(when))
    x = np.asarray(when)[order]
    traces = [
        _line(x, np.asarray(actual, float)[order], "Actual", REFERENCE),
        _line(x, np.asarray(predicted, float)[order], model, PRIMARY),
    ]
    if naive is not None:
        traces.append(_line(x, np.asarray(naive, float)[order], "Naive baseline", SERIES[1], "dot"))
    return _layout(go.Figure(traces), "Held-out test period: actual vs forecast", "Time", "Value")


def future_fig(
    history: pd.DataFrame, forecast: pd.DataFrame, time_col: str, target: str
) -> go.Figure:
    """Recent history followed by the forecast for the next periods (one series)."""
    recent = history.sort_values(time_col).tail(HISTORY_POINTS)
    fig = go.Figure(
        [
            _line(recent[time_col], recent[target], "History", REFERENCE),
            _line(forecast[time_col], forecast["forecast"], "Forecast", PRIMARY, "dash"),
        ]
    )
    return _layout(fig, "History and forecast", "Time", target)
