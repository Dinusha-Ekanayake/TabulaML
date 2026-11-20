"""Forecast the next ``horizon`` periods after the end of the history.

Standard practice after evaluation: the chosen model is refitted on ALL available history
(its settings were chosen and tested earlier), then future rows are created for the next
periods of every series. Because every feature is at least ``horizon`` periods old, all
the values the model needs for those rows are already known.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from tabulaml.core.types import PreprocessingConfig
from tabulaml.forecasting.features import FUTURE_FLAG, add_forecast_features, infer_step
from tabulaml.training.splits import order_key


def _future_rows(history: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    fc = config.forecast
    when = history[config.order_column]
    step = infer_step(when)
    series_col = fc.series_column
    groups = history.groupby(series_col) if series_col else [(None, history)]
    rows: list[dict[str, Any]] = []
    for key, part in groups:
        last = part[config.order_column].max()
        for k in range(1, fc.horizon + 1):
            row = {config.order_column: last + k * step, FUTURE_FLAG: True, "__step": k}
            if series_col:
                row[series_col] = key
            rows.append(row)
    return pd.DataFrame(rows)


def forecast_future(
    pipeline: Any,
    history: pd.DataFrame,
    config: PreprocessingConfig,
    feature_columns: list[str] | tuple[str, ...],
    refit: bool = True,
) -> tuple[pd.DataFrame, Any]:
    """Return (forecast table, model used). The table has the time, the series (if any),
    the step ahead (1..horizon) and the forecast."""
    from tabulaml.training.trainer import prepare  # local import: trainer imports features

    if config.forecast is None:
        raise ValueError("This model was not trained for forecasting.")
    hist = history.copy()
    hist[config.order_column] = order_key(hist[config.order_column])
    if hist[config.order_column].isna().all():
        raise ValueError(f"'{config.order_column}' has no readable dates or numbers.")
    model = pipeline
    if refit:
        prep = prepare(hist, config)
        model = clone(pipeline).fit(prep.X, prep.y)
    future = _future_rows(hist, config)
    combined = pd.concat([hist.assign(**{FUTURE_FLAG: False}), future], ignore_index=True)
    built = add_forecast_features(combined, config, drop_incomplete=False)
    rows = built[built[FUTURE_FLAG].fillna(False).astype(bool)]
    X = rows.reindex(columns=list(feature_columns))
    prediction = np.asarray(model.predict(X), dtype=float).ravel()
    keep = [c for c in (config.forecast.series_column, config.order_column) if c]
    table = rows[keep].assign(step_ahead=rows["__step"].astype(int), forecast=prediction)
    return table.reset_index(drop=True), model
