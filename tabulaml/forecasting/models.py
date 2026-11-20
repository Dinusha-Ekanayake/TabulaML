"""Forecasting baselines and the RF + ARIMA hybrid (kept deliberately simple).

* ``NaiveForecaster``: tomorrow looks like today. It predicts the most recent known value
  (the target lag ``h``) or, as "seasonal naive", the value one season ago. Every real
  model should beat it; if not, the model has learned nothing useful.
* ``HybridRFARIMARegressor``: the classic two-stage hybrid. ARIMA models the series' own
  pattern over time (trend and autocorrelation); a Random Forest then learns what ARIMA
  missed (its residuals) from the lag, calendar and weather features. The forecast is
  ARIMA's forecast plus the forest's correction.

Both expect rows in time order. The hybrid supports a single series, and it forecasts
the periods straight after its training rows (as in time-ordered CV, the held-out test
and "forecast the next periods").
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class NaiveForecaster(RegressorMixin, BaseEstimator):
    """Predicts the value of ``column`` (a target lag); the training mean if it is missing."""

    def __init__(self, column: str = "") -> None:
        self.column = column

    def fit(self, X: Any, y: Any) -> NaiveForecaster:
        if self.column not in pd.DataFrame(X).columns:
            raise ValueError(f"Naive baseline needs the column '{self.column}'.")
        self.fallback_ = float(np.nanmean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        values = pd.to_numeric(pd.DataFrame(X)[self.column], errors="coerce")
        return values.fillna(self.fallback_).to_numpy(dtype=float)


class HybridRFARIMARegressor(RegressorMixin, BaseEstimator):
    """ARIMA(p, d, q) for the time pattern + Random Forest on ARIMA's residuals."""

    def __init__(
        self,
        p: int = 1,
        d: int = 1,
        q: int = 1,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.p, self.d, self.q = p, d, q
        self.n_estimators, self.max_depth, self.random_state = n_estimators, max_depth, random_state

    def fit(self, X: Any, y: Any) -> HybridRFARIMARegressor:
        from statsmodels.tsa.arima.model import ARIMA  # imported lazily: optional at startup

        y = np.asarray(y, dtype=float)
        if len(y) < max(10, 3 * (self.p + self.d + self.q)):
            raise ValueError("RF + ARIMA needs a longer series in every training fold.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.arima_ = ARIMA(y, order=(self.p, self.d, self.q)).fit()
        fitted = np.asarray(self.arima_.fittedvalues, dtype=float)
        fitted[: self.d] = y[: self.d]  # differencing has no fitted value for the first rows
        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y - fitted)
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = np.asarray(self.arima_.forecast(steps=n), dtype=float)
        return base + self.forest_.predict(X)
