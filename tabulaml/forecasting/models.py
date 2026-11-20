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
from sklearn.base import BaseEstimator, RegressorMixin, clone
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
    """ARIMA(p, d, q) for the time pattern + a Random Forest pipeline on ARIMA's residuals.

    Fair to the other models: when predicting period t with horizon h, ARIMA only knows
    the series up to t - h. Those values come from the ``lag_column`` (the target h periods
    back), and the fitted ARIMA is updated with them without refitting its parameters.
    ``X`` is the raw feature table in time order, right after the training periods.
    """

    def __init__(
        self,
        lag_column: str = "",
        horizon: int = 1,
        residual_model: Any = None,
        p: int = 1,
        d: int = 1,
        q: int = 1,
    ) -> None:
        self.lag_column, self.horizon, self.residual_model = lag_column, horizon, residual_model
        self.p, self.d, self.q = p, d, q

    def _residual_model(self) -> Any:
        if self.residual_model is not None:
            return clone(self.residual_model)
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

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
        self.forest_ = self._residual_model().fit(X, y - fitted)
        return self

    def _arima_part(self, X: Any) -> np.ndarray:
        """h-step ARIMA forecasts, each from the values known h periods before the row."""
        h, n = max(int(self.horizon), 1), len(X)
        known = pd.to_numeric(pd.DataFrame(X)[self.lag_column], errors="coerce").to_numpy()
        results, used, out = self.arima_, 0, np.empty(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(n):
                new = max(0, j + 1 - h)  # actual values after training known for row j
                if new > used:
                    results = results.append(known[h + used : h + new], refit=False)
                    used = new
                out[j] = np.asarray(results.forecast(steps=j + 1 - used))[-1]
        return out

    def predict(self, X: Any) -> np.ndarray:
        return self._arima_part(X) + np.asarray(self.forest_.predict(X), dtype=float)
