"""Custom leakage-safe transformers used by the pipeline builder.

All learn their parameters in ``fit`` only (so inside a CV fold they only see that fold's
training rows), accept DataFrames or arrays, and implement ``get_feature_names_out``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from tabulaml.config import MISSING_TOKENS
from tabulaml.preprocessing.cleaning import coerce_numeric, parse_datetime

DATETIME_PARTS = ("year", "month", "dayofweek", "dayofyear", "days_since_ref")
_EPOCH = pd.Timestamp("1970-01-01")


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X, dtype=object)
    return pd.DataFrame(arr.reshape(len(arr), -1))


def _remember_input(est: BaseEstimator, X: Any) -> None:
    if isinstance(X, pd.DataFrame):
        est.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
    est.n_features_in_ = _as_frame(X).shape[1]


def _input_names(est: BaseEstimator) -> list[str]:
    names = getattr(est, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    return [f"x{i}" for i in range(est.n_features_in_)]


class ToNumeric(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Coerce every column to float (so a user type override to numeric never crashes)."""

    def fit(self, X: Any, y: Any = None) -> ToNumeric:
        _remember_input(self, X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X)
        return np.column_stack([coerce_numeric(frame[c]).to_numpy() for c in frame.columns])


def _category_text(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))  # 1.0 at fit and 1 at predict must be the same category
    return str(value)


class ToCategory(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Represent every value as a string category; missing stays NaN."""

    def fit(self, X: Any, y: Any = None) -> ToCategory:
        _remember_input(self, X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X)
        cols = [frame[c].map(_category_text).to_numpy(dtype=object) for c in frame.columns]
        return np.column_stack(cols)


class DatetimeFeatures(TransformerMixin, BaseEstimator):
    """Expand each datetime column into year, month, weekday, day-of-year and age in days.

    The reference date for ``days_since_ref`` is the training minimum, learned in ``fit``.
    """

    def fit(self, X: Any, y: Any = None) -> DatetimeFeatures:
        _remember_input(self, X)
        frame = _as_frame(X)
        self.ref_ = []
        for col in frame.columns:
            minimum = parse_datetime(frame[col], MISSING_TOKENS).min()
            self.ref_.append(_EPOCH if pd.isna(minimum) else minimum)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X)
        blocks = []
        for ref, col in zip(self.ref_, frame.columns, strict=True):
            dt = parse_datetime(frame[col], MISSING_TOKENS)
            blocks.extend(
                [
                    dt.dt.year,
                    dt.dt.month,
                    dt.dt.dayofweek,
                    dt.dt.dayofyear,
                    (dt - ref).dt.days,
                ]
            )
        return np.column_stack([b.astype("float64").to_numpy() for b in blocks])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = _input_names(self)
        return np.asarray([f"{n}_{p}" for n in names for p in DATETIME_PARTS], dtype=object)


class TextStats(TransformerMixin, BaseEstimator):
    """Character length and word count of free-text columns."""

    def fit(self, X: Any, y: Any = None) -> TextStats:
        _remember_input(self, X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X)
        blocks = []
        for col in frame.columns:
            text = frame[col].map(lambda v: np.nan if pd.isna(v) else str(v))
            blocks.append(text.str.len())
            blocks.append(text.str.split().str.len())
        return np.column_stack([b.astype("float64").to_numpy() for b in blocks])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names = _input_names(self)
        return np.asarray([f"{n}_{p}" for n in names for p in ("chars", "words")], dtype=object)


class OutlierClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to [Q1 - k*IQR, Q3 + k*IQR] learned on the training data.

    Columns with zero IQR (e.g. 0/1 missing indicators) are left untouched.
    """

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X: Any, y: Any = None) -> OutlierClipper:
        _remember_input(self, X)
        arr = np.asarray(X, dtype="float64")
        q1, q3 = np.nanpercentile(arr, [25, 75], axis=0)
        iqr = q3 - q1
        degenerate = ~(iqr > 0)
        self.lower_ = np.where(degenerate, -np.inf, q1 - self.factor * iqr)
        self.upper_ = np.where(degenerate, np.inf, q3 + self.factor * iqr)
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.clip(np.asarray(X, dtype="float64"), self.lower_, self.upper_)


class SkewCorrector(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Yeo-Johnson on columns whose training skew exceeds ``threshold`` (0 = every column)."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> SkewCorrector:
        _remember_input(self, X)
        arr = np.asarray(X, dtype="float64")
        selected = []
        for j in range(arr.shape[1]):
            col = arr[:, j][~np.isnan(arr[:, j])]
            if len(np.unique(col)) < 3:
                continue
            if abs(float(skew(col))) > self.threshold:
                selected.append(j)
        self.columns_ = np.asarray(selected, dtype=int)
        self.power_ = None
        if len(self.columns_):
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            self.power_ = pt.fit(arr[:, self.columns_])
        return self

    def transform(self, X: Any) -> np.ndarray:
        arr = np.array(X, dtype="float64", copy=True)
        if self.power_ is not None:
            arr[:, self.columns_] = self.power_.transform(arr[:, self.columns_])
        return arr
