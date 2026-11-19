"""Generic, schema-learned data cleaning (requirement P1).

``DataCleaner`` learns a per-column coercion plan from the training data and applies it to any
later frame (including single prediction records). It never mutates its input.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from tabulaml.config import FALSE_TOKENS, MISSING_TOKENS, TRUE_TOKENS

PLAN_NUMERIC = "numeric"
PLAN_NUMERIC_STRING = "numeric_string"
PLAN_BOOLEAN = "boolean"
PLAN_DATETIME = "datetime"
PLAN_CATEGORY = "category"

_CURRENCY_NOISE = re.compile(r"(?i)rs\.?|lkr|usd|\$|€|£|%|,|\s")
_BOOL_TOKENS = frozenset(TRUE_TOKENS) | frozenset(FALSE_TOKENS)
_NUMERIC_PARSE_RATIO = 0.95
_DATETIME_PARSE_RATIO = 0.90
_DATETIME_SAMPLE = 1000
_DATE_SHAPE = re.compile(r"\d.*[-/:. ]|[-/:. ].*\d")


def normalize_missing(s: pd.Series, tokens: tuple[str, ...] = MISSING_TOKENS) -> pd.Series:
    """Strip strings and turn placeholder tokens into NaN. Returns a new object Series."""
    token_set = {t.lower() for t in tokens}

    def fix(value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return np.nan if value.lower() in token_set else value
        return value

    return s.astype(object).map(fix)


def _parse_numbers(s: pd.Series) -> pd.Series:
    """Parse numeric strings such as 'Rs. 1,250' or '12%'. Unparseable values become NaN."""
    as_text = s.where(s.isna(), s.astype(str))
    return pd.to_numeric(as_text.str.replace(_CURRENCY_NOISE, "", regex=True), errors="coerce")


def coerce_numeric(s: pd.Series, tokens: tuple[str, ...] = MISSING_TOKENS) -> pd.Series:
    """Best-effort conversion of any Series to float64 (used for target validation)."""
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        return s.astype("float64")
    return _parse_numbers(normalize_missing(s, tokens)).astype("float64")


def _looks_like_datetime(values: pd.Series) -> bool:
    sample = values.astype(str)
    sample = sample.sample(min(len(sample), _DATETIME_SAMPLE), random_state=0)
    if not sample.map(lambda v: bool(_DATE_SHAPE.search(v))).mean() >= _DATETIME_PARSE_RATIO:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= _DATETIME_PARSE_RATIO)


def _plan_for(s: pd.Series, tokens: tuple[str, ...]) -> str:
    if is_bool_dtype(s):
        return PLAN_BOOLEAN
    if is_datetime64_any_dtype(s):
        return PLAN_DATETIME
    if is_numeric_dtype(s):
        return PLAN_NUMERIC
    present = normalize_missing(s, tokens).dropna()
    if present.empty:
        return PLAN_CATEGORY
    if present.astype(str).str.lower().isin(_BOOL_TOKENS).all():
        return PLAN_BOOLEAN
    if _parse_numbers(present).notna().mean() >= _NUMERIC_PARSE_RATIO:
        return PLAN_NUMERIC_STRING
    if _looks_like_datetime(present):
        return PLAN_DATETIME
    return PLAN_CATEGORY


def _canonical_forms(s: pd.Series) -> dict[str, str]:
    """Map lowercase -> most frequent spelling, so 'colombo ', 'COLOMBO' become 'Colombo'."""
    counts = s.dropna().astype(str).value_counts()
    canon: dict[str, str] = {}
    for value in counts.index:  # most frequent first
        canon.setdefault(value.lower(), value)
    return canon


def _to_boolean(s: pd.Series, tokens: tuple[str, ...]) -> pd.Series:
    if is_bool_dtype(s):
        return s.astype("float64")
    lower = normalize_missing(s, tokens).map(lambda v: v if pd.isna(v) else str(v).lower())
    mapping = {**dict.fromkeys(TRUE_TOKENS, 1.0), **dict.fromkeys(FALSE_TOKENS, 0.0)}
    return lower.map(mapping).astype("float64")


class DataCleaner(BaseEstimator, TransformerMixin):
    """Learn column coercions on ``fit`` and apply them in ``transform``.

    Output dtypes: float64 (numeric, boolean as 0/1), datetime64[ns], object (categories).
    Columns absent at transform time are added as NaN; extra columns are ignored.
    """

    def __init__(
        self,
        normalize_category_case: bool = True,
        missing_tokens: tuple[str, ...] = MISSING_TOKENS,
    ) -> None:
        self.normalize_category_case = normalize_category_case
        self.missing_tokens = missing_tokens

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataCleaner:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.plan_: dict[str, str] = {}
        self.category_maps_: dict[str, dict[str, str]] = {}
        for col in X.columns:
            plan = _plan_for(X[col], self.missing_tokens)
            self.plan_[col] = plan
            if plan == PLAN_CATEGORY and self.normalize_category_case:
                cleaned = normalize_missing(X[col], self.missing_tokens)
                self.category_maps_[col] = _canonical_forms(cleaned)
        return self

    def _transform_column(self, s: pd.Series, col: str) -> pd.Series:
        plan = self.plan_[col]
        if plan == PLAN_NUMERIC:
            return pd.to_numeric(s, errors="coerce").astype("float64")
        if plan == PLAN_NUMERIC_STRING:
            return _parse_numbers(normalize_missing(s, self.missing_tokens)).astype("float64")
        if plan == PLAN_BOOLEAN:
            return _to_boolean(s, self.missing_tokens)
        if plan == PLAN_DATETIME:
            text = normalize_missing(s, self.missing_tokens)
            return pd.to_datetime(text, errors="coerce", format="mixed")
        return self._clean_category(s, col)

    def _clean_category(self, s: pd.Series, col: str) -> pd.Series:
        text = normalize_missing(s, self.missing_tokens)
        text = text.map(lambda v: v if pd.isna(v) else str(v))
        canon = self.category_maps_.get(col)
        if not canon:
            return text
        mapped = text.map(lambda v: v if pd.isna(v) else canon.get(v.lower(), v))
        return mapped.astype(object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "plan_"):
            raise ValueError("DataCleaner is not fitted.")
        columns: dict[str, pd.Series] = {}
        for col in self.feature_names_in_:
            if col in X.columns:
                columns[col] = self._transform_column(X[col], col)
            else:
                columns[col] = pd.Series(np.nan, index=X.index, dtype="float64")
        return pd.DataFrame(columns, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: fit and apply a DataCleaner with default settings."""
    return DataCleaner().fit_transform(df)
