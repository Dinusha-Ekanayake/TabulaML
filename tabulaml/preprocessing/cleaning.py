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
from tabulaml.core.types import FeatureType

PLAN_NUMERIC = "numeric"
PLAN_NUMERIC_STRING = "numeric_string"
PLAN_BOOLEAN = "boolean"
PLAN_DATETIME = "datetime"
PLAN_CATEGORY = "category"
_PLAN_FOR_TYPE = {
    FeatureType.NUMERIC: PLAN_NUMERIC_STRING,
    FeatureType.BOOLEAN: PLAN_BOOLEAN,
    FeatureType.DATETIME: PLAN_DATETIME,
    FeatureType.CATEGORICAL: PLAN_CATEGORY,
    FeatureType.IDENTIFIER: PLAN_CATEGORY,
    FeatureType.TEXT: PLAN_CATEGORY,
    FeatureType.CONSTANT: PLAN_CATEGORY,
}

# \u20ac = euro, \u00a3 = pound: written as escapes so file-encoding mishaps cannot
# corrupt the pattern. Thousands separators are removed separately (see _parse_numbers).
_CURRENCY_NOISE = re.compile(r"(?i)rs\.?|lkr|usd|\$|\u20ac|\u00a3|%|\s")
_DECIMAL_COMMA_LIKE = re.compile(r"^-?\d{1,3},\d{1,2}$")  # "1,25": decimal or thousands?
_CODE_LIKE = re.compile(r"^0\d+$")  # "00100": a code with meaningful leading zeros
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


def _parse_numbers(s: pd.Series, decimal: str = ".", thousands: str | None = None) -> pd.Series:
    """Parse numeric strings such as 'Rs. 1,250', '12%' or (decimal=",") '1.250,5'.

    Unparseable values become NaN; the cleaner's parse report counts them.
    """
    as_text = s.where(s.isna(), s.astype(str))
    text = as_text.str.replace(_CURRENCY_NOISE, "", regex=True)
    if decimal == ",":
        text = text.str.replace(thousands or ".", "", regex=False)
        text = text.str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(thousands or ",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def coerce_numeric(
    s: pd.Series,
    tokens: tuple[str, ...] = MISSING_TOKENS,
    decimal: str = ".",
    thousands: str | None = None,
) -> pd.Series:
    """Best-effort conversion of any Series to float64."""
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        return s.astype("float64")
    return _parse_numbers(normalize_missing(s, tokens), decimal, thousands).astype("float64")


def _looks_like_codes(present: pd.Series) -> bool:
    """All-digit strings where some keep a leading zero ('00100') are codes, not numbers."""
    text = present.astype(str)
    return bool(text.str.fullmatch(r"\d+").all() and text.str.match(_CODE_LIKE).any())


def number_format_warning(s: pd.Series, decimal: str = ".") -> str | None:
    """Warn when values like '1,25' could be a decimal comma rather than a thousands mark."""
    present = normalize_missing(s).dropna().astype(str)
    if decimal != "." or present.empty:
        return None
    if present.str.match(_DECIMAL_COMMA_LIKE).mean() >= 0.5:
        return (
            "Values like '1,25' may use a decimal comma. Set the number format to "
            "decimal ',' if that is the case; they are read as thousands now."
        )
    return None


def _looks_like_datetime(values: pd.Series) -> bool:
    sample = values.astype(str)
    sample = sample.sample(min(len(sample), _DATETIME_SAMPLE), random_state=0)
    if not sample.map(lambda v: bool(_DATE_SHAPE.search(v))).mean() >= _DATETIME_PARSE_RATIO:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed", utc=True)
    except (ValueError, TypeError, OverflowError):
        return False
    return bool(parsed.notna().mean() >= _DATETIME_PARSE_RATIO)


def _plan_for(
    s: pd.Series, tokens: tuple[str, ...], decimal: str = ".", thousands: str | None = None
) -> str:
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
    if _looks_like_codes(present):
        return PLAN_CATEGORY
    if _parse_numbers(present, decimal, thousands).notna().mean() >= _NUMERIC_PARSE_RATIO:
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
    # Numeric spellings are accepted at transform time (e.g. a batch file using 1/0).
    true_tokens = (*TRUE_TOKENS, "1", "1.0")
    false_tokens = (*FALSE_TOKENS, "0", "0.0")
    mapping = {**dict.fromkeys(true_tokens, 1.0), **dict.fromkeys(false_tokens, 0.0)}
    return lower.map(mapping).astype("float64")


def parse_datetime(s: pd.Series, tokens: tuple[str, ...]) -> pd.Series:
    """Parse to timezone-naive datetime64; unparseable values become NaT."""
    if is_datetime64_any_dtype(s):
        return s.dt.tz_convert(None) if s.dt.tz is not None else s
    # utc=True makes mixed offsets parse to one dtype; naive strings keep their wall time.
    text = normalize_missing(s, tokens)
    parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    return parsed.dt.tz_convert(None)


def clean_target(s: pd.Series, regression: bool) -> pd.Series:
    """Clean a target column the same way for validation and training.

    Regression: float64 via ``coerce_numeric``. Classification: stripped strings with
    placeholder tokens as NaN and case variants merged ('yes ' / 'Yes' -> one class).
    """
    if regression:
        return coerce_numeric(s)
    text = normalize_missing(s).map(lambda v: v if pd.isna(v) else str(v))
    canon = _canonical_forms(text)
    return text.map(lambda v: v if pd.isna(v) else canon[v.lower()]).astype(object)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Learn column coercions on ``fit`` and apply them in ``transform``.

    Output dtypes: float64 (numeric, boolean as 0/1), datetime64[ns], object (categories).
    Columns absent at transform time are added as NaN; extra columns are ignored.
    """

    def __init__(
        self,
        normalize_category_case: bool = True,
        missing_tokens: tuple[str, ...] = MISSING_TOKENS,
        column_types: dict[str, FeatureType] | None = None,
        decimal: str = ".",
        thousands: str | None = None,
    ) -> None:
        self.normalize_category_case = normalize_category_case
        self.missing_tokens = missing_tokens
        self.column_types = column_types
        self.decimal = decimal
        self.thousands = thousands

    def _plan(self, s: pd.Series, col: str) -> str:
        """The user's explicit type wins; otherwise infer from the values."""
        explicit = (self.column_types or {}).get(col)
        if explicit is not None:
            return _PLAN_FOR_TYPE[FeatureType(explicit)]
        return _plan_for(s, self.missing_tokens, self.decimal, self.thousands)

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataCleaner:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.plan_: dict[str, str] = {}
        self.category_maps_: dict[str, dict[str, str]] = {}
        for col in X.columns:
            plan = self._plan(X[col], col)
            self.plan_[col] = plan
            if plan == PLAN_CATEGORY and self.normalize_category_case:
                cleaned = normalize_missing(X[col], self.missing_tokens)
                self.category_maps_[col] = _canonical_forms(cleaned)
        return self

    def parse_report(self, X: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Per column: values that were present but could NOT be parsed (became missing)."""
        report: dict[str, dict[str, Any]] = {}
        for col in self.feature_names_in_:
            if col not in X.columns or self.plan_[col] == PLAN_CATEGORY:
                continue
            present = normalize_missing(X[col], self.missing_tokens).notna()
            failed = present & self._transform_column(X[col], col).isna()
            if failed.any():
                examples = X.loc[failed, col].astype(str).unique()[:3].tolist()
                report[str(col)] = {
                    "failed": int(failed.sum()),
                    "share": float(failed.mean()),
                    "examples": examples,
                }
        return report

    def _transform_column(self, s: pd.Series, col: str) -> pd.Series:
        plan = self.plan_[col]
        if plan in (PLAN_NUMERIC, PLAN_NUMERIC_STRING):
            # Same parser for both: a numeric training column may arrive as text
            # ("1,200", "?") in a batch file and must not silently become NaN.
            return coerce_numeric(s, self.missing_tokens, self.decimal, self.thousands)
        if plan == PLAN_BOOLEAN:
            return _to_boolean(s, self.missing_tokens)
        if plan == PLAN_DATETIME:
            return parse_datetime(s, self.missing_tokens)
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
