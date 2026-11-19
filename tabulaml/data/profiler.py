"""Schema profiling of a cleaned DataFrame (requirement P2)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy.stats import skew

from tabulaml.config import (
    HIGH_CARD_THRESHOLD,
    ID_UNIQUE_RATIO,
    MOSTLY_MISSING,
    TEXT_AVG_LEN,
)
from tabulaml.core.types import ColumnProfile, DatasetSchema, FeatureType

_ID_TOKENS = frozenset({"id", "uuid", "guid", "code", "number", "key", "no"})
_LOW_CARD_NUMERIC = 10
_SAMPLE_VALUES = 5
_MIN_SKEW_VALUES = 3


def _name_looks_like_id(name: str) -> bool:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    return any(tok in _ID_TOKENS for tok in re.split(r"[^a-z0-9]+", spaced))


def _is_integer_like(values: pd.Series) -> bool:
    return bool(np.all(np.isclose(values, np.round(values))))


def _is_row_counter(values: pd.Series) -> bool:
    """Consecutive integers (0..n, 1001..1500), in any row order: a row index, not a feature.

    Sortedness alone is not used: a file sorted by a genuine integer feature would otherwise
    lose that feature by default.
    """
    n_unique = values.nunique()
    return bool(values.max() - values.min() + 1 == n_unique)


def _numeric_kind(s: pd.Series, present: pd.Series, unique_ratio: float) -> FeatureType:
    if set(present.unique()) <= {0, 1}:
        return FeatureType.BOOLEAN
    if unique_ratio > ID_UNIQUE_RATIO and _is_integer_like(present):
        if _name_looks_like_id(str(s.name)) or _is_row_counter(present):
            return FeatureType.IDENTIFIER
    return FeatureType.NUMERIC


def _text_kind(present: pd.Series, unique_ratio: float) -> FeatureType:
    avg_len = present.astype(str).str.len().mean()
    if unique_ratio > ID_UNIQUE_RATIO:
        return FeatureType.TEXT if avg_len > TEXT_AVG_LEN else FeatureType.IDENTIFIER
    if avg_len > TEXT_AVG_LEN and unique_ratio > 0.5:
        return FeatureType.TEXT
    return FeatureType.CATEGORICAL


def infer_feature_type(s: pd.Series, n_rows: int) -> FeatureType:
    """Infer a column's semantic type from its (already cleaned) values."""
    present = s.dropna()
    n_unique = present.nunique()
    if n_unique <= 1:
        return FeatureType.CONSTANT
    unique_ratio = n_unique / max(len(present), 1)
    if is_bool_dtype(s):
        return FeatureType.BOOLEAN
    if is_datetime64_any_dtype(s):
        return FeatureType.DATETIME
    if is_numeric_dtype(s):
        return _numeric_kind(s, present, unique_ratio)
    if _name_looks_like_id(str(s.name)) and unique_ratio > ID_UNIQUE_RATIO:
        return FeatureType.IDENTIFIER
    return _text_kind(present, unique_ratio)


def _skewness(present: pd.Series) -> float | None:
    if len(present) < _MIN_SKEW_VALUES or present.nunique() < 2:
        return None
    value = float(skew(present.astype(float), nan_policy="omit"))
    return None if np.isnan(value) else value


def _outlier_ratio(present: pd.Series) -> float | None:
    if len(present) < 4:
        return None
    q1, q3 = present.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    outside = (present < q1 - 1.5 * iqr) | (present > q3 + 1.5 * iqr)
    return float(outside.mean())


def _warnings(ftype: FeatureType, missing: float, n_unique: int) -> tuple[str, ...]:
    out: list[str] = []
    if missing > MOSTLY_MISSING:
        out.append("mostly_missing")
    if ftype == FeatureType.IDENTIFIER:
        out.append("identifier")
    if ftype == FeatureType.CONSTANT:
        out.append("constant")
    if ftype == FeatureType.CATEGORICAL and n_unique > HIGH_CARD_THRESHOLD:
        out.append("high_cardinality")
    if ftype == FeatureType.NUMERIC and n_unique <= _LOW_CARD_NUMERIC:
        out.append("low_card_numeric")
    return tuple(out)


def _profile_column(df: pd.DataFrame, name: str) -> ColumnProfile:
    s = df[name]
    present = s.dropna()
    n_unique = int(present.nunique())
    ftype = infer_feature_type(s, len(df))
    numeric = ftype in (FeatureType.NUMERIC, FeatureType.BOOLEAN)
    missing = float(s.isna().mean()) if len(s) else 0.0
    return ColumnProfile(
        name=name,
        raw_dtype=str(s.dtype),
        inferred_type=ftype,
        missing_ratio=missing,
        n_unique=n_unique,
        unique_ratio=n_unique / max(len(present), 1),
        sample_values=tuple(str(v) for v in present.drop_duplicates().head(_SAMPLE_VALUES)),
        skewness=_skewness(present) if ftype == FeatureType.NUMERIC else None,
        outlier_ratio=_outlier_ratio(present) if numeric else None,
        warnings=_warnings(ftype, missing, n_unique),
    )


def profile_dataset(df_clean: pd.DataFrame) -> DatasetSchema:
    """Profile every column of a cleaned frame."""
    columns = tuple(_profile_column(df_clean, str(c)) for c in df_clean.columns)
    return DatasetSchema(
        columns=columns,
        n_rows=len(df_clean),
        n_duplicate_rows=int(df_clean.duplicated().sum()),
    )
