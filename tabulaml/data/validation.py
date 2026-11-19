"""Target validation, leakage detection and supported-input checks (P4, P12, P14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import roc_auc_score

from tabulaml.config import LEAKAGE_AUC, LEAKAGE_CORR, MIN_ROWS
from tabulaml.core.types import (
    DatasetSchema,
    FeatureType,
    ProblemType,
    ValidationIssue,
)
from tabulaml.preprocessing.cleaning import coerce_numeric, normalize_missing

MAX_CLASSES_WARN = 50
LOW_CARD_TARGET = 10
NUMERIC_TARGET_RATIO = 0.95
LEAKAGE_SAMPLE_ROWS = 20_000
MIN_GROUP_SIZE_FOR_PURITY = 3
PURE_GROUP_FRACTION = 0.98
_SEED = 0


def _issue(level: str, code: str, message: str, column: str | None = None) -> ValidationIssue:
    return ValidationIssue(level, code, message, column)


def suggest_problem_type(s: pd.Series) -> ProblemType:
    """Guess classification vs regression from the target's values."""
    numeric = coerce_numeric(s)
    present = normalize_missing(s).dropna()
    is_numeric = len(present) > 0 and numeric.notna().sum() / len(present) >= NUMERIC_TARGET_RATIO
    if is_numeric and numeric.nunique() > LOW_CARD_TARGET:
        return ProblemType.REGRESSION
    return ProblemType.CLASSIFICATION


def _classification_issues(y: pd.Series, target: str, cv_folds: int) -> list[ValidationIssue]:
    counts = y.astype(str).value_counts()
    if len(counts) < 2:
        return [_issue("error", "single_class", "Target has fewer than 2 classes.", target)]
    issues: list[ValidationIssue] = []
    small = counts[counts < cv_folds]
    if len(small):
        names = ", ".join(map(str, small.index[:5]))
        msg = f"Classes with fewer than {cv_folds} rows (needed for {cv_folds}-fold CV): {names}."
        issues.append(_issue("error", "class_too_small", msg, target))
    if len(counts) > MAX_CLASSES_WARN:
        msg = f"Target has {len(counts)} distinct values; it may be continuous (try regression)."
        issues.append(_issue("warning", "many_classes", msg, target))
    return issues


def _regression_issues(y_raw: pd.Series, target: str) -> list[ValidationIssue]:
    numeric = coerce_numeric(y_raw)
    present = normalize_missing(y_raw).dropna()
    if len(present) == 0 or numeric.notna().sum() / len(present) < NUMERIC_TARGET_RATIO:
        return [_issue("error", "target_not_numeric", "Regression needs a numeric target.", target)]
    n_unique = numeric.nunique()
    if n_unique <= 1:
        return [_issue("error", "constant_target", "Target has a single value.", target)]
    if n_unique <= LOW_CARD_TARGET:
        msg = f"Target has only {n_unique} distinct values; it may be categorical."
        return [_issue("warning", "target_looks_categorical", msg, target)]
    return []


def validate_target(
    df: pd.DataFrame, target: str, problem_type: ProblemType, cv_folds: int
) -> list[ValidationIssue]:
    if target not in df.columns:
        return [_issue("error", "target_missing", f"Column '{target}' not found.", target)]
    y_raw = df[target]
    issues: list[ValidationIssue] = []
    n_missing = int(normalize_missing(y_raw).isna().sum())
    if n_missing:
        msg = f"{n_missing} rows have a missing target and will be dropped."
        issues.append(_issue("info", "target_missing_rows", msg, target))
    if n_missing == len(df):
        return [*issues, _issue("error", "target_all_missing", "Target is empty.", target)]
    if problem_type == ProblemType.REGRESSION:
        return issues + _regression_issues(y_raw, target)
    y = normalize_missing(y_raw).dropna()
    return issues + _classification_issues(y, target, cv_folds)


# ----------------------------------------------------------------------------- leakage


def _as_float(s: pd.Series) -> pd.Series | None:
    if is_datetime64_any_dtype(s):
        return pd.Series(s.astype("int64").where(s.notna()), index=s.index).astype("float64")
    if is_bool_dtype(s) or is_numeric_dtype(s):
        return s.astype("float64")
    return None


def _auc_strength(values: pd.Series, y_codes: np.ndarray) -> float:
    """Best single-feature one-vs-rest AUC, folded so 0.5 is chance and 1.0 is perfect."""
    filled = values.fillna(values.median() if values.notna().any() else 0.0).to_numpy()
    best = 0.5
    for cls in np.unique(y_codes):
        positive = y_codes == cls
        if positive.all() or not positive.any():
            continue
        auc = roc_auc_score(positive, filled)
        best = max(best, auc, 1 - auc)
    return float(best)


def _corr_strength(values: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([values, y], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2:
        return 0.0
    pearson = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    spearman = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
    return float(np.nanmax([pearson, spearman, 0.0]))


def _signal(values: pd.Series, y: pd.Series, y_codes: np.ndarray | None) -> tuple[float, float]:
    """Return (strength, threshold) of a numeric feature against the target."""
    if y_codes is None:
        return _corr_strength(values, y), LEAKAGE_CORR
    return _auc_strength(values, y_codes), LEAKAGE_AUC


def _categorical_leak(s: pd.Series, y: pd.Series, y_codes: np.ndarray | None) -> bool:
    valid = s.notna()
    if valid.sum() == 0 or s[valid].nunique() < 2:
        return False
    group_sizes = s[valid].map(s[valid].value_counts())
    if group_sizes.mean() < MIN_GROUP_SIZE_FOR_PURITY:
        return False
    if y_codes is not None:
        codes = pd.Series(y_codes, index=y.index)[valid]
        pure = codes.groupby(s[valid]).transform("nunique") == 1
        return bool(pure.mean() >= PURE_GROUP_FRACTION)
    yv = y[valid].astype(float)
    total = ((yv - yv.mean()) ** 2).sum()
    if total == 0:
        return False
    between = ((yv.groupby(s[valid]).transform("mean") - yv.mean()) ** 2).sum()
    return bool(between / total >= LEAKAGE_CORR**2)


def _check_column(col: str, s: pd.Series, y: pd.Series, y_codes: np.ndarray | None) -> str | None:
    """Return a human-readable reason when ``s`` looks like leakage, else None."""
    if s.astype(str).equals(y.astype(str)):
        return "is identical to the target"
    missing = s.isna()
    if 0 < missing.sum() < len(s):
        strength, threshold = _signal(missing.astype("float64"), y, y_codes)
        if strength >= threshold:
            return "its missingness almost perfectly predicts the target"
    numeric = _as_float(s)
    if numeric is not None:
        strength, threshold = _signal(numeric, y, y_codes)
        if strength >= threshold:
            return f"is almost perfectly related to the target (strength {strength:.3f})"
        return None
    if _categorical_leak(s, y, y_codes):
        return "maps almost one-to-one onto the target"
    return None


def detect_leakage(
    df: pd.DataFrame, target: str, problem_type: ProblemType
) -> list[ValidationIssue]:
    """Flag features that carry (nearly) the target itself. Expects a cleaned frame."""
    if target not in df.columns:
        return []
    keep = df[target].notna()
    data = df[keep]
    if len(data) > LEAKAGE_SAMPLE_ROWS:
        data = data.sample(LEAKAGE_SAMPLE_ROWS, random_state=_SEED)
    if problem_type == ProblemType.REGRESSION:
        y = coerce_numeric(data[target])
        y_codes = None
    else:
        y = data[target]
        y_codes = pd.factorize(y.astype(str))[0]
    issues: list[ValidationIssue] = []
    for col in data.columns:
        if col == target or data[col].nunique(dropna=True) <= 1:
            continue
        reason = _check_column(col, data[col], y, y_codes)
        if reason:
            msg = (
                f"'{col}' looks like target leakage: {reason}. "
                "Exclude it unless it is known at prediction time."
            )
            issues.append(_issue("warning", "leakage", msg, col))
    return issues


def check_supported(df: pd.DataFrame, schema: DatasetSchema, target: str) -> list[ValidationIssue]:
    """Explain unsupported inputs instead of silently proceeding (P14)."""
    issues: list[ValidationIssue] = []
    if len(df) < MIN_ROWS:
        msg = f"Only {len(df)} rows; at least {MIN_ROWS} are needed."
        issues.append(_issue("error", "too_few_rows", msg))
    features = [c for c in schema.columns if c.name != target]
    unusable = {FeatureType.IDENTIFIER, FeatureType.CONSTANT}
    usable = [c for c in features if c.inferred_type not in unusable]
    if not usable:
        msg = "No usable features: every other column is an identifier or constant."
        issues.append(_issue("error", "no_usable_features", msg))
    elif all(c.inferred_type == FeatureType.TEXT for c in usable):
        msg = "All features are free text, which TabulaML does not support."
        issues.append(_issue("error", "all_text_features", msg))
    return issues
