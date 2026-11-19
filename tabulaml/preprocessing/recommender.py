"""Automatic, explained preprocessing recommendations (Automatic mode, P5 and P15).

Only decisions that do not depend on the model family are made explicit here (include or
exclude, target encoding for high cardinality, imbalance, target transform). Everything
that does depend on it stays "auto" and is resolved by the builder per profile, on
training folds only.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy.stats import skew

from tabulaml.config import (
    HIGH_CARD_THRESHOLD,
    IMBALANCE_MINORITY,
    MOSTLY_MISSING,
    OUTLIER_RATIO_CLIP,
    RARE_MIN_FREQ,
    SKEW_THRESHOLD,
)
from tabulaml.core.types import (
    ColumnConfig,
    ColumnProfile,
    DatasetSchema,
    FeatureType,
    ImbalanceStrategy,
    PreprocessingConfig,
    ProblemType,
    ValidationIssue,
)
from tabulaml.preprocessing.cleaning import clean_target

_DROP_TYPES = {
    FeatureType.IDENTIFIER: "identifier (unique per row), so it carries no general pattern",
    FeatureType.CONSTANT: "constant (a single value), so it carries no information",
    FeatureType.TEXT: "free text; TabulaML does not model long text by default",
}


def _exclusion_reason(col: ColumnProfile, leaks: dict[str, str]) -> str | None:
    if col.name in leaks:
        return f"Excluded: possible target leakage ({leaks[col.name]})."
    if col.inferred_type in _DROP_TYPES:
        return f"Excluded: {_DROP_TYPES[col.inferred_type]}."
    if col.missing_ratio > MOSTLY_MISSING:
        return f"Excluded: {col.missing_ratio:.0%} missing (more than {MOSTLY_MISSING:.0%})."
    return None


def _numeric_reason(col: ColumnProfile) -> str:
    parts = [f"Numeric; {col.missing_ratio:.1%} missing -> median imputation"]
    if col.missing_ratio > 0:
        parts[0] += " plus a missing-indicator"
    if col.skewness is not None and abs(col.skewness) > SKEW_THRESHOLD:
        parts.append(f"skew {col.skewness:.2f} -> Yeo-Johnson for scale-sensitive models")
    if col.outlier_ratio and col.outlier_ratio > OUTLIER_RATIO_CLIP:
        parts.append(f"{col.outlier_ratio:.1%} IQR outliers -> extreme values clipped")
    parts.append("standard scaling for scale-sensitive models; untouched for tree models")
    return "; ".join(parts) + "."


def _categorical(col: ColumnProfile) -> ColumnConfig:
    if col.n_unique > HIGH_CARD_THRESHOLD:
        reason = (
            f"Categorical with {col.n_unique} levels (> {HIGH_CARD_THRESHOLD}) -> cross-fitted "
            "target encoding; missing values become their own category."
        )
        return ColumnConfig(col.name, col.inferred_type, encoding="target", reason=reason)
    reason = (
        f"Categorical with {col.n_unique} levels -> one-hot for scale-sensitive models, "
        f"ordinal for tree models; levels under {RARE_MIN_FREQ:.0%} grouped; unseen levels "
        "handled at prediction time."
    )
    return ColumnConfig(col.name, col.inferred_type, reason=reason)


def _column_config(col: ColumnProfile, leaks: dict[str, str]) -> ColumnConfig:
    excluded = _exclusion_reason(col, leaks)
    if excluded:
        return ColumnConfig(col.name, col.inferred_type, include=False, reason=excluded)
    if col.inferred_type == FeatureType.CATEGORICAL:
        return _categorical(col)
    if col.inferred_type == FeatureType.NUMERIC:
        return ColumnConfig(col.name, col.inferred_type, reason=_numeric_reason(col))
    if col.inferred_type == FeatureType.BOOLEAN:
        reason = "Boolean -> 0/1 with most-frequent imputation."
        return ColumnConfig(col.name, col.inferred_type, reason=reason)
    reason = "Date -> year, month, weekday, day-of-year and days since the earliest date."
    return ColumnConfig(col.name, col.inferred_type, reason=reason)


def _imbalance(y: pd.Series | None) -> tuple[ImbalanceStrategy, str | None]:
    if y is None:
        return ImbalanceStrategy.NONE, None
    shares = clean_target(y, regression=False).dropna().value_counts(normalize=True)
    if len(shares) < 2:
        return ImbalanceStrategy.NONE, None
    minority = float(shares.min())
    if minority < IMBALANCE_MINORITY:
        return ImbalanceStrategy.CLASS_WEIGHT, (
            f"Smallest class is {minority:.1%} of rows (< {IMBALANCE_MINORITY:.0%}) -> balanced "
            "class weights (SMOTE and undersampling are available in Custom mode)."
        )
    return ImbalanceStrategy.NONE, f"Classes are reasonably balanced (smallest {minority:.1%})."


def _target_transform(y: pd.Series | None) -> tuple[str, str | None]:
    if y is None:
        return "none", None
    values = clean_target(y, regression=True).dropna()
    if len(values) < 3 or values.nunique() < 3:
        return "none", None
    target_skew = float(skew(values))
    if target_skew > SKEW_THRESHOLD and float(values.min()) >= 0:
        return "log1p", (
            f"Target skew is {target_skew:.2f} and values are non-negative -> models learn "
            "log(1 + y); predictions are converted back automatically."
        )
    return "none", None


def _summary(schema: DatasetSchema, columns: Sequence[ColumnConfig]) -> list[str]:
    kept = sum(c.include for c in columns)
    lines = [f"{schema.n_rows:,} rows; {kept} of {len(columns)} candidate features kept."]
    if schema.n_duplicate_rows:
        lines.append(
            f"{schema.n_duplicate_rows} exact duplicate rows will be dropped before splitting."
        )
    for c in columns:
        if not c.include:
            lines.append(f"'{c.name}': {c.reason}")
    lines.append(
        "All learned steps (imputation, encoding, scaling, outlier bounds, skew correction) "
        "are fitted on training folds only."
    )
    return lines


def recommend_config(
    schema: DatasetSchema,
    target: str,
    problem_type: ProblemType,
    leakage: Sequence[ValidationIssue] = (),
    y: pd.Series | None = None,
) -> PreprocessingConfig:
    """Build an Automatic-mode config with a reason for every decision."""
    leaks = {i.column: i.message.split(": ", 1)[-1].rstrip(".") for i in leakage if i.column}
    columns = tuple(_column_config(c, leaks) for c in schema.columns if c.name != target)
    explanations = _summary(schema, columns)
    imbalance, target_transform = ImbalanceStrategy.NONE, "none"
    if problem_type == ProblemType.CLASSIFICATION:
        imbalance, note = _imbalance(y)
    else:
        target_transform, note = _target_transform(y)
    if note:
        explanations.append(note)
    return PreprocessingConfig(
        problem_type=problem_type,
        target=target,
        mode="auto",
        columns=columns,
        rare_category_min_frequency=RARE_MIN_FREQ,
        high_cardinality_threshold=HIGH_CARD_THRESHOLD,
        imbalance=imbalance,
        target_transform=target_transform,
        explanations=tuple(explanations),
    )
