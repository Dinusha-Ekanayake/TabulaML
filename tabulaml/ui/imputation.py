"""Pure helpers for choosing a missing-value (imputation) method per column.

Only the methods the pipeline builder really applies for a column's type are offered,
so a choice is never silently ignored.
"""

from __future__ import annotations

from dataclasses import replace

from tabulaml.core.types import ColumnConfig, FeatureType, PreprocessingConfig

IMPUTE_METHODS: dict[FeatureType, tuple[str, ...]] = {
    FeatureType.NUMERIC: ("auto", "median", "mean", "most_frequent", "constant", "knn"),
    FeatureType.CATEGORICAL: ("auto", "most_frequent", "constant"),
}
_NUMERIC_LABELS = {
    "auto": "Automatic (median)",
    "median": "Median",
    "mean": "Mean (average)",
    "most_frequent": "Most frequent value",
    "constant": "Constant 0",
    "knn": "K-nearest neighbours (KNN)",
}
_CATEGORICAL_LABELS = {
    "auto": "Automatic (own 'missing' category)",
    "most_frequent": "Most frequent category",
    "constant": "Own 'missing' category",
}


def impute_options(feature_type: FeatureType) -> tuple[str, ...]:
    """Methods available for a column type; empty when the type has a fixed rule."""
    return IMPUTE_METHODS.get(feature_type, ())


def impute_label(method: str, feature_type: FeatureType) -> str:
    labels = _CATEGORICAL_LABELS if feature_type == FeatureType.CATEGORICAL else _NUMERIC_LABELS
    return labels.get(method, method)


def imputable_columns(config: PreprocessingConfig) -> list[ColumnConfig]:
    """Included feature columns whose imputation method can be chosen."""
    return [
        c
        for c in config.columns
        if c.include and c.name != config.target and c.feature_type in IMPUTE_METHODS
    ]


def with_imputation(config: PreprocessingConfig, methods: dict[str, str]) -> PreprocessingConfig:
    """Return a NEW config with the given ``{column: method}`` choices applied.

    Raises ValueError for unknown columns or methods the column's type does not support.
    Returns ``config`` itself when nothing changes.
    """
    by_name = {c.name: c for c in config.columns}
    new = config
    for name, method in methods.items():
        if name not in by_name:
            raise ValueError(f"Unknown column '{name}'.")
        col = by_name[name]
        allowed = impute_options(col.feature_type)
        if method not in allowed:
            raise ValueError(
                f"'{method}' is not available for {col.feature_type.value} column '{name}'."
            )
        if col.impute != method:
            new = new.with_column(replace(col, impute=method))
    return replace(new, mode="custom") if new != config else config
