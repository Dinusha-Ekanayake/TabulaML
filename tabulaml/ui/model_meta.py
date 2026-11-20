"""Presentation metadata for model cards (UI only; no ML logic)."""

from __future__ import annotations

GROUP_ICONS = {
    "Baselines / Linear": ":material/show_chart:",
    "Trees & Bagging": ":material/park:",
    "Boosting": ":material/rocket_launch:",
    "Other (SVM / KNN / ANN)": ":material/hub:",
    "Ensembles": ":material/groups:",
    "Forecasting baselines & hybrid": ":material/timeline:",
}

# key -> (speed, best for)
MODEL_META: dict[str, tuple[str, str]] = {
    "logreg": ("Fast", "A transparent baseline; linear effects"),
    "linreg": ("Fast", "A transparent baseline; linear effects"),
    "ridge": ("Fast", "Many correlated features"),
    "lasso": ("Fast", "Selecting a few strong features"),
    "elasticnet": ("Fast", "Correlated features with selection"),
    "dt": ("Fast", "Readable rules; quick checks"),
    "rf": ("Medium", "Strong default for mixed tabular data"),
    "extratrees": ("Medium", "Noisy data; lower variance than RF"),
    "bagging": ("Medium", "Stabilising a single deep tree"),
    "adaboost": ("Medium", "Clean data with simple boundaries"),
    "gb": ("Medium", "Small-to-medium data, smooth signal"),
    "histgb": ("Fast", "Large data; native missing values"),
    "xgb": ("Medium", "High accuracy with regularisation"),
    "lgbm": ("Fast", "Large data; fastest booster"),
    "catboost": ("Slow", "Many categories; robust defaults"),
    "svm": ("Slow", "Small data with complex boundaries"),
    "knn": ("Fast", "Local patterns; low-dimensional data"),
    "mlp": ("Medium", "Non-linear interactions (ANN)"),
    "voting": ("Slow", "Averaging diverse models"),
    "stacking": ("Slow", "Learning how to combine models"),
    "hybrid_rf_ann": ("Slow", "RF plus an ANN correcting its errors"),
    "naive": ("Fast", "The yardstick every forecast must beat"),
    "seasonal_naive": ("Fast", "Strong yearly / weekly patterns"),
    "rf_arima": ("Medium", "Trend + autocorrelation with non-linear corrections"),
}

SPEED_CLASS = {"Fast": "tml-tag-fast", "Medium": "tml-tag-medium", "Slow": "tml-tag-slow"}


def meta_for(key: str) -> tuple[str, str]:
    return MODEL_META.get(key, ("Medium", ""))
