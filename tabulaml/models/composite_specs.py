"""Composite models: ensembles, hybrids and forecasting baselines.

They are assembled by ``training.pipeline_factory`` (from base learners or forecast
features), so their registry factory is only a placeholder.
"""

from __future__ import annotations

from tabulaml.models.spec import BOTH, REG, TREE, ModelSpec, _ensemble_placeholder

COMPOSITE_SPECS: list[ModelSpec] = [
    ModelSpec(
        "voting",
        "Voting Ensemble",
        "Ensembles",
        TREE,
        BOTH,
        _ensemble_placeholder,
        False,
        "Averages the predictions of the chosen base models (soft voting).",
    ),
    ModelSpec(
        "stacking",
        "Stacking Ensemble",
        "Ensembles",
        TREE,
        BOTH,
        _ensemble_placeholder,
        False,
        "A meta-model learns how to combine out-of-fold base-model predictions.",
    ),
    ModelSpec(
        "hybrid_rf_ann",
        "Hybrid RF + ANN",
        "Ensembles",
        TREE,
        frozenset({REG}),
        _ensemble_placeholder,
        False,
        "Random Forest, plus a neural network trained on the forest's out-of-fold residuals.",
    ),
    ModelSpec(
        "naive",
        "Naive forecast (last known value)",
        "Forecasting baselines & hybrid",
        TREE,
        frozenset({REG}),
        _ensemble_placeholder,
        False,
        "Predicts the most recent value known at forecast time. Real models must beat it.",
        forecast_only=True,
    ),
    ModelSpec(
        "seasonal_naive",
        "Seasonal naive",
        "Forecasting baselines & hybrid",
        TREE,
        frozenset({REG}),
        _ensemble_placeholder,
        False,
        "Predicts the value one season ago (needs a seasonal lag, e.g. 52 weeks).",
        forecast_only=True,
    ),
    ModelSpec(
        "rf_arima",
        "Hybrid RF + ARIMA",
        "Forecasting baselines & hybrid",
        TREE,
        frozenset({REG}),
        _ensemble_placeholder,
        False,
        "ARIMA models the time pattern; a Random Forest learns what ARIMA missed from the "
        "lag, calendar and other features. Single series only.",
        forecast_only=True,
    ),
]
