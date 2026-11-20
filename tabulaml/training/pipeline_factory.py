"""Assemble the full, leakage-safe pipeline for one model key.

    DataCleaner (user types + number format) -> ColumnTransformer (profile-specific)
    -> [oversampler | SMOTE(-NC) | undersampler] -> model (default or user parameters)

An ``imblearn`` Pipeline is used so samplers only run during ``fit`` (never on validation,
test or prediction rows). Regression target transforms are wrapped in a
TransformedTargetRegressor so predictions come back on the original scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import PowerTransformer

from tabulaml.core.types import (
    ImbalanceStrategy,
    PreprocessingConfig,
    ProblemType,
    ValidationScheme,
)
from tabulaml.forecasting.features import lag_name
from tabulaml.forecasting.models import HybridRFARIMARegressor, NaiveForecaster
from tabulaml.models.ensembles import HybridRFANNRegressor, build_stacking, build_voting
from tabulaml.models.params import param_specs, validate_params
from tabulaml.models.registry import FORECAST_KEYS, class_weight_params, get_model
from tabulaml.preprocessing.builder import build_feature_transformer
from tabulaml.preprocessing.cleaning import DataCleaner
from tabulaml.training.samplers import CategoryAwareSMOTE

SMOTE_MAX_NEIGHBOURS = 5
_WRAPPERS = (MultiOutputClassifier, MultiOutputRegressor)
_SAMPLERS = {
    ImbalanceStrategy.SMOTE,
    ImbalanceStrategy.UNDERSAMPLE,
    ImbalanceStrategy.RANDOM_OVERSAMPLE,
}
_LABELS = {
    ImbalanceStrategy.NONE: "none",
    ImbalanceStrategy.CLASS_WEIGHT: "class weights",
    ImbalanceStrategy.RANDOM_OVERSAMPLE: "random oversampling",
    ImbalanceStrategy.SMOTE: "SMOTE (SMOTE-NC keeps categories valid)",
    ImbalanceStrategy.UNDERSAMPLE: "random undersampling",
}


def _smote(config: PreprocessingConfig, y: Any) -> CategoryAwareSMOTE | None:
    """SMOTE with k small enough for the minority class inside one CV training fold."""
    if y is None:
        return CategoryAwareSMOTE(random_state=config.random_state)
    minority = int(np.bincount(np.asarray(y, dtype=int)).min())
    in_fold = minority * (config.cv_folds - 1) // config.cv_folds
    k = min(SMOTE_MAX_NEIGHBOURS, in_fold - 1)
    if k < 1:
        return None
    return CategoryAwareSMOTE(k_neighbors=k, random_state=config.random_state)


def _sampler(config: PreprocessingConfig, y: Any, as_base: bool) -> Any:
    if as_base or config.problem_type != ProblemType.CLASSIFICATION or config.is_multi_output:
        return None
    if config.imbalance == ImbalanceStrategy.SMOTE:
        return _smote(config, y)
    if config.imbalance == ImbalanceStrategy.RANDOM_OVERSAMPLE:
        return RandomOverSampler(random_state=config.random_state)
    if config.imbalance == ImbalanceStrategy.UNDERSAMPLE:
        return RandomUnderSampler(random_state=config.random_state)
    return None


def _wants_class_weight(config: PreprocessingConfig, as_base: bool) -> bool:
    if config.problem_type != ProblemType.CLASSIFICATION or config.is_multi_output:
        return False
    if config.imbalance == ImbalanceStrategy.CLASS_WEIGHT:
        return True
    return as_base and config.imbalance in _SAMPLERS  # ensembles fall back to class weights


def effective_imbalance(key: str, config: PreprocessingConfig, as_base: bool = False) -> str:
    """What the model REALLY uses for class imbalance, in plain words."""
    if config.problem_type != ProblemType.CLASSIFICATION:
        return ""
    if config.is_multi_output:
        if config.imbalance == ImbalanceStrategy.NONE:
            return "none"
        return "none (class weights and resampling are not available with several targets)"
    if key in ("voting", "stacking") and config.imbalance in _SAMPLERS:
        return "class weights on base models (samplers are not used inside ensembles)"
    if _wants_class_weight(config, as_base) and not get_model(key).supports_class_weight:
        return "none (this model has no class weights)"
    return _LABELS[config.imbalance]


def _base_estimator(model_key: str, config: PreprocessingConfig, y: Any, as_base: bool) -> Any:
    if model_key == "hybrid_rf_ann":
        model = HybridRFANNRegressor(random_state=config.random_state)
    else:
        model = get_model(model_key).factory(config.problem_type, config.random_state)
    if _wants_class_weight(config, as_base):
        labels = np.asarray(y) if y is not None else np.array([0, 1])  # xgb needs real counts
        model.set_params(**class_weight_params(model_key, labels))
    manual = config.model_params.get(model_key)
    if manual:
        model.set_params(**_solver_safe(model_key, validate_params(model_key, manual)))
    return model


def _solver_safe(key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Logistic regression: L1 / elastic net need the 'saga' solver; l1_ratio is only
    passed for elastic net (scikit-learn warns about it otherwise)."""
    if key != "logreg":
        return params
    out = dict(params)
    penalty = out.get("penalty", "l2")
    if penalty != "elasticnet":
        out.pop("l1_ratio", None)
    if penalty in ("l1", "elasticnet"):
        out |= {"solver": "saga", "max_iter": 5000}
        if penalty == "elasticnet":
            out.setdefault("l1_ratio", 0.5)
    return out


def _check_target_transform(config: PreprocessingConfig, y: Any) -> None:
    if config.target_transform == "log1p" and y is not None and np.nanmin(y) <= -1:
        raise ValueError(
            "log1p needs targets greater than -1; this target has smaller values. "
            "Choose 'yeo_johnson' or 'none' as the target transform."
        )


class BoundedLog1p(TransformerMixin, BaseEstimator):
    """log1p target transform whose inverse stays near the training range.

    A linear model can extrapolate to huge values on the log scale; ``expm1`` of those is
    astronomically large (RMSE ~ 1e20). Predictions are clipped, ON THE LOG SCALE, to the
    training range widened by ``margin`` (1.0 = up to e times beyond the largest target)."""

    def __init__(self, margin: float = 1.0) -> None:
        self.margin = margin

    def fit(self, y: Any, *_: Any) -> BoundedLog1p:
        z = np.log1p(np.asarray(y, dtype=float))
        self.low_, self.high_ = np.nanmin(z) - self.margin, np.nanmax(z) + self.margin
        return self

    def transform(self, y: Any) -> np.ndarray:
        return np.log1p(np.asarray(y, dtype=float))

    def inverse_transform(self, z: Any) -> np.ndarray:
        return np.expm1(np.clip(np.asarray(z, dtype=float), self.low_, self.high_))


def safe_expm1(values: Any) -> np.ndarray:
    """Kept for bundles saved by earlier versions (inverse_func of the old log1p wrapper)."""
    return np.expm1(np.clip(np.asarray(values, dtype=float), None, 50.0))


def _wrap_target(estimator: Any, config: PreprocessingConfig, as_base: bool) -> Any:
    if as_base or config.problem_type != ProblemType.REGRESSION:
        return estimator
    if config.target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=estimator, transformer=BoundedLog1p(), check_inverse=False
        )
    if config.target_transform == "yeo_johnson":
        return TransformedTargetRegressor(regressor=estimator, transformer=PowerTransformer())
    return estimator


def _multi(estimator: Any, config: PreprocessingConfig, as_base: bool) -> Any:
    """One copy of the model per target when there are several targets."""
    if as_base or not config.is_multi_output:
        return estimator
    if config.problem_type == ProblemType.CLASSIFICATION:
        return MultiOutputClassifier(estimator)
    return MultiOutputRegressor(estimator)


INTERNAL_CV_KEYS = frozenset({"stacking", "hybrid_rf_ann"})


def model_unavailable(key: str, config: PreprocessingConfig) -> str | None:
    """Why a model cannot run with these settings (None = it can).

    Stacking and the RF + ANN hybrid build out-of-fold predictions with their own plain
    K-fold, which would mix groups or use the future under grouped / time-ordered
    validation, so they are only offered with random validation."""
    if key in INTERNAL_CV_KEYS and config.validation != ValidationScheme.RANDOM:
        return (
            f"{get_model(key).name} uses its own internal cross-validation, which does not "
            "keep groups together or respect time order. Use it with random validation."
        )
    return _forecast_unavailable(key, config)


forecast_model_unavailable = model_unavailable  # name used by earlier callers


def _forecast_unavailable(key: str, config: PreprocessingConfig) -> str | None:
    fc = config.forecast
    if key not in FORECAST_KEYS:
        return None
    if fc is None:
        return f"{get_model(key).name} is only available in forecasting mode."
    if key == "rf_arima" and fc.series_column:
        return "RF + ARIMA supports a single series; remove the series column in step 2."
    if key == "seasonal_naive" and not (fc.seasonal_lag and fc.seasonal_lag >= fc.horizon):
        return "Seasonal naive needs a seasonal lag (set it in step 2)."
    return None


def _check_available(key: str, config: PreprocessingConfig) -> None:
    reason = model_unavailable(key, config)
    if reason:
        raise ValueError(reason)


def _naive(key: str, config: PreprocessingConfig) -> NaiveForecaster:
    fc = config.forecast
    k = fc.seasonal_lag if key == "seasonal_naive" else fc.horizon
    return NaiveForecaster(column=lag_name(config.target, k))


def _rf_arima(config: PreprocessingConfig) -> HybridRFARIMARegressor:
    """ARIMA on the target + Random Forest (full feature pipeline) on ARIMA's residuals."""
    manual = validate_params("rf_arima", config.model_params.get("rf_arima", {}))
    order = {k: int(manual[k]) for k in ("p", "d", "q") if k in manual}
    forest = get_model("rf").factory(ProblemType.REGRESSION, config.random_state)
    if "n_estimators" in manual:
        forest.set_params(n_estimators=int(manual["n_estimators"]))
    residual = Pipeline(
        [
            ("cleaner", make_cleaner(config)),
            ("features", build_feature_transformer(config, get_model("rf").profile)),
            ("model", forest),
        ]
    )
    fc = config.forecast
    return HybridRFARIMARegressor(
        lag_column=lag_name(config.target, fc.horizon),
        horizon=fc.horizon,
        residual_model=residual,
        **order,
    )


def make_cleaner(config: PreprocessingConfig) -> DataCleaner:
    """Cleaner that honours the user's column types and number format."""
    types = {c.name: c.feature_type for c in config.columns if c.include}
    return DataCleaner(
        normalize_category_case=config.normalize_category_case,
        column_types=types,
        decimal=config.decimal,
        thousands=config.thousands,
    )


def build_model_pipeline(
    model_key: str,
    config: PreprocessingConfig,
    y: Any = None,
    base_keys: Sequence[str] | None = None,
    as_base: bool = False,
) -> Any:
    """Unfitted estimator for ``model_key``; ``y`` is the (encoded) training target, used
    for class weights, SMOTE neighbour counts and target-transform checks."""
    spec = get_model(model_key)
    if config.problem_type not in spec.tasks:
        raise ValueError(f"{spec.name} does not support {config.problem_type.value}.")
    if not as_base:
        _check_target_transform(config, y)
    if not as_base:
        _check_available(model_key, config)
    if model_key in ("naive", "seasonal_naive"):
        return _naive(model_key, config)  # predicts on the original scale: never wrapped
    if model_key == "rf_arima":
        return _rf_arima(config)  # needs the raw target lags: never target-transformed
    if model_key == "voting":
        ensemble = _multi(build_voting(base_keys, config, y), config, as_base)
        return _wrap_target(ensemble, config, as_base)
    if model_key == "stacking":
        ensemble = _multi(build_stacking(base_keys, config, y), config, as_base)
        return _wrap_target(ensemble, config, as_base)
    steps: list[tuple[str, Any]] = [
        ("cleaner", make_cleaner(config)),
        ("features", build_feature_transformer(config, spec.profile)),
    ]
    sampler = _sampler(config, y, as_base)
    if sampler is not None:
        steps.append(("sampler", sampler))
    model = _base_estimator(model_key, config, y, as_base)
    steps.append(("model", _multi(model, config, as_base)))
    return _wrap_target(Pipeline(steps), config, as_base)


def inner_model(estimator: Any) -> Any:
    """The final model inside a (target-wrapped) pipeline."""
    est = getattr(estimator, "regressor", estimator)
    steps = getattr(est, "named_steps", None)
    model = steps["model"] if steps and "model" in steps else est
    return model.estimator if isinstance(model, _WRAPPERS) else model


def param_prefix(estimator: Any) -> str | None:
    """Path to the model's parameters for set_params / search spaces; None for ensembles."""
    est = getattr(estimator, "regressor", None)
    base = "regressor__" if est is not None else ""
    target = est if est is not None else estimator
    steps = getattr(target, "named_steps", None)
    if not steps or "model" not in steps:
        return None
    inner = "estimator__" if isinstance(steps["model"], _WRAPPERS) else ""
    return f"{base}model__{inner}"


def model_params(estimator: Any, key: str) -> dict[str, Any]:
    """Current values of the documented hyperparameters (for the leaderboard and report)."""
    model = inner_model(estimator)
    names = [s.name for s in param_specs(key)]
    current = model.get_params() if hasattr(model, "get_params") else {}
    return {n: current[n] for n in names if n in current}
