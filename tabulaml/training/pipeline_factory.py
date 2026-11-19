"""Assemble the full, leakage-safe pipeline for one model key.

    DataCleaner -> ColumnTransformer (profile-specific) -> [SMOTE | undersampler] -> model

An ``imblearn`` Pipeline is used so samplers only run during ``fit`` (never on validation,
test or prediction rows). Regression with ``target_transform == "log1p"`` is wrapped in a
TransformedTargetRegressor so predictions come back on the original scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import TransformedTargetRegressor

from tabulaml.core.types import ImbalanceStrategy, PreprocessingConfig, ProblemType
from tabulaml.models.ensembles import HybridRFANNRegressor, build_stacking, build_voting
from tabulaml.models.registry import class_weight_params, get_model
from tabulaml.preprocessing.builder import build_feature_transformer
from tabulaml.preprocessing.cleaning import DataCleaner

SMOTE_MAX_NEIGHBOURS = 5
_SAMPLERS = {ImbalanceStrategy.SMOTE, ImbalanceStrategy.UNDERSAMPLE}


def _smote(config: PreprocessingConfig, y: Any) -> SMOTE | None:
    """SMOTE with k small enough for the minority class inside one CV training fold."""
    if y is None:
        return SMOTE(random_state=config.random_state)
    minority = int(np.bincount(np.asarray(y, dtype=int)).min())
    in_fold = minority * (config.cv_folds - 1) // config.cv_folds
    k = min(SMOTE_MAX_NEIGHBOURS, in_fold - 1)
    if k < 1:
        return None
    return SMOTE(k_neighbors=k, random_state=config.random_state)


def _sampler(config: PreprocessingConfig, y: Any, as_base: bool) -> Any:
    if as_base or config.problem_type != ProblemType.CLASSIFICATION:
        return None
    if config.imbalance == ImbalanceStrategy.SMOTE:
        return _smote(config, y)
    if config.imbalance == ImbalanceStrategy.UNDERSAMPLE:
        return RandomUnderSampler(random_state=config.random_state)
    return None


def _wants_class_weight(config: PreprocessingConfig, as_base: bool) -> bool:
    if config.problem_type != ProblemType.CLASSIFICATION:
        return False
    if config.imbalance == ImbalanceStrategy.CLASS_WEIGHT:
        return True
    return as_base and config.imbalance in _SAMPLERS  # ensembles fall back to class weights


def _base_estimator(model_key: str, config: PreprocessingConfig, y: Any, as_base: bool) -> Any:
    if model_key == "hybrid_rf_ann":
        return HybridRFANNRegressor(random_state=config.random_state)
    model = get_model(model_key).factory(config.problem_type, config.random_state)
    if _wants_class_weight(config, as_base):
        labels = np.asarray(y) if y is not None else np.array([0, 1])  # xgb needs real counts
        model.set_params(**class_weight_params(model_key, labels))
    return model


def _wrap_target(estimator: Any, config: PreprocessingConfig, as_base: bool) -> Any:
    if as_base or config.problem_type != ProblemType.REGRESSION:
        return estimator
    if config.target_transform != "log1p":
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def build_model_pipeline(
    model_key: str,
    config: PreprocessingConfig,
    y: Any = None,
    base_keys: Sequence[str] | None = None,
    as_base: bool = False,
) -> Any:
    """Unfitted estimator for ``model_key``; ``y`` is the (encoded) training target, used
    for class weights and SMOTE neighbour counts. ``as_base`` marks ensemble members."""
    spec = get_model(model_key)
    if config.problem_type not in spec.tasks:
        raise ValueError(f"{spec.name} does not support {config.problem_type.value}.")
    if model_key == "voting":
        return _wrap_target(build_voting(base_keys, config, y), config, as_base)
    if model_key == "stacking":
        return _wrap_target(build_stacking(base_keys, config, y), config, as_base)
    steps: list[tuple[str, Any]] = [
        ("cleaner", DataCleaner(normalize_category_case=config.normalize_category_case)),
        ("features", build_feature_transformer(config, spec.profile)),
    ]
    sampler = _sampler(config, y, as_base)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", _base_estimator(model_key, config, y, as_base)))
    return _wrap_target(Pipeline(steps), config, as_base)
