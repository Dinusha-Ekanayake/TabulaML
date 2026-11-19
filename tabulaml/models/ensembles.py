"""Voting, stacking and the RF + ANN hybrid.

Each base learner of voting/stacking is a complete pipeline (cleaner -> features matched to
its own profile -> model), so heterogeneous preprocessing lives inside one estimator and
stays leakage-safe: stacking trains its meta-model on internal out-of-fold predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tabulaml.core.types import PreprocessingConfig, ProblemType

DEFAULT_BASES = ("rf", "xgb", "lgbm")
MIN_BASES = 2


def _bases(base_keys: Sequence[str] | None, config: PreprocessingConfig, y: Any) -> list:
    from tabulaml.training.pipeline_factory import build_model_pipeline  # avoid import cycle

    keys = list(base_keys or DEFAULT_BASES)
    if len(keys) < MIN_BASES:
        raise ValueError(f"Choose at least {MIN_BASES} base models for an ensemble.")
    nested = {"voting", "stacking"}
    if nested & set(keys):
        raise ValueError("Ensembles cannot contain other voting/stacking ensembles.")
    return [(k, build_model_pipeline(k, config, y=y, as_base=True)) for k in keys]


def build_voting(
    base_keys: Sequence[str] | None, config: PreprocessingConfig, y: Any = None
) -> Any:
    estimators = _bases(base_keys, config, y)
    if config.problem_type == ProblemType.CLASSIFICATION:
        return VotingClassifier(estimators, voting="soft", n_jobs=1)
    return VotingRegressor(estimators, n_jobs=1)


def build_stacking(
    base_keys: Sequence[str] | None, config: PreprocessingConfig, y: Any = None
) -> Any:
    estimators = _bases(base_keys, config, y)
    if config.problem_type == ProblemType.CLASSIFICATION:
        meta = LogisticRegression(max_iter=1000)
        return StackingClassifier(estimators, final_estimator=meta, cv=config.cv_folds, n_jobs=1)
    return StackingRegressor(estimators, final_estimator=RidgeCV(), cv=config.cv_folds, n_jobs=1)


class HybridRFANNRegressor(RegressorMixin, BaseEstimator):
    """Random Forest plus a neural network that learns what the forest gets wrong.

    1. Out-of-fold RF predictions (``cv`` folds) give honest residuals y - rf(x).
    2. A scaled MLP is trained on x -> residual. Using out-of-fold residuals stops the MLP
       from fitting the forest's in-sample overfit noise (which is ~0 on training rows).
    3. The RF is refit on all rows. Prediction = rf(x) + mlp(x).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        cv: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.cv = cv
        self.random_state = random_state

    def _forest(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=self.random_state
        )

    def fit(self, X: Any, y: Any) -> HybridRFANNRegressor:
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype="float64")
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(self._forest(), X, y, cv=folds)
        self.mlp_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        self.hidden_layer_sizes,
                        early_stopping=True,
                        max_iter=500,
                        random_state=self.random_state,
                    ),
                ),
            ]
        ).fit(X, y - oof)
        self.rf_ = self._forest().fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype="float64")
        return self.rf_.predict(X) + self.mlp_.predict(X)
