"""Model catalogue: every model available by key, with its preprocessing profile.

Factories return a fresh, unfitted estimator with sensible defaults. Voting, stacking and
the RF+ANN hybrid are assembled in ``models.ensembles`` from user-chosen base learners.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from tabulaml.core.types import ModelProfile, ProblemType

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
BOTH = frozenset({CLS, REG})
TREE, SCALED = ModelProfile.TREE, ModelProfile.SCALE_SENSITIVE
ENSEMBLE_KEYS = frozenset({"voting", "stacking", "hybrid_rf_ann"})
RECOMMENDED = {
    CLS: ["logreg", "rf", "xgb", "lgbm", "catboost"],
    REG: ["ridge", "rf", "xgb", "lgbm", "catboost"],
}
SVM_MAX_ROWS = 20_000

Factory = Callable[[ProblemType, int], Any]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    group: str
    profile: ModelProfile
    tasks: frozenset[ProblemType]
    factory: Factory
    supports_class_weight: bool
    description: str


def _pick(cls_factory: Callable[[int], Any], reg_factory: Callable[[int], Any]) -> Factory:
    def factory(problem_type: ProblemType, random_state: int) -> Any:
        maker = cls_factory if problem_type == CLS else reg_factory
        return maker(random_state)

    return factory


def _only(task: ProblemType, maker: Callable[[int], Any]) -> Factory:
    def factory(problem_type: ProblemType, random_state: int) -> Any:
        if problem_type != task:
            raise ValueError(f"Model only supports {task.value}.")
        return maker(random_state)

    return factory


def _ensemble_placeholder(problem_type: ProblemType, random_state: int) -> Any:
    raise ValueError("Ensembles are built by tabulaml.models.ensembles from base learners.")


_BOOST = {"n_estimators": 400, "learning_rate": 0.05}

_SPECS: list[ModelSpec] = [
    ModelSpec(
        "logreg",
        "Logistic Regression",
        "Baselines / Linear",
        SCALED,
        frozenset({CLS}),
        _only(CLS, lambda rs: LogisticRegression(max_iter=2000)),
        True,
        "Linear baseline for classification; needs scaled, one-hot inputs.",
    ),
    ModelSpec(
        "linreg",
        "Linear Regression",
        "Baselines / Linear",
        SCALED,
        frozenset({REG}),
        _only(REG, lambda rs: LinearRegression()),
        False,
        "Ordinary least squares baseline.",
    ),
    ModelSpec(
        "ridge",
        "Ridge Regression",
        "Baselines / Linear",
        SCALED,
        frozenset({REG}),
        _only(REG, lambda rs: Ridge(alpha=1.0)),
        False,
        "L2-regularised linear regression.",
    ),
    ModelSpec(
        "lasso",
        "Lasso Regression",
        "Baselines / Linear",
        SCALED,
        frozenset({REG}),
        _only(REG, lambda rs: Lasso(alpha=0.01, max_iter=5000)),
        False,
        "L1-regularised linear regression (feature selection).",
    ),
    ModelSpec(
        "elasticnet",
        "Elastic Net",
        "Baselines / Linear",
        SCALED,
        frozenset({REG}),
        _only(REG, lambda rs: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        False,
        "Mix of L1 and L2 regularisation.",
    ),
    ModelSpec(
        "dt",
        "Decision Tree",
        "Trees & Bagging",
        TREE,
        BOTH,
        _pick(
            lambda rs: DecisionTreeClassifier(min_samples_leaf=2, random_state=rs),
            lambda rs: DecisionTreeRegressor(min_samples_leaf=2, random_state=rs),
        ),
        True,
        "Single tree; interpretable but high variance.",
    ),
    ModelSpec(
        "rf",
        "Random Forest",
        "Trees & Bagging",
        TREE,
        BOTH,
        _pick(
            lambda rs: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs),
            lambda rs: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        ),
        True,
        "Bagged decision trees with random feature subsets.",
    ),
    ModelSpec(
        "extratrees",
        "Extra Trees",
        "Trees & Bagging",
        TREE,
        BOTH,
        _pick(
            lambda rs: ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=rs),
            lambda rs: ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=rs),
        ),
        True,
        "Extremely randomised trees: random split thresholds.",
    ),
    ModelSpec(
        "bagging",
        "Bagging (Decision Trees)",
        "Trees & Bagging",
        TREE,
        BOTH,
        _pick(
            lambda rs: BaggingClassifier(
                DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=rs
            ),
            lambda rs: BaggingRegressor(
                DecisionTreeRegressor(), n_estimators=100, n_jobs=-1, random_state=rs
            ),
        ),
        False,
        "Bootstrap aggregation of full decision trees.",
    ),
    ModelSpec(
        "adaboost",
        "AdaBoost",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: AdaBoostClassifier(n_estimators=200, algorithm="SAMME", random_state=rs),
            lambda rs: AdaBoostRegressor(n_estimators=200, random_state=rs),
        ),
        False,
        "Adaptive boosting that re-weights misclassified rows.",
    ),
    ModelSpec(
        "gb",
        "Gradient Boosting",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: GradientBoostingClassifier(random_state=rs),
            lambda rs: GradientBoostingRegressor(random_state=rs),
        ),
        False,
        "scikit-learn gradient boosting.",
    ),
    ModelSpec(
        "histgb",
        "Histogram Gradient Boosting",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: HistGradientBoostingClassifier(random_state=rs),
            lambda rs: HistGradientBoostingRegressor(random_state=rs),
        ),
        True,
        "Fast histogram-based gradient boosting (LightGBM-style).",
    ),
    ModelSpec(
        "xgb",
        "XGBoost",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: XGBClassifier(
                **_BOOST,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
            lambda rs: XGBRegressor(
                **_BOOST,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=rs,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
        True,
        "Regularised gradient boosting (XGBoost).",
    ),
    ModelSpec(
        "lgbm",
        "LightGBM",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: LGBMClassifier(**_BOOST, num_leaves=31, random_state=rs, verbosity=-1),
            lambda rs: LGBMRegressor(**_BOOST, num_leaves=31, random_state=rs, verbosity=-1),
        ),
        True,
        "Leaf-wise gradient boosting (LightGBM).",
    ),
    ModelSpec(
        "catboost",
        "CatBoost",
        "Boosting",
        TREE,
        BOTH,
        _pick(
            lambda rs: CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                random_seed=rs,
                allow_writing_files=False,
            ),
            lambda rs: CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                random_seed=rs,
                allow_writing_files=False,
            ),
        ),
        True,
        "Ordered gradient boosting with symmetric trees (CatBoost).",
    ),
    ModelSpec(
        "svm",
        "Support Vector Machine",
        "Other (SVM / KNN / ANN)",
        SCALED,
        BOTH,
        _pick(lambda rs: SVC(probability=True, random_state=rs), lambda rs: SVR()),
        True,
        "Kernel SVM; slow on large data.",
    ),
    ModelSpec(
        "knn",
        "K-Nearest Neighbours",
        "Other (SVM / KNN / ANN)",
        SCALED,
        BOTH,
        _pick(lambda rs: KNeighborsClassifier(15), lambda rs: KNeighborsRegressor(15)),
        False,
        "Distance-based; needs scaled inputs.",
    ),
    ModelSpec(
        "mlp",
        "Neural Network (ANN / MLP)",
        "Other (SVM / KNN / ANN)",
        SCALED,
        BOTH,
        _pick(
            lambda rs: MLPClassifier((64, 32), early_stopping=True, max_iter=500, random_state=rs),
            lambda rs: MLPRegressor((64, 32), early_stopping=True, max_iter=500, random_state=rs),
        ),
        False,
        "Feed-forward neural network (multi-layer perceptron).",
    ),
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
]

MODEL_REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def list_models(problem_type: ProblemType) -> list[ModelSpec]:
    return [s for s in _SPECS if problem_type in s.tasks]


def get_model(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'.")
    return MODEL_REGISTRY[key]


def class_weight_params(key: str, y: np.ndarray) -> dict[str, Any]:
    """Estimator params that re-weight classes for model ``key`` (empty if unsupported)."""
    if not get_model(key).supports_class_weight:
        return {}
    if key == "catboost":
        return {"auto_class_weights": "Balanced"}
    if key == "xgb":
        counts = np.bincount(np.asarray(y, dtype=int))
        if len(counts) == 2 and counts[1] > 0:
            return {"scale_pos_weight": float(counts[0] / counts[1])}
        return {}
    return {"class_weight": "balanced"}
