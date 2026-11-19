"""Hyperparameter specifications per model: manual widgets, validation and tuning spaces.

Every spec explains itself (``help``), so the UI and the run report can say what a
parameter does. Ensembles are configured through their base models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import loguniform, randint, uniform


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # int | float | choice
    default: Any
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()
    help: str = ""


def _i(name, default, low, high, help_, log=False):  # noqa: ANN001, ANN202 - terse table
    return ParamSpec(name, "int", default, low, high, log, (), help_)


def _f(name, default, low, high, help_, log=False):  # noqa: ANN001, ANN202
    return ParamSpec(name, "float", default, low, high, log, (), help_)


def _c(name, default, choices, help_):  # noqa: ANN001, ANN202
    return ParamSpec(name, "choice", default, None, None, False, tuple(choices), help_)


TREES = _i("n_estimators", 300, 50, 1000, "Number of trees; more is steadier but slower.")
DEPTH = _c("max_depth", None, (None, 4, 6, 8, 12, 20), "Maximum tree depth; None grows fully.")
LEAF = _i("min_samples_leaf", 1, 1, 20, "Minimum rows in a leaf; larger smooths the model.")
FEATS = _c("max_features", "sqrt", ("sqrt", "log2", 0.5, 1.0), "Features tried at each split.")
LR = _f("learning_rate", 0.05, 0.005, 0.3, "Step size of each boosting round.", log=True)
ROUNDS = _i("n_estimators", 400, 50, 1500, "Boosting rounds.")

PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "logreg": (_f("C", 1.0, 1e-3, 100.0, "Inverse regularisation strength.", log=True),),
    "ridge": (_f("alpha", 1.0, 1e-3, 100.0, "L2 penalty strength.", log=True),),
    "lasso": (_f("alpha", 0.01, 1e-4, 10.0, "L1 penalty strength.", log=True),),
    "elasticnet": (
        _f("alpha", 0.01, 1e-4, 10.0, "Penalty strength.", log=True),
        _f("l1_ratio", 0.5, 0.0, 1.0, "Share of L1 in the penalty."),
    ),
    "dt": (DEPTH, LEAF),
    "rf": (TREES, DEPTH, LEAF, FEATS),
    "extratrees": (TREES, DEPTH, LEAF, FEATS),
    "bagging": (
        _i("n_estimators", 100, 10, 500, "Number of bagged trees."),
        _f("max_samples", 1.0, 0.3, 1.0, "Share of rows in each bootstrap sample."),
    ),
    "adaboost": (
        _i("n_estimators", 200, 25, 1000, "Boosting rounds."),
        _f("learning_rate", 1.0, 0.01, 2.0, "Weight of each weak learner.", log=True),
    ),
    "gb": (
        _i("n_estimators", 100, 50, 1000, "Boosting rounds."),
        LR,
        _i("max_depth", 3, 2, 8, "Depth of each tree."),
        _f("subsample", 1.0, 0.5, 1.0, "Row share per round (stochastic boosting)."),
    ),
    "histgb": (
        _i("max_iter", 100, 50, 1000, "Boosting rounds."),
        LR,
        _i("max_leaf_nodes", 31, 8, 128, "Leaves per tree."),
        _f("l2_regularization", 0.0, 0.0, 5.0, "L2 penalty on leaf values."),
    ),
    "xgb": (
        ROUNDS,
        LR,
        _i("max_depth", 6, 2, 12, "Depth of each tree."),
        _f("subsample", 0.8, 0.5, 1.0, "Row share per round."),
        _f("colsample_bytree", 0.8, 0.4, 1.0, "Feature share per tree."),
        _f("min_child_weight", 1.0, 0.5, 20.0, "Minimum evidence in a leaf.", log=True),
        _f("reg_lambda", 1.0, 1e-3, 50.0, "L2 penalty.", log=True),
    ),
    "lgbm": (
        ROUNDS,
        LR,
        _i("num_leaves", 31, 8, 256, "Leaves per tree (main complexity knob)."),
        _i("min_child_samples", 20, 5, 100, "Minimum rows in a leaf."),
        _f("colsample_bytree", 1.0, 0.4, 1.0, "Feature share per tree."),
    ),
    "catboost": (
        _i("iterations", 500, 100, 2000, "Boosting rounds."),
        LR,
        _i("depth", 6, 3, 10, "Depth of the symmetric trees."),
        _f("l2_leaf_reg", 3.0, 0.5, 30.0, "L2 penalty on leaves.", log=True),
    ),
    "svm": (
        _f("C", 1.0, 1e-2, 100.0, "Margin softness: larger fits harder.", log=True),
        _c("gamma", "scale", ("scale", "auto", 0.001, 0.01, 0.1, 1.0), "RBF kernel width."),
        _c("kernel", "rbf", ("rbf", "linear", "poly"), "Kernel shape."),
    ),
    "knn": (
        _i("n_neighbors", 15, 1, 100, "Neighbours that vote."),
        _c("weights", "uniform", ("uniform", "distance"), "Closer neighbours count more?"),
    ),
    "mlp": (
        _c(
            "hidden_layer_sizes",
            (64, 32),
            ((32,), (64,), (64, 32), (128, 64)),
            "Neurons per hidden layer.",
        ),
        _f("alpha", 1e-4, 1e-6, 1e-1, "L2 penalty.", log=True),
        _f("learning_rate_init", 1e-3, 1e-4, 1e-1, "Initial learning rate.", log=True),
    ),
    "hybrid_rf_ann": (TREES,),
}


def param_specs(key: str) -> tuple[ParamSpec, ...]:
    return PARAM_SPECS.get(key, ())


def validate_params(key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Keep known parameters, clip numbers into range, reject unknown choices."""
    specs = {s.name: s for s in param_specs(key)}
    clean: dict[str, Any] = {}
    for name, value in params.items():
        spec = specs.get(name)
        if spec is None:
            raise ValueError(f"Unknown parameter '{name}' for model '{key}'.")
        if spec.kind == "choice":
            if value not in spec.choices:
                raise ValueError(f"'{value}' is not an allowed value for {name}.")
            clean[name] = value
        else:
            caster = int if spec.kind == "int" else float
            clean[name] = caster(min(max(float(value), spec.low), spec.high))
    return clean


def search_space(key: str) -> dict[str, Any]:
    """scipy distributions / lists for RandomizedSearchCV."""
    space: dict[str, Any] = {}
    for s in param_specs(key):
        if s.kind == "choice":
            space[s.name] = list(s.choices)
        elif s.kind == "int":
            space[s.name] = randint(int(s.low), int(s.high) + 1)
        elif s.log:
            space[s.name] = loguniform(s.low, s.high)
        else:
            space[s.name] = uniform(s.low, s.high - s.low)
    return space
