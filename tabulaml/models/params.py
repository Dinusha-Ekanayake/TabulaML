"""Hyperparameter specifications per model: manual widgets, validation and tuning spaces.

Every spec explains itself (``help``), so the UI and the run report can say what a
parameter does. Ensembles are configured through their base models.
"""

from __future__ import annotations

import math
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
    tunable: bool = True  # False: manual only (e.g. it changes which solver is valid)


def _i(name, default, low, high, help_, log=False):  # noqa: ANN001, ANN202 - terse table
    return ParamSpec(name, "int", default, low, high, log, (), help_)


def _f(name, default, low, high, help_, log=False):  # noqa: ANN001, ANN202
    return ParamSpec(name, "float", default, low, high, log, (), help_)


def _c(name, default, choices, help_, tunable=True):  # noqa: ANN001, ANN202
    return ParamSpec(name, "choice", default, None, None, False, tuple(choices), help_, tunable)


TREES = _i("n_estimators", 300, 50, 1000, "Number of trees; more is steadier but slower.", log=True)
DEPTH = _c("max_depth", None, (None, 4, 6, 8, 12, 20), "Maximum tree depth; None grows fully.")
LEAF = _i("min_samples_leaf", 1, 1, 20, "Minimum rows in a leaf; larger smooths the model.")
FEATS = _c("max_features", "sqrt", ("sqrt", "log2", 0.5, 1.0), "Features tried at each split.")
LR = _f("learning_rate", 0.05, 0.005, 0.3, "Step size of each boosting round.", log=True)
ROUNDS = _i("n_estimators", 400, 50, 1500, "Boosting rounds.", log=True)

PARAM_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "logreg": (
        _f(
            "C", 1.0, 1e-3, 100.0, "Inverse regularisation strength (smaller = stronger).", log=True
        ),
        _c(
            "penalty",
            "l2",
            ("l2", "l1", "elasticnet"),
            "Penalty type: L2 shrinks all weights; L1 can set weights to zero (feature "
            "selection); elastic net mixes both.",
            tunable=False,
        ),
        ParamSpec(
            "l1_ratio",
            "float",
            0.5,
            0.0,
            1.0,
            False,
            (),
            "Elastic net only: 0 = pure L2 (Ridge), 1 = pure L1 (Lasso), between = both.",
            False,
        ),
    ),
    "ridge": (_f("alpha", 1.0, 1e-3, 100.0, "L2 penalty strength.", log=True),),
    "lasso": (_f("alpha", 0.01, 1e-4, 10.0, "L1 penalty strength.", log=True),),
    "elasticnet": (
        _f("alpha", 0.01, 1e-4, 10.0, "Penalty strength.", log=True),
        _f(
            "l1_ratio",
            0.5,
            0.0,
            1.0,
            "Mix of the penalties: 0 = pure Ridge (L2), 1 = pure Lasso (L1), between = both.",
        ),
    ),
    "dt": (
        DEPTH,
        LEAF,
        _f("ccp_alpha", 0.0, 0.0, 0.05, "Cost-complexity pruning: larger cuts weak branches."),
    ),
    "rf": (TREES, DEPTH, LEAF, FEATS),
    "extratrees": (TREES, DEPTH, LEAF, FEATS),
    "bagging": (
        _i("n_estimators", 100, 10, 500, "Number of bagged trees.", log=True),
        _f("max_samples", 1.0, 0.3, 1.0, "Share of rows in each bootstrap sample."),
    ),
    "adaboost": (
        _i("n_estimators", 200, 25, 1000, "Boosting rounds.", log=True),
        _f("learning_rate", 1.0, 0.01, 2.0, "Weight of each weak learner.", log=True),
    ),
    "gb": (
        _i("n_estimators", 100, 50, 1000, "Boosting rounds.", log=True),
        LR,
        _i("max_depth", 3, 2, 8, "Depth of each tree."),
        _f("subsample", 1.0, 0.5, 1.0, "Row share per round (stochastic boosting)."),
        LEAF,
    ),
    "histgb": (
        _i("max_iter", 100, 50, 1000, "Boosting rounds.", log=True),
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
        _f("reg_lambda", 1.0, 1e-3, 50.0, "L2 penalty on leaf weights.", log=True),
        _f("reg_alpha", 0.0, 0.0, 10.0, "L1 penalty on leaf weights."),
        _f("gamma", 0.0, 0.0, 5.0, "Minimum loss reduction needed to split (pruning)."),
    ),
    "lgbm": (
        ROUNDS,
        LR,
        _i("num_leaves", 31, 8, 256, "Leaves per tree (main complexity knob)."),
        _i("min_child_samples", 20, 5, 100, "Minimum rows in a leaf."),
        _f("colsample_bytree", 1.0, 0.4, 1.0, "Feature share per tree."),
        _f("reg_lambda", 0.0, 0.0, 10.0, "L2 penalty on leaf weights."),
        _f("reg_alpha", 0.0, 0.0, 10.0, "L1 penalty on leaf weights."),
    ),
    "catboost": (
        _i("iterations", 500, 100, 2000, "Boosting rounds.", log=True),
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
    "rf_arima": (
        _i("p", 1, 0, 3, "ARIMA: how many past values the series depends on (AR order)."),
        _i("d", 1, 0, 2, "ARIMA: differencing; 1 removes a trend, 0 keeps the level."),
        _i("q", 1, 0, 3, "ARIMA: how many past forecast errors are used (MA order)."),
        TREES,
    ),
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
        if not s.tunable:
            continue
        if s.kind == "choice":
            space[s.name] = list(s.choices)
        elif s.kind == "int":
            space[s.name] = randint(int(s.low), int(s.high) + 1)
        elif s.log:
            space[s.name] = loguniform(s.low, s.high)
        else:
            space[s.name] = uniform(s.low, s.high - s.low)
    return space


LOG_GRID_FACTOR = 3.0  # log-scale grids step by x3 around the default (0.1, 0.3, 1 ...)
COUNT_GRID_FACTOR = 2.0  # counts (trees, rounds) step by x2: 150, 300, 600
LINEAR_GRID_STEPS = 4  # linear grids step by a quarter of the documented range


def grid_values(spec: ParamSpec, n: int = 3, center: Any = None) -> list[Any]:
    """``n`` values centred on ``center`` (the model's default) for a grid search.

    Log-scale settings step by x3 (e.g. 0.01, 0.03, 0.1); others by a quarter of their
    range. Values stay inside the documented range. Choice settings use every choice."""
    if spec.kind == "choice":
        return list(spec.choices)
    n = max(2, int(n))
    mid = float(spec.default if center is None else center)
    offsets = [k - (n - 1) // 2 for k in range(n + LINEAR_GRID_STEPS)]
    if spec.log and mid > 0:
        factor = COUNT_GRID_FACTOR if spec.kind == "int" else LOG_GRID_FACTOR
        raw = [mid * factor**k for k in offsets]
    else:
        step = (spec.high - spec.low) / LINEAR_GRID_STEPS
        raw = [mid + k * step for k in offsets]
    values: list[Any] = []
    for v in raw:
        v = min(max(v, spec.low), spec.high)
        v = int(round(v)) if spec.kind == "int" else float(f"{v:.3g}")
        if v not in values:
            values.append(v)
        if len(values) == n:
            break
    return sorted(values)


def grid_size(grid: dict[str, list[Any]]) -> int:
    """Number of combinations in a grid (product of the value counts)."""
    return math.prod(len(v) for v in grid.values()) if grid else 0


def model_defaults(key: str, problem_type: Any) -> dict[str, Any]:
    """The values a model REALLY starts with (used to prefill the manual form).

    Library defaults such as ``None`` fall back to the documented spec default."""
    from tabulaml.models.registry import get_model  # local: registry imports this module

    try:
        current = get_model(key).factory(problem_type, 0).get_params()
    except (ValueError, KeyError):
        current = {}
    out = {}
    for spec in param_specs(key):
        value = current.get(spec.name)
        if value is None or (spec.kind == "choice" and value not in spec.choices):
            value = spec.default
        out[spec.name] = value
    return out
