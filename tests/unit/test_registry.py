"""Every model fits through the full pipeline; ensembles of XGB/LGBM/CatBoost work (risk R1)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_friedman1
from sklearn.metrics import r2_score

from tabulaml.core.types import ImbalanceStrategy, ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.models.ensembles import HybridRFANNRegressor
from tabulaml.models.registry import (
    ENSEMBLE_KEYS,
    MODEL_REGISTRY,
    class_weight_params,
    get_model,
    list_models,
)
from tabulaml.preprocessing.cleaning import clean_frame
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.pipeline_factory import build_model_pipeline

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
FAST = {"n_estimators": 20, "iterations": 20}


def _shrink(est):
    """Make boosted/bagged models tiny so the suite stays fast."""
    params = est.get_params()
    small = {k: v for k, v in ((p, FAST.get(p.split("__")[-1])) for p in params) if v is not None}
    return est.set_params(**small) if small else est


def _data(ptype, messy_df, regression_df):
    if ptype == CLS:
        return messy_df.drop(columns="target"), messy_df["target"].to_numpy(), messy_df, "target"
    return regression_df.drop(columns="y"), regression_df["y"].to_numpy(), regression_df, "y"


def _config(df, target, ptype, **overrides):
    cfg = recommend_config(profile_dataset(clean_frame(df)), target, ptype, y=df[target])
    return cfg.__class__(**{**cfg.__dict__, **overrides})


# Forecasting-only models need forecast features; they are covered in test_forecasting.py.
CASES = [
    (k, t)
    for k, s in MODEL_REGISTRY.items()
    for t in s.tasks
    if k not in ENSEMBLE_KEYS and not s.forecast_only
]


@pytest.mark.parametrize(("key", "ptype"), CASES)
def test_every_model_fits_and_predicts(key, ptype, messy_df, regression_df):
    X, y, df, target = _data(ptype, messy_df, regression_df)
    pipe = _shrink(clone(build_model_pipeline(key, _config(df, target, ptype), y=y)))
    pipe.fit(X, y)
    assert pipe.predict(X.head(7)).shape == (7,)


@pytest.mark.parametrize("key", ["stacking", "voting"])
@pytest.mark.parametrize("ptype", [CLS, REG])
def test_ensembles_of_boosters_work(key, ptype, messy_df, regression_df):
    X, y, df, target = _data(ptype, messy_df, regression_df)
    cfg = _config(df, target, ptype, cv_folds=3)
    est = _shrink(
        clone(build_model_pipeline(key, cfg, y=y, base_keys=["rf", "xgb", "lgbm", "catboost"]))
    )
    est.fit(X, y)
    assert est.predict(X.head(5)).shape == (5,)
    if ptype == CLS:
        assert np.allclose(est.predict_proba(X.head(5)).sum(axis=1), 1)


def test_ensemble_needs_two_bases(messy_df):
    cfg = _config(messy_df, "target", CLS)
    with pytest.raises(ValueError, match="at least 2"):
        build_model_pipeline("voting", cfg, base_keys=["rf"])


def test_log1p_wraps_regression(regression_df):
    df = regression_df.assign(y=np.exp(regression_df["y"] / 3))
    cfg = _config(df, "y", REG, target_transform="log1p")
    est = _shrink(build_model_pipeline("rf", cfg))
    assert type(est).__name__ == "TransformedTargetRegressor"
    est.fit(df.drop(columns="y"), df["y"])
    assert (est.predict(df.drop(columns="y").head(3)) > 0).all()


@pytest.mark.parametrize("strategy", list(ImbalanceStrategy))
def test_imbalance_strategies_fit(strategy, messy_df):
    X, y = messy_df.drop(columns="target"), messy_df["target"].to_numpy()
    cfg = _config(messy_df, "target", CLS, imbalance=strategy)
    for key in ("logreg", "xgb", "catboost"):
        _shrink(build_model_pipeline(key, cfg, y=y)).fit(X, y)


def test_smote_dropped_when_minority_too_small(messy_df):
    y = np.array([0] * 195 + [1] * 5)
    cfg = _config(messy_df, "target", CLS, imbalance=ImbalanceStrategy.SMOTE)
    pipe = build_model_pipeline("logreg", cfg, y=y)
    assert dict(pipe.steps)["sampler"].k_neighbors == 3  # 5 minority rows -> 4 per fold -> k=3
    y_tiny = np.array([0] * 198 + [1] * 2)
    assert "sampler" not in dict(build_model_pipeline("logreg", cfg, y=y_tiny).steps)


def test_class_weight_params():
    assert class_weight_params("xgb", np.array([0, 0, 0, 1]))["scale_pos_weight"] == 3
    assert class_weight_params("catboost", np.array([0, 1])) == {"auto_class_weights": "Balanced"}
    assert class_weight_params("knn", np.array([0, 1])) == {}


def test_registry_lookup_and_task_filter():
    assert get_model("rf").name == "Random Forest"
    assert "logreg" not in {s.key for s in list_models(REG)}
    assert "hybrid_rf_ann" not in {s.key for s in list_models(CLS)}
    with pytest.raises(KeyError):
        get_model("nope")
    with pytest.raises(ValueError):
        build_model_pipeline(
            "logreg", _config(pd.DataFrame({"x": range(40), "y": np.arange(40) * 1.5}), "y", REG)
        )


def test_hybrid_rf_ann_learns_signal():
    X, y = make_friedman1(n_samples=400, random_state=0)
    model = HybridRFANNRegressor(n_estimators=100).fit(X[:300], y[:300])
    assert r2_score(y[300:], model.predict(X[300:])) > 0.6
