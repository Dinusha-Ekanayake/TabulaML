"""Regularisation options: every documented penalty reaches the model and trains."""

import pytest
from sklearn.base import clone

from tabulaml.core.types import ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.models.params import param_specs, search_space
from tabulaml.preprocessing.cleaning import clean_frame
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.pipeline_factory import build_model_pipeline, inner_model

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
pytestmark = pytest.mark.usefixtures("fast_models")


def _setup(ptype, messy_df, regression_df, params):
    df, target = (messy_df, "target") if ptype == CLS else (regression_df, "y")
    cfg = recommend_config(profile_dataset(clean_frame(df)), target, ptype, y=df[target])
    cfg = cfg.__class__(**{**cfg.__dict__, "model_params": params})
    return df.drop(columns=target), df[target].to_numpy(), cfg


@pytest.mark.parametrize(
    ("key", "ptype", "params"),
    [
        ("logreg", CLS, {"penalty": "l1", "C": 0.5}),
        ("logreg", CLS, {"penalty": "elasticnet", "l1_ratio": 0.3}),
        ("logreg", CLS, {"penalty": "l2", "l1_ratio": 0.3}),  # l1_ratio ignored for l2
        ("dt", REG, {"ccp_alpha": 0.01}),
        ("gb", CLS, {"min_samples_leaf": 5}),
        ("xgb", REG, {"reg_alpha": 1.0, "gamma": 0.5, "reg_lambda": 2.0}),
        ("lgbm", CLS, {"reg_alpha": 0.5, "reg_lambda": 1.0}),
    ],
)
def test_penalties_reach_the_model_and_train(key, ptype, params, messy_df, regression_df):
    X, y, cfg = _setup(ptype, messy_df, regression_df, {key: params})
    pipe = clone(build_model_pipeline(key, cfg, y=y))
    model = inner_model(pipe)
    for name, value in params.items():
        if key == "logreg" and name == "l1_ratio" and params.get("penalty") != "elasticnet":
            assert model.get_params()["l1_ratio"] is None
            continue
        assert model.get_params()[name] == pytest.approx(value) or model.get_params()[name] == value
    if key == "logreg" and params["penalty"] != "l2":
        assert model.solver == "saga"
    pipe.fit(X, y)
    assert len(pipe.predict(X.head(5))) == 5


def test_manual_only_params_are_not_tuned():
    assert "penalty" not in search_space("logreg") and "C" in search_space("logreg")
    assert {s.name for s in param_specs("xgb")} >= {"reg_alpha", "reg_lambda", "gamma"}


def test_grid_values_centre_on_defaults_and_stay_in_range():
    from tabulaml.models.params import grid_size, grid_values, model_defaults

    defaults = model_defaults("xgb", REG)
    specs = {s.name: s for s in param_specs("xgb")}
    assert grid_values(specs["n_estimators"], 3, defaults["n_estimators"]) == [200, 400, 800]
    lr = grid_values(specs["learning_rate"], 3, defaults["learning_rate"])
    assert lr[1] == pytest.approx(0.05) and lr[0] < lr[1] < lr[2]
    alpha = grid_values(specs["reg_alpha"], 4, 0.0)
    assert alpha[0] == 0.0 and len(alpha) == 4 and max(alpha) <= specs["reg_alpha"].high
    assert grid_size({"a": [1, 2], "b": [1, 2, 3]}) == 6
    assert model_defaults("dt", CLS)["min_samples_leaf"] == 2  # the model's real default
    assert model_defaults("rf", REG)["max_features"] == 1.0


def test_grid_search_runs_on_dev_rows_only(messy_df, regression_df):
    from dataclasses import replace

    from tabulaml.training.trainer import train_models

    _, _, cfg = _setup(REG, messy_df, regression_df, {})
    grid = {"rf": {"n_estimators": (60, 120), "min_samples_leaf": (1, 5)}}
    run = train_models(regression_df, replace(cfg, cv_folds=3, grids=grid), ["rf"])
    r = run.result_for("rf")
    assert r.status == "ok" and len(r.tuning_history) == 4
    assert r.params["min_samples_leaf"] in (1, 5)


def test_oversized_grid_fails_the_model_not_the_run(messy_df, regression_df):
    from dataclasses import replace

    from tabulaml.training.trainer import train_models

    _, _, cfg = _setup(REG, messy_df, regression_df, {})
    huge = {"xgb": {"n_estimators": tuple(range(50, 1050, 50)), "max_depth": tuple(range(2, 13))}}
    run = train_models(regression_df, replace(cfg, cv_folds=3, grids=huge), ["xgb", "ridge"])
    assert run.result_for("xgb").status == "failed" and "limit" in run.result_for("xgb").error
    assert run.result_for("ridge").status == "ok"


def test_models_offered_match_the_problem_type():
    from tabulaml.models.registry import list_models

    classification = {s.key for s in list_models(CLS)}
    regression = {s.key for s in list_models(REG)}
    forecasting = {s.key for s in list_models(REG, forecasting=True)}
    assert "logreg" in classification and "logreg" not in regression
    assert {"linreg", "ridge", "lasso", "elasticnet", "hybrid_rf_ann"}.isdisjoint(classification)
    assert {"naive", "seasonal_naive", "rf_arima"}.isdisjoint(classification | regression)
    assert regression < forecasting and {"naive", "rf_arima"} <= forecasting


def test_forecast_models_explain_when_unavailable():
    from tabulaml.core.types import ForecastConfig, PreprocessingConfig
    from tabulaml.training.pipeline_factory import forecast_model_unavailable

    many = PreprocessingConfig(
        REG, "y", "auto", (), forecast=ForecastConfig(series_column="shop", seasonal_lag=None)
    )
    assert "single series" in forecast_model_unavailable("rf_arima", many)
    assert "seasonal lag" in forecast_model_unavailable("seasonal_naive", many)
    assert forecast_model_unavailable("naive", many) is None
    assert forecast_model_unavailable("rf", many) is None
    plain = PreprocessingConfig(REG, "y", "auto", ())
    assert "forecasting mode" in forecast_model_unavailable("naive", plain)


def test_l1_and_l2_together_in_one_model(messy_df, regression_df):
    """Ridge + Lasso at once: Elastic Net (regression), elastic-net logistic regression and
    XGBoost with both reg_alpha (L1) and reg_lambda (L2), all set manually."""
    X, y, cfg = _setup(
        REG, messy_df, regression_df, {"elasticnet": {"alpha": 0.1, "l1_ratio": 0.3}}
    )
    model = inner_model(clone(build_model_pipeline("elasticnet", cfg, y=y)))
    assert (model.alpha, model.l1_ratio) == (0.1, 0.3)
    X, y, cfg = _setup(
        CLS,
        messy_df,
        regression_df,
        {"logreg": {"penalty": "elasticnet", "l1_ratio": 0.7, "C": 2.0}},
    )
    pipe = clone(build_model_pipeline("logreg", cfg, y=y)).fit(X, y)
    model = inner_model(pipe)
    assert (model.penalty, model.l1_ratio, model.C, model.solver) == (
        "elasticnet",
        0.7,
        2.0,
        "saga",
    )
