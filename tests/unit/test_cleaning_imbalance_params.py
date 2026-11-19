"""P12 cleaner types / number formats, P13 category-safe resampling, P15 parameters."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ImbalanceStrategy,
    PreprocessingConfig,
    ProblemType,
    ValidationScheme,
)
from tabulaml.models.params import param_specs, search_space, validate_params
from tabulaml.preprocessing.cleaning import DataCleaner, number_format_warning
from tabulaml.training.pipeline_factory import (
    build_model_pipeline,
    effective_imbalance,
    inner_model,
    model_params,
)
from tabulaml.training.samplers import CategoryAwareSMOTE
from tabulaml.training.trainer import train_models
from tabulaml.training.tuning import tune

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION


# ------------------------------------------------------------------ P12 cleaner
def test_leading_zero_codes_stay_text():
    out = DataCleaner().fit_transform(pd.DataFrame({"zip": ["00100", "00200", "10300"] * 10}))
    assert out["zip"].iloc[0] == "00100"


def test_explicit_type_overrides_inference():
    df = pd.DataFrame({"code": [1, 2, 3] * 10})
    out = DataCleaner(column_types={"code": FeatureType.CATEGORICAL}).fit_transform(df)
    assert out["code"].dtype == object


def test_decimal_comma_format():
    df = pd.DataFrame({"eu": ["1,25", "2.500,5", "3,75"] * 10})
    assert DataCleaner(decimal=",").fit_transform(df)["eu"].iloc[:3].tolist() == [
        1.25,
        2500.5,
        3.75,
    ]
    assert number_format_warning(pd.Series(["1,25", "2,50"] * 5)) is not None
    assert number_format_warning(pd.Series(["1,250", "12,500"] * 5)) is None


def test_parse_report_counts_failures_not_missing():
    cleaner = DataCleaner().fit(pd.DataFrame({"n": [1.0, 2.0, 3.0] * 10}))
    report = cleaner.parse_report(pd.DataFrame({"n": ["4", "abc", None, "?"]}))
    assert report["n"]["failed"] == 1 and report["n"]["examples"] == ["abc"]


def test_log1p_blocked_for_negative_targets():
    cfg = PreprocessingConfig(
        REG, "y", "auto", (ColumnConfig("x", FeatureType.NUMERIC),), target_transform="log1p"
    )
    with pytest.raises(ValueError, match="log1p"):
        build_model_pipeline("ridge", cfg, y=np.array([-5.0, 1.0, 2.0]))
    yj = replace(cfg, target_transform="yeo_johnson")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=80)})
    est = build_model_pipeline("ridge", yj, y=df["x"].to_numpy() - 5).fit(df, df["x"] - 5)
    assert np.isfinite(est.predict(df)).all()


# ------------------------------------------------------------------ P13 resampling
def test_smote_keeps_category_codes_valid():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 4, 300), rng.normal(size=300)])
    y = (rng.random(300) < 0.15).astype(int)
    sampler = CategoryAwareSMOTE(k_neighbors=3, random_state=0)
    Xr, yr = sampler.fit_resample(X, y)
    assert sampler.method_ == "SMOTE-NC" and len(yr) > len(y)
    assert np.allclose(Xr[:, 0], np.round(Xr[:, 0])) and set(np.unique(Xr[:, 0])) <= {0, 1, 2, 3}


def _cls_cfg(**kw):
    cols = (ColumnConfig("c", FeatureType.CATEGORICAL), ColumnConfig("x", FeatureType.NUMERIC))
    return PreprocessingConfig(CLS, "y", "auto", cols, **kw)


@pytest.mark.parametrize("strategy", list(ImbalanceStrategy))
def test_every_imbalance_strategy_trains(strategy):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "c": rng.choice(list("abcd"), 300),
            "x": rng.normal(size=300),
            "y": np.where(rng.random(300) < 0.2, "yes", "no"),
        }
    )
    run = train_models(
        df,
        _cls_cfg(imbalance=strategy, cv_folds=3),
        ["logreg", "voting"],
        ensemble_bases={"voting": ["logreg", "dt"]},
    )
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]
    assert run.result_for("logreg").effective_imbalance
    if strategy in (ImbalanceStrategy.SMOTE, ImbalanceStrategy.UNDERSAMPLE):
        assert "ensembles" in run.result_for("voting").effective_imbalance


def test_effective_imbalance_reports_unsupported_class_weights():
    text = effective_imbalance("knn", _cls_cfg(imbalance=ImbalanceStrategy.CLASS_WEIGHT))
    assert "no class weights" in text


# ------------------------------------------------------------------ P15 parameters
def test_every_param_spec_is_valid_for_its_model():
    from tabulaml.models.registry import MODEL_REGISTRY

    for key, spec in MODEL_REGISTRY.items():
        for ptype in spec.tasks:
            if key in ("voting", "stacking", "hybrid_rf_ann"):
                continue
            model = spec.factory(ptype, 0)
            for p in param_specs(key):
                model.set_params(**{p.name: p.default})  # raises if not a real parameter
                assert p.help
            assert search_space(key) or not param_specs(key)


def test_manual_params_reach_the_model_and_are_validated():
    cfg = _cls_cfg(model_params={"rf": {"n_estimators": 77, "max_depth": 6}})
    est = build_model_pipeline("rf", cfg)
    assert inner_model(est).n_estimators == 77
    assert model_params(est, "rf")["max_depth"] == 6
    assert validate_params("rf", {"n_estimators": 5000})["n_estimators"] == 1000  # clipped
    with pytest.raises(ValueError):
        validate_params("rf", {"bogus": 1})


def test_tuning_uses_dev_rows_and_same_cv(monkeypatch):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "g": np.repeat(np.arange(20), 10),
            "c": rng.choice(list("ab"), 200),
            "x": rng.normal(size=200),
        }
    )
    df["y"] = np.where(df["x"] + rng.normal(scale=0.5, size=200) > 0, "p", "n")
    cfg = replace(
        _cls_cfg(cv_folds=3),
        validation=ValidationScheme.GROUPED,
        group_column="g",
        tuning={"rf": 5},
    )
    seen = {}
    import tabulaml.training.tuning as tuning_mod

    real = tuning_mod.RandomizedSearchCV.fit

    def spy(self, X, y=None, groups=None, **kw):
        seen["rows"], seen["groups"] = len(X), groups is not None
        return real(self, X, y, groups=groups, **kw)

    monkeypatch.setattr(tuning_mod.RandomizedSearchCV, "fit", spy)
    run = train_models(df, cfg, ["rf"])
    r = run.result_for("rf")
    assert r.status == "ok" and r.tuning_history is not None and len(r.tuning_history) == 5
    assert seen["groups"] and seen["rows"] == len(run.y_dev)  # never the test rows
    assert set(r.params) >= {"n_estimators", "max_depth"}


def test_tune_skips_ensembles():
    cfg = _cls_cfg()
    est = build_model_pipeline("voting", cfg, base_keys=["logreg", "dt"])
    same, history = tune(est, "voting", pd.DataFrame(), np.array([]), None, cfg, "f1", 5)
    assert same is est and history is None
