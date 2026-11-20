"""E2: several targets at once (multi-label / multi-output classification and regression)."""

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
)
from tabulaml.evaluation.importance import permutation_importance_df
from tabulaml.evaluation.multi_metrics import (
    hamming_loss,
    multi_output_metrics,
    per_target_metrics,
    subset_accuracy,
)
from tabulaml.inference.persistence import bundle_bytes, load_bundle
from tabulaml.inference.predictor import Predictor, make_input_spec
from tabulaml.training.trainer import prepare, train_models
from tabulaml.workflow import Roles, auto_configure

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
pytestmark = pytest.mark.usefixtures("fast_models")


@pytest.fixture
def multi_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 240
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cat": cat,
            "flag": np.where(x1 + rng.normal(scale=0.3, size=n) > 0, "yes", "no"),
            "level": pd.cut(x2, [-np.inf, -0.5, 0.5, np.inf], labels=["low", "mid", "high"]),
            "amount": 2 * x1 + (cat == "a") * 3 + rng.normal(scale=0.2, size=n) + 10,
            "cost": -x2 + rng.normal(scale=0.2, size=n) + 5,
        }
    )
    df.loc[[0, 1], "level"] = np.nan  # rows missing ANY target are dropped
    return df


def _cfg(problem, target, extra, **kw):
    cols = (
        ColumnConfig("x1", FeatureType.NUMERIC),
        ColumnConfig("x2", FeatureType.NUMERIC),
        ColumnConfig("cat", FeatureType.CATEGORICAL, encoding="target"),
    )
    return PreprocessingConfig(problem, target, "auto", cols, cv_folds=3, extra_targets=extra, **kw)


def test_metric_definitions():
    t = np.array([[1, 0], [1, 1], [0, 0], [2, 1]])
    p = np.array([[1, 0], [1, 0], [0, 0], [1, 1]])
    assert subset_accuracy(t, p) == 0.5
    assert hamming_loss(t, p) == 0.25
    m = multi_output_metrics(CLS, t, p)
    assert {"f1_macro", "accuracy", "subset_accuracy", "hamming_loss"} <= set(m)
    per = per_target_metrics(REG, t.astype(float), p.astype(float), ["a", "b"])
    assert set(per) == {"a", "b"} and "rmse" in per["a"]


def test_prepare_two_targets(multi_df):
    cfg = _cfg(CLS, "flag", ("level",))
    assert cfg.targets == ("flag", "level")
    prep = prepare(multi_df, cfg)
    assert prep.y.shape == (len(multi_df) - 2, 2)
    assert set(prep.target_labels) == {"flag", "level"}
    assert "level" not in prep.X.columns and "flag" not in prep.X.columns
    assert any("missing a target" in n for n in prep.notes)


def test_multi_classification_trains_and_predicts(multi_df):
    cfg = _cfg(CLS, "flag", ("level",), imbalance=ImbalanceStrategy.SMOTE, tuning={"rf": 3})
    run = train_models(
        multi_df, cfg, ["logreg", "rf", "voting"], ensemble_bases={"voting": ["logreg", "dt"]}
    )
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]
    assert run.primary_metric == "f1_macro" and run.is_multi_output
    rf = run.result_for("rf")
    assert set(rf.per_target) == {"flag", "level"}
    assert rf.y_pred.shape == (len(run.y_test), 2) and rf.oof_pred.shape[1] == 2
    assert "several targets" in rf.effective_imbalance
    assert rf.tuning_history is not None and "n_estimators" in rf.params

    spec = make_input_spec(multi_df[list(run.feature_columns)], cfg)
    predictor = Predictor(rf.pipeline, cfg, None, spec, target_labels=run.target_labels)
    out = predictor.predict_batch(multi_df.head(5))
    assert set(out["prediction_flag"]) <= {"yes", "no"}
    assert set(out["prediction_level"]) <= {"low", "mid", "high"}
    one = predictor.predict_one({"x1": 1.0, "x2": 0.0, "cat": "zzz"})
    assert set(one["prediction"]) == {"flag", "level"}

    loaded = load_bundle(bundle_bytes(rf, run, spec, fmt="pickle"))
    assert loaded.predict_batch(multi_df.head(3))["prediction_flag"].notna().all()

    imp = permutation_importance_df(
        rf.pipeline, run.X_test, run.y_test, CLS, "f1_macro", n_repeats=2
    )
    assert set(imp["feature"]) == {"x1", "x2", "cat"}


def test_multi_regression_with_target_transform(multi_df):
    cfg = _cfg(REG, "amount", ("cost",), target_transform="log1p")
    run = train_models(
        multi_df, cfg, ["ridge", "rf", "stacking"], ensemble_bases={"stacking": ["ridge", "dt"]}
    )
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]
    assert run.primary_metric == "rmse"
    best = run.best()
    assert best.test_metrics["r2"] > 0.5
    assert set(best.per_target) == {"amount", "cost"}


def test_auto_configure_excludes_other_targets(multi_df):
    roles = Roles(target="flag", problem_type=CLS, extra_targets=("level",), cv_folds=3)
    auto = auto_configure(multi_df, roles)
    cfg = auto.config
    level = next(c for c in cfg.columns if c.name == "level")
    assert not level.include and "target" in level.reason.lower()
    assert cfg.extra_targets == ("level",)
    assert cfg.imbalance == ImbalanceStrategy.NONE
    assert not [i for i in auto.split_issues if i.level == "error"]


def test_single_target_unchanged(multi_df):
    cfg = replace(_cfg(CLS, "flag", ()), extra_targets=())
    assert cfg.targets == ("flag",)
    assert prepare(multi_df, cfg).y.ndim == 1
