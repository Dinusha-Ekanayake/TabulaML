"""P11: grouped / time-ordered validation, holdout-first recommendations, incomplete CV."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
    ValidationScheme,
)
from tabulaml.data.samples import load_sample
from tabulaml.training.splits import make_cv, make_holdout, order_key
from tabulaml.training.trainer import prepare
from tabulaml.workflow import Roles, auto_configure, suggest_validation

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "shop_id": np.repeat([f"S{i}" for i in range(30)], n // 30),
            "day": pd.date_range("2024-01-01", periods=n).astype(str),
            "x": rng.normal(size=n),
            "y": rng.choice(["a", "b"], n),
        }
    )


def _cfg(scheme, group=None, order=None, ptype=CLS):
    return PreprocessingConfig(
        ptype,
        "y",
        "auto",
        (ColumnConfig("x", FeatureType.NUMERIC),),
        validation=scheme,
        group_column=group,
        order_column=order,
        cv_folds=4,
    )


def test_grouped_holdout_and_cv_never_share_a_group():
    cfg = _cfg(ValidationScheme.GROUPED, group="shop_id")
    prep = prepare(_frame(), cfg)
    split = make_holdout(prep.X, prep.y, cfg, prep.extra)
    groups = split.groups
    assert not set(groups[split.dev]) & set(groups[split.test])
    y_dev, g_dev = prep.y[split.dev], groups[split.dev]
    for tr, te in make_cv(cfg, y_dev).split(prep.X.iloc[split.dev], y_dev, g_dev):
        assert not set(g_dev[tr]) & set(g_dev[te])


def test_time_ordered_split_trains_on_the_past_only():
    df = _frame().sample(frac=1, random_state=1)  # shuffled file order
    cfg = _cfg(ValidationScheme.TIME_ORDERED, order="day")
    prep = prepare(df, cfg)
    split = make_holdout(prep.X, prep.y, cfg, prep.extra)
    when = order_key(prep.extra["day"])
    assert when.iloc[split.dev].max() < when.iloc[split.test].min()
    dev_when = when.iloc[split.dev].reset_index(drop=True)
    for tr, te in make_cv(cfg, prep.y[split.dev]).split(np.zeros(len(split.dev))):
        assert dev_when.iloc[tr].max() < dev_when.iloc[te].min()


def test_split_does_not_depend_on_feature_selection():
    df = _frame()
    a = _cfg(ValidationScheme.RANDOM)
    b = replace(a, columns=())
    pa, pb = prepare(df, a), prepare(df, b)
    sa = make_holdout(pa.X, pa.y, a, pa.extra)
    sb = make_holdout(pb.X, pb.y, b, pb.extra)
    assert (sa.test == sb.test).all()


def test_recommendations_never_see_test_labels():
    """Corrupting ONLY the test rows' target must not change the automatic config."""
    df, card = load_sample("lk_tea_auction_catalogue")
    roles = Roles(card.target, card.problem_type, ValidationScheme.GROUPED, "sale_id")
    base = auto_configure(df, roles)
    prep = prepare(df, base.config)
    split = make_holdout(prep.X, prep.y, base.config, prep.extra)
    poisoned = df.copy()
    test_rows = prep.index[split.test]
    poisoned.loc[test_rows, "category"] = poisoned.loc[test_rows, card.target]  # leak in test only
    again = auto_configure(poisoned, roles)
    assert [c.include for c in again.config.columns] == [c.include for c in base.config.columns]
    assert again.config.imbalance == base.config.imbalance


def test_auto_configure_excludes_role_columns_and_explains():
    df, card = load_sample("lk_colombo_dengue")
    roles = Roles(
        card.target, card.problem_type, ValidationScheme.TIME_ORDERED, order_column="week_start"
    )
    result = auto_configure(df, roles)
    by = {c.name: c for c in result.config.columns}
    assert not by["week_start"].include and "time-order" in by["week_start"].reason
    assert any("held out before" in e for e in result.config.explanations)
    assert result.n_test == pytest.approx(0.2 * result.n_dev / 0.8, abs=2)


def test_split_issues_follow_fold_settings():
    df = _frame()
    tight = auto_configure(df, Roles("y", CLS, ValidationScheme.GROUPED, "shop_id", cv_folds=40))
    assert "too_few_groups" in {i.code for i in tight.split_issues}
    fine = auto_configure(df, Roles("y", CLS, cv_folds=3))
    assert not [i for i in fine.split_issues if i.level == "error"]


def test_suggest_validation_detects_repeated_ids():
    scheme, col, reason = suggest_validation(_frame(), "y")
    assert scheme == ValidationScheme.GROUPED and col == "shop_id" and reason
    df = pd.DataFrame({"x": range(50), "y": [0, 1] * 25})
    assert suggest_validation(df, "y")[0] == ValidationScheme.RANDOM


def test_incomplete_model_is_never_best():
    cfg = _cfg(ValidationScheme.RANDOM)
    ok = ModelResult("rf", "RF", "ok", {"f1": (0.5, 0.5)}, {"f1": 0.5}, 1, 0.1)
    partial = ModelResult("xgb", "XGB", "incomplete", {"f1": (0.9, np.nan)}, {"f1": 0.9}, 1, 0.1)
    run = TrainingRun(CLS, cfg, "f1", (partial, ok), ("a", "b"), None, None)
    assert run.best().key == "rf"
    assert run.leaderboard()["key"].tolist() == ["rf", "xgb"]
