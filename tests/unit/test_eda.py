"""Explore-data helpers: summaries and charts for every column type."""

import numpy as np
import pandas as pd

from tabulaml.core.types import FeatureType, ProblemType
from tabulaml.evaluation.eda_plots import (
    column_summary,
    correlation_fig,
    distribution_fig,
    strongest_pairs,
    target_relation_fig,
)

NUM, CAT, DT = FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.DATETIME


def _frame():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "x_twin": x * 2 + rng.normal(scale=0.01, size=n),
            "noise": rng.normal(size=n),
            "city": rng.choice(["Colombo", "Kandy", "Galle"], n),
            "when": pd.date_range("2022-01-01", periods=n, freq="D"),
            "label": np.where(x > 0, "yes", "no"),
            "amount": 3 * x + rng.normal(size=n),
        }
    )


def test_summaries_per_type():
    df = _frame().assign(x=lambda d: d["x"].mask(d.index < 30))
    num = column_summary(df["x"], NUM)
    assert num["missing"] == "10.0%" and {"mean", "median", "skew"} <= set(num)
    cat = column_summary(df["city"], CAT)
    assert cat["distinct"] == 3 and "most common" in cat


def test_distribution_charts_for_every_type():
    df = _frame()
    assert len(distribution_fig(df["x"], NUM, "x").data) == 2  # histogram + box
    assert distribution_fig(df["city"], CAT, "city").data[0].type == "bar"
    assert distribution_fig(df["when"], DT, "when").data[0].type == "scatter"


def test_target_relations_for_both_problem_types():
    df = _frame()
    cls = ProblemType.CLASSIFICATION
    assert len(target_relation_fig(df["x"], df["label"], NUM, cls, "x", "label").data) == 2
    stacked = target_relation_fig(df["city"], df["label"], CAT, cls, "city", "label")
    assert stacked.layout.barmode == "stack"
    reg = ProblemType.REGRESSION
    assert (
        target_relation_fig(df["x"], df["amount"], NUM, reg, "x", "amount").data[0].mode
        == "markers"
    )
    assert (
        target_relation_fig(df["city"], df["amount"], CAT, reg, "city", "amount").data[0].type
        == "box"
    )


def test_correlations_and_strongest_pairs():
    df = _frame()[["x", "x_twin", "noise"]]
    assert correlation_fig(df) is not None
    a, b, r = strongest_pairs(df)[0]
    assert {a, b} == {"x", "x_twin"} and r > 0.99
    assert correlation_fig(df[["x"]]) is None and strongest_pairs(df[["x"]]) == []


def test_feature_relevance_ranks_the_informative_feature_first():
    from tabulaml.evaluation.feature_relevance import feature_relevance, relevance_fig

    df = _frame()
    types = {"x": NUM, "noise": NUM, "city": CAT, "when": DT}
    reg = feature_relevance(
        df, ["x", "noise", "city", "when"], types, "amount", ProblemType.REGRESSION
    )
    assert reg.iloc[0]["feature"] == "x" and reg.iloc[0]["correlation"] > 0.8
    assert np.isnan(reg.set_index("feature").loc["city", "correlation"])  # numbers only
    cls = feature_relevance(df, ["x", "noise"], types, "label", ProblemType.CLASSIFICATION)
    assert cls.iloc[0]["feature"] == "x" and cls.iloc[0]["correlation"] > 0.5
    assert relevance_fig(reg, "amount").data[0].type == "bar"
    empty = feature_relevance(df.head(0), ["x"], types, "amount", ProblemType.REGRESSION)
    assert empty.empty
