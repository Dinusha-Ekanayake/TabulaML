"""E5: Random Forest dashboard helpers."""

from dataclasses import replace

import pytest

from tabulaml.core.types import ColumnConfig, FeatureType, PreprocessingConfig, ProblemType
from tabulaml.evaluation.rf_insights import (
    forest_summary,
    head_to_head,
    impurity_importance,
    verdict,
)
from tabulaml.training.trainer import train_models

pytestmark = pytest.mark.usefixtures("fast_models")


def _cfg(**kw):
    cols = (
        ColumnConfig("x1", FeatureType.NUMERIC),
        ColumnConfig("x2", FeatureType.NUMERIC),
        ColumnConfig("cat", FeatureType.CATEGORICAL),
    )
    return PreprocessingConfig(ProblemType.REGRESSION, "y", "auto", cols, cv_folds=3, **kw)


def test_head_to_head_and_forest(regression_df):
    run = train_models(regression_df, _cfg(target_transform="yeo_johnson"), ["rf", "xgb", "ridge"])
    table = head_to_head(run)
    assert list(table["rival"]) == ["XGBoost"]
    row = table.iloc[0]
    assert row["rf_wins"] + row["rf_losses"] + row["ties"] == 3
    assert verdict(table)
    summary = forest_summary(run.result_for("rf").pipeline)  # target-wrapped pipeline
    assert summary["trees"] > 0 and summary["mean_depth"] > 0
    imp = impurity_importance(run.result_for("rf").pipeline)
    assert 0 < len(imp) <= 15 and (imp["importance"] >= 0).all()
    assert any("x1" in f for f in imp["feature"])


def test_no_rf_or_no_rival(regression_df):
    run = train_models(regression_df, _cfg(), ["ridge"])
    assert head_to_head(run).empty and verdict(head_to_head(run)) is None
    assert forest_summary(run.result_for("ridge").pipeline) is None
    assert impurity_importance(run.result_for("ridge").pipeline).empty
    run = train_models(regression_df, replace(_cfg(), cv_folds=2), ["rf"])
    assert head_to_head(run).empty
