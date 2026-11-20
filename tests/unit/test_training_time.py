"""E1: time estimate, row cap, cooperative cancellation and the background job."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    PreprocessingConfig,
    ProblemType,
    ValidationScheme,
)
from tabulaml.training.estimate import estimate_training
from tabulaml.training.jobs import TrainingJob
from tabulaml.training.trainer import TrainingCancelled, prepare, train_models

REG = ProblemType.REGRESSION
pytestmark = pytest.mark.usefixtures("fast_models")


def _cfg(**kw):
    cols = (
        ColumnConfig("x1", FeatureType.NUMERIC),
        ColumnConfig("x2", FeatureType.NUMERIC),
        ColumnConfig("cat", FeatureType.CATEGORICAL),
    )
    return PreprocessingConfig(REG, "y", "auto", cols, cv_folds=3, **kw)


def test_estimate_grows_with_rows_and_models():
    small = estimate_training(1_000, 10, ["ridge"], 5)
    more_rows = estimate_training(100_000, 10, ["ridge"], 5)
    more_models = estimate_training(1_000, 10, ["ridge", "rf", "catboost"], 5)
    assert small.seconds_low < small.seconds_high
    assert more_rows.seconds_high > small.seconds_high
    assert more_models.seconds_high > small.seconds_high
    assert small.level == "quick" and "s" in small.label


def test_estimate_warns_about_slow_choices():
    est = estimate_training(
        200_000,
        60,
        ["stacking", "svm", "rf"],
        5,
        tuning={"rf": 20},
        ensemble_bases={"stacking": ("rf", "catboost")},
        knn_imputation=True,
    )
    text = " ".join(est.warnings)
    assert est.level == "long"
    for word in (
        "Stacking",
        "Hyperparameter search",
        "SVM will be skipped",
        "KNN imputation",
        "Slowest",
    ):
        assert word in text
    assert est.per_model["stacking"] > est.per_model["rf"] / 21


def test_row_cap_random_does_not_mutate(regression_df):
    before = regression_df.copy()
    prep = prepare(regression_df, _cfg(max_rows=50))
    assert len(prep.X) == 50 and any("Row cap" in n for n in prep.notes)
    pd.testing.assert_frame_equal(regression_df, before)
    again = prepare(regression_df, _cfg(max_rows=50))
    assert np.array_equal(prep.index, again.index)  # seeded, reproducible


def test_row_cap_time_ordered_keeps_latest(regression_df):
    df = regression_df.assign(t=pd.date_range("2020-01-01", periods=len(regression_df)))
    cfg = _cfg(max_rows=40, validation=ValidationScheme.TIME_ORDERED, order_column="t")
    prep = prepare(df, cfg)
    assert sorted(prep.index) == list(range(len(df) - 40, len(df)))


def test_row_cap_grouped_keeps_whole_groups(regression_df):
    df = regression_df.assign(g=np.repeat(np.arange(20), 10))
    cfg = _cfg(max_rows=55, validation=ValidationScheme.GROUPED, group_column="g")
    prep = prepare(df, cfg)
    kept = df.loc[prep.index, "g"]
    assert len(prep.X) <= 55
    assert (kept.value_counts() == 10).all()  # no group was split


def test_cancel_keeps_finished_models(regression_df):
    calls = {"n": 0}

    def stop_after_first_model() -> bool:
        calls["n"] += 1
        return calls["n"] > 6  # first model: 1 start + 3 folds + 1 final check

    run = train_models(regression_df, _cfg(), ["ridge", "rf"], should_stop=stop_after_first_model)
    assert run.cancelled and [r.key for r in run.results] == ["ridge"]
    assert any("Cancelled" in n for n in run.notes)


def test_cancel_before_any_model_raises(regression_df):
    with pytest.raises(TrainingCancelled):
        train_models(regression_df, _cfg(), ["ridge"], should_stop=lambda: True)


def test_job_finishes_and_reports_progress(regression_df):
    job = TrainingJob(df=regression_df, config=_cfg(), model_keys=["ridge"]).start()
    snap = job.wait(120)
    assert snap.state == "done" and snap.fraction == 1.0 and snap.run is not None
    assert job.config == _cfg() and snap.remaining is None


def test_job_cancel_and_failure(regression_df):
    job = TrainingJob(df=regression_df, config=_cfg(), model_keys=["ridge"])
    job.cancel()
    assert job.start().wait(120).state == "cancelled"

    def boom(**_):
        raise RuntimeError("bad data")

    failed = TrainingJob(boom, config=_cfg()).start().wait(10)
    assert failed.state == "failed" and "bad data" in failed.error


def test_row_cap_ignored_when_small(regression_df):
    prep = prepare(regression_df, replace(_cfg(), max_rows=10_000))
    assert len(prep.X) == len(regression_df)
