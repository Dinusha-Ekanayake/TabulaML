"""Regression tests for the external review: dev-only cleaning, internal-CV models,
forecast horizon gaps and timestamp-safe splits, and cloud history isolation."""

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ForecastConfig,
    PreprocessingConfig,
    ProblemType,
    ValidationScheme,
)
from tabulaml.training.splits import make_cv, make_holdout, time_ranks
from tabulaml.workflow import Roles, auto_configure

REG = ProblemType.REGRESSION
TIME = ValidationScheme.TIME_ORDERED


def test_recommendations_use_a_cleaner_fitted_on_development_rows_only():
    n = 100
    df = pd.DataFrame(
        {
            "t": pd.date_range("2020-01-01", periods=n).astype(str),
            "code": [str(i % 7) for i in range(80)] + [f"text-{i}" for i in range(20)],
            "y": np.arange(n, dtype=float),
        }
    )
    roles = Roles(target="y", problem_type=REG, validation=TIME, order_column="t", cv_folds=3)
    auto = auto_configure(df, roles)
    code_dev = next(c for c in auto.schema_dev.columns if c.name == "code")
    assert code_dev.inferred_type == FeatureType.NUMERIC  # the held-out text never leaks in
    code_all = next(c for c in auto.schema_all.columns if c.name == "code")
    assert code_all.inferred_type != FeatureType.NUMERIC  # the full-data profile still sees it


@pytest.mark.parametrize("scheme", [ValidationScheme.GROUPED, ValidationScheme.TIME_ORDERED])
def test_internal_cv_models_are_unavailable_without_random_validation(scheme):
    from tabulaml.training.pipeline_factory import build_model_pipeline, model_unavailable

    cfg = PreprocessingConfig(REG, "y", "auto", (), validation=scheme)
    for key in ("stacking", "hybrid_rf_ann"):
        assert "internal cross-validation" in model_unavailable(key, cfg)
        with pytest.raises(ValueError, match="internal cross-validation"):
            build_model_pipeline(key, cfg, base_keys=["ridge", "dt"])
    assert model_unavailable("voting", cfg) is None
    assert model_unavailable("stacking", PreprocessingConfig(REG, "y", "auto", ())) is None


def _forecast_cfg(horizon: int, series: str | None = None) -> PreprocessingConfig:
    return PreprocessingConfig(
        REG,
        "y",
        "auto",
        (ColumnConfig("x", FeatureType.NUMERIC),),
        validation=TIME,
        order_column="t",
        cv_folds=3,
        forecast=ForecastConfig(horizon=horizon, series_column=series),
    )


def test_forecast_cv_and_holdout_respect_the_horizon_gap():
    n = 60
    X = pd.DataFrame({"x": np.arange(n), "t": pd.date_range("2021", periods=n, freq="W")})
    cfg = _forecast_cfg(horizon=4)
    split = make_holdout(X[["x"]], np.zeros(n), cfg, X[["t"]])
    assert split.test.min() - split.dev.max() >= 4  # 3 time points left out (h - 1)
    ranks = split.groups[split.dev]
    for train, test in make_cv(cfg, np.zeros(len(split.dev))).split(split.dev, None, ranks):
        assert ranks[test].min() - ranks[train].max() >= 4


def test_shared_timestamps_never_straddle_a_split():
    times = pd.date_range("2022-01-01", periods=30, freq="D")
    X = pd.DataFrame({"t": np.repeat(times, 3), "shop": np.tile(["A", "B", "C"], 30)})
    X["x"] = np.arange(len(X))
    cfg = _forecast_cfg(horizon=1, series="shop")
    split = make_holdout(X[["x"]], np.zeros(len(X)), cfg, X[["t"]])
    assert set(X["t"].iloc[split.dev]).isdisjoint(set(X["t"].iloc[split.test]))
    ranks = split.groups[split.dev]
    for train, test in make_cv(cfg, np.zeros(len(split.dev))).split(split.dev, None, ranks):
        assert set(ranks[train]).isdisjoint(set(ranks[test]))
    assert len(np.unique(time_ranks(X["t"]))) == 30


def test_cloud_history_stays_in_the_session(tmp_path, monkeypatch, regression_df):
    import streamlit as st

    import tabulaml.config as app_config
    from tabulaml.training.trainer import train_models
    from tabulaml.ui import history

    cfg = PreprocessingConfig(REG, "y", "auto", (ColumnConfig("x1", FeatureType.NUMERIC),))
    run = train_models(regression_df, cfg, ["ridge"])
    monkeypatch.setenv("TABULAML_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr(app_config, "CLOUD_MODE", True)
    st.session_state.pop(history.SESSION_KEY, None)
    assert history.save_run(run, "visitor data") is None
    assert not (tmp_path / "experiments").exists()  # nothing written on the shared server
    assert history.list_runs()[0]["dataset"] == "visitor data"
    monkeypatch.setattr(app_config, "CLOUD_MODE", False)
    assert history.save_run(run, "local data") is not None  # local use keeps files
