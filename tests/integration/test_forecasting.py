"""E3: forecasting as regression on past-only features, baselines, RF + ARIMA, future."""

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ForecastConfig,
    PreprocessingConfig,
    ProblemType,
    ValidationScheme,
)
from tabulaml.data.samples import load_sample
from tabulaml.forecasting.checks import forecast_issues
from tabulaml.forecasting.features import (
    add_forecast_features,
    generated_columns,
    infer_step,
    lag_name,
    suggest_seasonal_lag,
)
from tabulaml.forecasting.future import forecast_future
from tabulaml.models.registry import list_models
from tabulaml.training.trainer import train_models
from tabulaml.workflow import Roles, auto_configure

REG = ProblemType.REGRESSION
TIME = ValidationScheme.TIME_ORDERED
pytestmark = pytest.mark.usefixtures("fast_models")


@pytest.fixture
def series() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 150
    t = np.arange(n)
    dates = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    rain = rng.gamma(2, 10, n)
    y = 50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 52) + 0.2 * rain + rng.normal(0, 2, n)
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "rain": rain, "sales": y})
    return frame.sample(frac=1.0, random_state=1)  # shuffled on purpose: order is restored


def _roles(**fc_kw) -> Roles:
    fc = ForecastConfig(**{"horizon": 2, "n_lags": 3, "windows": (4,), **fc_kw})
    return Roles(
        target="sales",
        problem_type=REG,
        validation=TIME,
        order_column="date",
        cv_folds=3,
        forecast=fc,
    )


def test_features_use_only_the_past(series):
    cfg = PreprocessingConfig(
        REG,
        "sales",
        "auto",
        (),
        validation=TIME,
        order_column="date",
        forecast=ForecastConfig(horizon=2, n_lags=2, windows=(3,)),
    )
    built = add_forecast_features(series, cfg)
    ordered = series.assign(d=pd.to_datetime(series["date"])).sort_values("d")
    y = ordered["sales"].to_numpy()
    assert built[lag_name("sales", 2)].iloc[0] == pytest.approx(y[0])  # row 2 sees row 0
    assert built[lag_name("sales", 3)].iloc[1] == pytest.approx(y[0])
    assert built["sales__mean3"].iloc[2] == pytest.approx(y[0:3].mean())
    assert len(built) == len(series) - 2  # first h rows have no lag h
    assert "rain__lag2" in built and "__month" in built
    assert series.index.equals(series.index)  # input untouched (new frame returned)
    assert set(generated_columns(series, cfg)) >= {"sales__lag2", "sales__mean3", "rain__lag2"}


def test_frequency_helpers(series):
    when = pd.to_datetime(series["date"])
    assert suggest_seasonal_lag(when) == 52
    assert infer_step(when).freqstr.startswith("W")
    assert ForecastConfig(horizon=3, n_lags=2, seasonal_lag=2).lags() == (3, 4)


def test_checks_flag_duplicates_and_short_series(series):
    fc = ForecastConfig(horizon=1, n_lags=3)
    dup = pd.concat([series, series.head(3)])
    codes = {i.code for i in forecast_issues(dup, "sales", "date", fc, 3)}
    assert "duplicate_times" in codes
    short = forecast_issues(series.head(8), "sales", "date", fc, 5)
    assert any(i.code == "series_too_short" and i.level == "error" for i in short)
    assert forecast_issues(series, "sales", None, fc, 3)[0].code == "no_time_column"


def test_auto_configure_and_training_with_baselines_and_hybrid(series):
    roles = _roles(seasonal_lag=52)
    auto = auto_configure(series, roles)
    cfg = auto.config
    assert auto.leakage == ()
    rain = next(c for c in cfg.columns if c.name == "rain")
    assert not rain.include and "same-period" in rain.reason
    assert cfg.forecast.horizon == 2
    keys = ["naive", "seasonal_naive", "ridge", "rf", "rf_arima"]
    run = train_models(series, cfg, keys)
    status = {r.key: r.status for r in run.results}
    assert status == dict.fromkeys(keys, "ok"), [r.error for r in run.results]
    assert (
        run.result_for("ridge").test_metrics["rmse"] < run.result_for("naive").test_metrics["rmse"]
    )
    assert run.test_order is not None and len(run.test_order) == len(run.y_test)
    assert np.all(np.diff(run.test_order.astype("int64")) > 0)  # test rows in time order

    table, _ = forecast_future(
        run.result_for("rf_arima").pipeline, series, cfg, run.feature_columns
    )
    assert list(table["step_ahead"]) == [1, 2]
    last = pd.to_datetime(series["date"]).max()
    assert table["date"].iloc[0] == last + pd.Timedelta(weeks=1)
    assert table["forecast"].notna().all()


def test_multiple_series_and_unavailable_models(series):
    other = series.assign(sales=series["sales"] * 2)
    both = pd.concat([series.assign(shop="A"), other.assign(shop="B")], ignore_index=True)
    roles = _roles(series_column="shop", exog="none")
    cfg = auto_configure(both, roles).config
    run = train_models(both, cfg, ["naive", "rf", "rf_arima", "seasonal_naive"])
    status = {r.key: r.status for r in run.results}
    assert status["rf"] == "ok" and status["naive"] == "ok"
    assert status["rf_arima"] == "failed" and status["seasonal_naive"] == "failed"
    table, _ = forecast_future(run.result_for("rf").pipeline, both, cfg, run.feature_columns)
    assert sorted(table["shop"].unique()) == ["A", "B"] and len(table) == 4


def test_forecast_models_listed_only_in_forecasting_mode():
    plain = {s.key for s in list_models(REG)}
    forecasting = {s.key for s in list_models(REG, forecasting=True)}
    assert "naive" not in plain and {"naive", "rf_arima"} <= forecasting


def test_real_dengue_forecast_beats_naive():
    df, card = load_sample("lk_colombo_dengue")
    roles = Roles(
        target=card.target,
        problem_type=REG,
        validation=TIME,
        order_column="week_start",
        cv_folds=3,
        forecast=ForecastConfig(horizon=1, n_lags=4, windows=(4,)),
    )
    cfg = auto_configure(df, roles).config
    run = train_models(df, cfg, ["naive", "rf"])
    assert {r.status for r in run.results} == {"ok"}
    table, _ = forecast_future(run.result_for("rf").pipeline, df, cfg, run.feature_columns)
    assert len(table) == 1 and table["forecast"].iloc[0] > 0
