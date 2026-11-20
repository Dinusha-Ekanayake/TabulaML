"""Forecasting benchmark on the real Colombo dengue series (uses the app's engine).

    python scripts/forecast_benchmark.py            # writes reports/forecast_summary.md

Horizons 1 and 4 weeks; naive baselines, Ridge, Random Forest, XGBoost and the RF + ARIMA
hybrid; time-ordered 5-fold CV on the training weeks plus the held-out latest 20 %.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tabulaml.core.types import ForecastConfig, ProblemType, ValidationScheme
from tabulaml.data.samples import load_sample
from tabulaml.training.trainer import train_models
from tabulaml.workflow import Roles, auto_configure

logger = logging.getLogger("forecast_benchmark")
MODELS = ["naive", "seasonal_naive", "ridge", "rf", "xgb", "rf_arima"]
OUT = Path("reports/forecast_summary.md")


def run_horizon(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    fc = ForecastConfig(horizon=horizon, n_lags=4, seasonal_lag=52, windows=(4,))
    roles = Roles(
        target=target,
        problem_type=ProblemType.REGRESSION,
        validation=ValidationScheme.TIME_ORDERED,
        order_column="week_start",
        forecast=fc,
    )
    run = train_models(df, auto_configure(df, roles).config, MODELS)
    naive = run.result_for("naive").test_metrics.get("rmse")
    rows = []
    for r in run.results:
        if r.status == "failed":
            rows.append({"horizon": horizon, "model": r.name, "status": r.error})
            continue
        cv = pd.Series(r.cv_scores.get("rmse", ())).dropna()
        test = r.test_metrics.get("rmse")
        rows.append(
            {
                "horizon": horizon,
                "model": r.name,
                "cv_rmse": f"{cv.mean():.1f} ± {cv.std():.1f}",
                "test_rmse": round(test, 1),
                "test_mae": round(r.test_metrics.get("mae"), 1),
                "vs_naive": f"{(naive - test) / naive:+.1%}" if naive else "",
            }
        )
    return pd.DataFrame(rows)


def _markdown(table: pd.DataFrame) -> str:
    cells = table.fillna("").astype(str)
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.to_numpy()]
    return chr(10).join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df, card = load_sample("lk_colombo_dengue")
    table = pd.concat([run_horizon(df, card.target, h) for h in (1, 4)], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# Forecasting benchmark: Colombo weekly dengue cases\n\n"
        "Time-ordered 5-fold CV on the training weeks; test = latest 20 % of weeks. "
        "Features: target lags h..h+3, seasonal lag 52, 4-week rolling mean, calendar parts, "
        "weather lagged by h. `vs_naive` = RMSE improvement over the naive baseline on the "
        "test weeks (positive = better).\n\n" + _markdown(table) + "\n",
        encoding="utf-8",
    )
    logger.info("%s", table.to_string(index=False))
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
