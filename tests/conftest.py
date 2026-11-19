"""Shared fixtures: small messy frames for classification and regression."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    income = rng.lognormal(10, 0.8, n)
    income[rng.choice(n, 10, replace=False)] = np.nan
    income[0] = income[1:].max() * 20  # outlier
    charge = [f"Rs. {int(v):,}" for v in rng.integers(500, 5000, n)]
    city = rng.choice(["Colombo", "kandy ", "GALLE", "Jaffna"], n)
    flag = rng.choice(["Yes", "no", "Y", "N"], n)
    start = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    target = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame(
        {
            "row_id": [f"ID{i:05d}" for i in range(n)],
            "income": income,
            "charge": charge,
            "city": city,
            "flag": flag,
            "start_date": start.strftime("%Y-%m-%d"),
            "constant_col": 1,
            "target": target,
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    y = 3 * x1 - 2 * x2 + (cat == "a") * 1.5 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})
