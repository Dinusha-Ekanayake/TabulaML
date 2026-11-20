"""Hyperparameter tuning: random search or grid search (kept deliberately simple).

* **Random search** tries ``n_iter`` random combinations from each model's documented
  ranges. Good when there are many settings: it covers the space cheaply.
* **Grid search** tries EVERY combination of a small list of values per chosen setting.
  Exhaustive and easy to explain, but the number of combinations multiplies quickly.

Both score every combination with the SAME cross-validation as the leaderboard, using
only the dev rows. The held-out test set is never touched, so the final test score stays
honest. (The chosen model's CV score is slightly optimistic, because it was picked as the
best; the test score is the fair one.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tabulaml.core.types import LOWER_IS_BETTER, PreprocessingConfig
from tabulaml.evaluation.metrics import display_value, scoring_map
from tabulaml.models.params import grid_size, search_space, validate_params
from tabulaml.training.pipeline_factory import param_prefix
from tabulaml.training.splits import make_cv

MIN_ITER, MAX_ITER = 5, 100
MAX_GRID = 200  # combinations; larger grids are refused with a clear message


def _clean_grid(key: str, grid: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    """Validate every value (clipped into its documented range, duplicates removed)."""
    clean: dict[str, list[Any]] = {}
    for name, values in grid.items():
        checked = [validate_params(key, {name: v})[name] for v in values]
        clean[name] = list(dict.fromkeys(checked))
    if grid_size(clean) > MAX_GRID:
        raise ValueError(
            f"The grid has {grid_size(clean)} combinations; the limit is {MAX_GRID}. "
            "Search fewer settings or fewer values each."
        )
    return clean


def tune(
    estimator: Any,
    key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    config: PreprocessingConfig,
    primary: str,
    n_iter: int = 0,
    positive: int | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[Any, pd.DataFrame | None]:
    """Return (estimator with the best parameters, search history) or (estimator, None).

    A non-empty ``grid`` runs a grid search; otherwise a random search of ``n_iter``."""
    prefix = param_prefix(estimator)
    space = search_space(key)
    if prefix is None or not (space or grid):  # ensembles: tune their base models instead
        return estimator, None
    common = {
        "scoring": scoring_map(config.problem_type, y, positive)[primary],
        "cv": make_cv(config, y),
        "refit": False,
        "error_score": np.nan,
        "n_jobs": 1,
    }
    if grid:
        params = {f"{prefix}{n}": v for n, v in _clean_grid(key, grid).items()}
        search: Any = GridSearchCV(estimator, params, **common)
    else:
        search = RandomizedSearchCV(
            estimator,
            {f"{prefix}{name}": dist for name, dist in space.items()},
            n_iter=int(min(max(n_iter, MIN_ITER), MAX_ITER)),
            random_state=config.random_state,
            **common,
        )
    search.fit(X, y, groups=groups)
    return estimator.set_params(**search.best_params_), _history(search, prefix, primary)


def _history(search: Any, prefix: str, primary: str) -> pd.DataFrame:
    history = pd.DataFrame(search.cv_results_["params"]).rename(
        columns=lambda c: c.removeprefix(prefix)
    )
    history[f"cv_{primary}"] = [
        display_value(primary, v) for v in search.cv_results_["mean_test_score"]
    ]
    history = history.sort_values(f"cv_{primary}", ascending=primary in LOWER_IS_BETTER)
    return history.reset_index(drop=True)
