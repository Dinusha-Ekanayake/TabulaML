"""Automatic hyperparameter tuning with a random search (kept deliberately simple).

The search tries ``n_iter`` random combinations from each model's documented ranges and
scores every combination with the SAME cross-validation as the leaderboard, using only the
dev rows. The held-out test set is never touched, so the final test score stays honest.
(The chosen model's CV score is slightly optimistic, because it was picked as the best;
the test score is the fair one.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from tabulaml.core.types import LOWER_IS_BETTER, PreprocessingConfig
from tabulaml.evaluation.metrics import display_value, scoring_map
from tabulaml.models.params import search_space
from tabulaml.training.pipeline_factory import param_prefix
from tabulaml.training.splits import make_cv

MIN_ITER, MAX_ITER = 5, 100


def tune(
    estimator: Any,
    key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    config: PreprocessingConfig,
    primary: str,
    n_iter: int,
    positive: int | None = None,
) -> tuple[Any, pd.DataFrame | None]:
    """Return (estimator with the best parameters, search history) or (estimator, None)."""
    prefix = param_prefix(estimator)
    space = search_space(key)
    if prefix is None or not space:  # ensembles: tune their base models instead
        return estimator, None
    search = RandomizedSearchCV(
        estimator,
        {f"{prefix}{name}": dist for name, dist in space.items()},
        n_iter=int(min(max(n_iter, MIN_ITER), MAX_ITER)),
        scoring=scoring_map(config.problem_type, y, positive)[primary],
        cv=make_cv(config, y),
        refit=False,
        random_state=config.random_state,
        error_score=np.nan,
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    best = {k: v for k, v in search.best_params_.items()}
    history = pd.DataFrame(search.cv_results_["params"]).rename(
        columns=lambda c: c.removeprefix(prefix)
    )
    history[f"cv_{primary}"] = [
        display_value(primary, v) for v in search.cv_results_["mean_test_score"]
    ]
    history = history.sort_values(f"cv_{primary}", ascending=primary in LOWER_IS_BETTER)
    return estimator.set_params(**best), history.reset_index(drop=True)
