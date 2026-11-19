"""Model-agnostic permutation importance on the ORIGINAL columns.

The whole pipeline (cleaner + preprocessing + model) is permuted column by column on the
held-out test set, so importances refer to the columns the user knows, not to encoded
one-hot or target-encoded features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance

from tabulaml.core.types import ProblemType
from tabulaml.evaluation.metrics import scoring_map

MAX_ROWS = 2000


def permutation_importance_df(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    problem_type: ProblemType,
    metric: str,
    y_reference: Any = None,
    n_repeats: int = 5,
    random_state: int = 42,
    positive: int | None = None,
) -> pd.DataFrame:
    """Drop in ``metric`` when each column is shuffled (bigger = more important)."""
    X = X_test
    y = pd.Series(y_test, index=X_test.index)
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=random_state)
        y = y.loc[X.index]
    reference = y_reference if y_reference is not None else y_test
    scorers = scoring_map(problem_type, reference, positive)
    result = permutation_importance(
        pipeline,
        X,
        y.to_numpy(),
        scoring=scorers[metric],
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
