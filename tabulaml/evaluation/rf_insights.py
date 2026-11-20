"""Random Forest dashboard data: RF vs boosting head-to-head and a look inside the forest.

Pure functions (no Streamlit), shared by the UI and tests. The head-to-head uses the
cross-validation folds, which are identical for every model, so comparing fold by fold is
fair and never touches the held-out test set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tabulaml.core.types import LOWER_IS_BETTER, TrainingRun
from tabulaml.training.pipeline_factory import inner_model

RIVALS = ("xgb", "lgbm", "catboost", "gb", "histgb", "extratrees", "bagging", "dt")
TIE_TOLERANCE = 1e-9
TOP_FEATURES = 15


def _usable(run: TrainingRun, key: str) -> bool:
    try:
        return run.result_for(key).status in ("ok", "incomplete")
    except KeyError:
        return False


def head_to_head(run: TrainingRun, metric: str | None = None) -> pd.DataFrame:
    """One row per rival: mean CV scores, difference and RF's fold wins / losses / ties."""
    metric = metric or run.primary_metric
    if not _usable(run, "rf"):
        return pd.DataFrame()
    rf = np.asarray(run.result_for("rf").cv_scores.get(metric, ()), dtype=float)
    lower = metric in LOWER_IS_BETTER
    rows = []
    for key in RIVALS:
        if not _usable(run, key):
            continue
        other = np.asarray(run.result_for(key).cv_scores.get(metric, ()), dtype=float)
        if len(other) != len(rf) or not len(rf):
            continue
        diff = other - rf if lower else rf - other  # > 0: RF better on that fold
        valid = ~np.isnan(diff)
        rows.append(
            {
                "rival": run.result_for(key).name,
                "rf_mean": float(np.nanmean(rf)),
                "rival_mean": float(np.nanmean(other)),
                "rf_advantage": float(np.nanmean(diff)),
                "rf_wins": int((diff[valid] > TIE_TOLERANCE).sum()),
                "rf_losses": int((diff[valid] < -TIE_TOLERANCE).sum()),
                "ties": int((np.abs(diff[valid]) <= TIE_TOLERANCE).sum()),
            }
        )
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame) -> str | None:
    """Plain-language summary of the head-to-head (no statistics beyond fold counts)."""
    if table.empty:
        return None
    best = table.sort_values("rf_advantage").iloc[0]  # the strongest rival
    folds = int(best["rf_wins"] + best["rf_losses"] + best["ties"])
    if best["rf_wins"] > best["rf_losses"]:
        outcome = "Random Forest is ahead of every rival here"
    elif best["rf_wins"] < best["rf_losses"]:
        outcome = f"{best['rival']} beats Random Forest"
    else:
        outcome = f"Random Forest and {best['rival']} are level"
    return (
        f"{outcome}: against its strongest rival ({best['rival']}) it wins "
        f"{int(best['rf_wins'])} of {folds} folds and loses {int(best['rf_losses'])}. "
        "With few folds, a 1-fold difference is not a real difference."
    )


def _forest(pipeline: Any) -> Any:
    """The FITTED forest (a target-transformed pipeline keeps it in ``regressor_``)."""
    fitted = getattr(pipeline, "regressor_", None) or pipeline
    model = inner_model(fitted)
    return model if hasattr(model, "estimators_") and hasattr(model, "max_features") else None


def forest_summary(pipeline: Any) -> dict[str, float] | None:
    """Trees, average depth and average leaves of a fitted Random Forest."""
    forest = _forest(pipeline)
    if forest is None:
        return None
    trees = forest.estimators_
    return {
        "trees": float(len(trees)),
        "mean_depth": float(np.mean([t.get_depth() for t in trees])),
        "mean_leaves": float(np.mean([t.get_n_leaves() for t in trees])),
        "max_features": str(forest.max_features),
    }


def _feature_names(pipeline: Any, n: int) -> list[str]:
    est = getattr(pipeline, "regressor_", None) or pipeline
    steps = getattr(est, "named_steps", {})
    features = steps.get("features") if steps else None
    try:
        names = [str(x).split("__", 1)[-1] for x in features.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - some encoders cannot name their outputs
        names = []
    return names if len(names) == n else [f"feature {i}" for i in range(n)]


def impurity_importance(pipeline: Any) -> pd.DataFrame:
    """The forest's own (impurity-based) importance, on the encoded features.

    Quick and built in, but it favours features with many distinct values; the
    permutation importance in the final test tab is the more reliable measure."""
    forest = _forest(pipeline)
    if forest is None or not hasattr(forest, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    values = np.asarray(forest.feature_importances_, dtype=float)
    frame = pd.DataFrame({"feature": _feature_names(pipeline, len(values)), "importance": values})
    return (
        frame.sort_values("importance", ascending=False).head(TOP_FEATURES).reset_index(drop=True)
    )
