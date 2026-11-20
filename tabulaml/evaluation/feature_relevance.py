"""How strongly each feature relates to the target (computed on TRAINING rows only).

* **Mutual information** (MI): how much knowing the feature reduces uncertainty about the
  target. Works for numbers and categories and catches non-linear links; 0 = unrelated.
* **Correlation** (Pearson r, number features only): the strength and direction of a
  straight-line relationship with the target (regression) or with the positive class.

Both are screening aids shown before training, not a guarantee of model importance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from tabulaml.core.types import FeatureType, ProblemType
from tabulaml.evaluation.plots import PRIMARY, _layout

MAX_ROWS = 5000
TOP_SHOWN = 25


def _encode(s: pd.Series, ftype: FeatureType) -> tuple[np.ndarray, bool]:
    """Numeric codes for MI: numbers as they are (median-filled), others as category codes."""
    if ftype in (FeatureType.NUMERIC, FeatureType.BOOLEAN):
        values = pd.to_numeric(s, errors="coerce")
        return values.fillna(values.median() if values.notna().any() else 0).to_numpy(), False
    if ftype == FeatureType.DATETIME:
        when = pd.to_datetime(s, errors="coerce")
        days = (when - when.min()).dt.days.astype(float)
        return days.fillna(days.median() if days.notna().any() else 0).to_numpy(), False
    return s.astype(str).astype("category").cat.codes.to_numpy(), True


def feature_relevance(
    frame: pd.DataFrame,
    features: Sequence[str],
    types: dict[str, FeatureType],
    target: str,
    problem_type: ProblemType,
    random_state: int = 42,
) -> pd.DataFrame:
    """One row per feature: type, mutual information and correlation with the target."""
    data = frame[[*features, target]].dropna(subset=[target])
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=random_state)
    if data.empty or not features:
        return pd.DataFrame(columns=["feature", "type", "mutual_information", "correlation"])
    classification = problem_type == ProblemType.CLASSIFICATION
    y_raw = data[target]
    y = (
        y_raw.astype(str).astype("category").cat.codes
        if classification
        else pd.to_numeric(y_raw, errors="coerce")
    )
    keep = pd.notna(y)
    columns, discrete = [], []
    for name in features:
        values, is_discrete = _encode(data[name], types.get(name, FeatureType.CATEGORICAL))
        columns.append(values)
        discrete.append(is_discrete)
    X = np.column_stack(columns)[keep.to_numpy()]
    y = np.asarray(y[keep])
    scorer = mutual_info_classif if classification else mutual_info_regression
    mi = scorer(X, y, discrete_features=np.array(discrete), random_state=random_state)
    rows = []
    for j, name in enumerate(features):
        corr = np.nan
        if types.get(name) == FeatureType.NUMERIC and np.std(X[:, j]) > 0 and np.std(y) > 0:
            target_values = y if not classification or len(np.unique(y)) > 2 else (y == y.max())
            corr = float(np.corrcoef(X[:, j], target_values.astype(float))[0, 1])
        rows.append(
            {
                "feature": name,
                "type": types.get(name, FeatureType.CATEGORICAL).value,
                "mutual_information": float(mi[j]),
                "correlation": corr,
            }
        )
    table = pd.DataFrame(rows).sort_values("mutual_information", ascending=False)
    return table.reset_index(drop=True)


def relevance_fig(table: pd.DataFrame, target: str) -> go.Figure:
    """Horizontal bars of mutual information, strongest at the top."""
    shown = table.head(TOP_SHOWN).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=shown["mutual_information"],
            y=shown["feature"],
            orientation="h",
            marker={"color": PRIMARY},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    title = f"Relevance to {target} (mutual information, training rows)"
    return _layout(
        fig, title, "Mutual information (0 = unrelated)", "", height=max(320, 24 * len(shown) + 120)
    )
