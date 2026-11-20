"""Exploratory data analysis: column summaries and distribution charts (no Streamlit).

Used before preprocessing so the user can understand the data: the shape of every column,
its relationship with the target (training rows only) and correlations between numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tabulaml.core.types import FeatureType, ProblemType
from tabulaml.evaluation.plots import PRIMARY, SERIES, _layout

TOP_CATEGORIES = 20
MAX_POINTS = 3000
MAX_CORR_COLUMNS = 25
BINS = 40


def column_summary(s: pd.Series, ftype: FeatureType) -> dict[str, Any]:
    """Key facts about one (cleaned) column, for a row of stat cards."""
    present = s.dropna()
    out: dict[str, Any] = {
        "rows": int(len(s)),
        "missing": f"{s.isna().mean():.1%}",
        "distinct": int(present.nunique()),
    }
    if ftype == FeatureType.NUMERIC and len(present):
        values = pd.to_numeric(present, errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean()
        out |= {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "skew": float(values.skew()) if len(values) > 2 else 0.0,
            "outliers (1.5 IQR)": f"{outliers:.1%}",
        }
    elif len(present):
        counts = present.astype(str).value_counts()
        out |= {
            "most common": str(counts.index[0]),
            "its share": f"{counts.iloc[0] / len(present):.1%}",
        }
    return out


def _numeric_fig(values: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(
        [
            go.Histogram(
                x=values,
                nbinsx=BINS,
                marker={"color": PRIMARY, "line": {"width": 1, "color": "rgba(255,255,255,.6)"}},
                name="Rows",
                hovertemplate="%{x}<br>%{y} rows<extra></extra>",
            ),
            go.Box(
                x=values,
                name="Spread",
                marker={"color": SERIES[1]},
                boxpoints=False,
                yaxis="y2",
                hoverinfo="x",
            ),
        ]
    )
    fig.update_layout(
        yaxis={"domain": [0, 0.8]},
        yaxis2={"domain": [0.85, 1], "showticklabels": False},
        showlegend=False,
    )
    return _layout(fig, f"Distribution of {name}", name, "Rows")


def _category_fig(values: pd.Series, name: str) -> go.Figure:
    counts = values.astype(str).value_counts()
    shown = counts.head(TOP_CATEGORIES)
    title = f"Most common values of {name}"
    if len(counts) > TOP_CATEGORIES:
        title += f" (top {TOP_CATEGORIES} of {len(counts)})"
    fig = go.Figure(
        go.Bar(
            x=shown.to_numpy()[::-1],
            y=shown.index.to_numpy()[::-1],
            orientation="h",
            marker={"color": PRIMARY},
            hovertemplate="%{y}: %{x} rows<extra></extra>",
        )
    )
    return _layout(fig, title, "Rows", "", height=max(320, 22 * len(shown) + 120))


def _datetime_fig(values: pd.Series, name: str) -> go.Figure:
    when = pd.to_datetime(values, errors="coerce").dropna()
    span_days = (when.max() - when.min()).days if len(when) else 0
    freq = "D" if span_days <= 120 else "W" if span_days <= 1500 else "MS"
    counts = when.dt.to_period(freq[0]).dt.start_time.value_counts().sort_index()
    fig = go.Figure(
        go.Scatter(
            x=counts.index,
            y=counts.to_numpy(),
            mode="lines+markers",
            line={"color": PRIMARY, "width": 2},
            marker={"size": 4},
        )
    )
    return _layout(fig, f"Rows over time ({name})", name, "Rows")


def distribution_fig(s: pd.Series, ftype: FeatureType, name: str) -> go.Figure:
    """Histogram + box for numbers, top categories for text, counts over time for dates."""
    present = s.dropna()
    if ftype == FeatureType.NUMERIC:
        return _numeric_fig(pd.to_numeric(present, errors="coerce").dropna(), name)
    if ftype == FeatureType.DATETIME:
        return _datetime_fig(present, name)
    return _category_fig(present, name)


def _sample(x: pd.Series, y: pd.Series, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) > MAX_POINTS:
        frame = frame.sample(MAX_POINTS, random_state=seed)
    return frame["x"], frame["y"]


def target_relation_fig(
    x: pd.Series,
    y: pd.Series,
    ftype: FeatureType,
    problem_type: ProblemType,
    name: str,
    target: str,
) -> go.Figure:
    """How ``x`` relates to the target (call with TRAINING rows only)."""
    x, y = _sample(x, y)
    numeric_x = ftype == FeatureType.NUMERIC
    if problem_type == ProblemType.CLASSIFICATION:
        return _by_class(pd.to_numeric(x, errors="coerce") if numeric_x else x, y, numeric_x, name)
    if numeric_x:
        fig = go.Figure(
            go.Scatter(
                x=pd.to_numeric(x, errors="coerce"),
                y=pd.to_numeric(y, errors="coerce"),
                mode="markers",
                marker={"color": PRIMARY, "size": 6, "opacity": 0.5},
                hovertemplate=f"{name} %{{x}}<br>{target} %{{y}}<extra></extra>",
            )
        )
        return _layout(fig, f"{target} against {name} (training rows)", name, target)
    top = x.astype(str).value_counts().head(15).index
    keep = x.astype(str).isin(top)
    fig = go.Figure(
        go.Box(
            x=x[keep].astype(str),
            y=pd.to_numeric(y[keep], errors="coerce"),
            marker={"color": PRIMARY},
            boxpoints=False,
        )
    )
    return _layout(fig, f"{target} by {name} (training rows)", name, target)


def _by_class(x: pd.Series, y: pd.Series, numeric_x: bool, name: str) -> go.Figure:
    classes = y.astype(str).value_counts().index[: len(SERIES)]
    fig = go.Figure()
    if numeric_x:
        for i, label in enumerate(classes):
            fig.add_trace(
                go.Box(
                    x=x[y.astype(str) == label],
                    name=str(label),
                    marker={"color": SERIES[i]},
                    boxpoints=False,
                )
            )
        return _layout(fig, f"{name} by class (training rows)", name, "Class")
    top = x.astype(str).value_counts().head(15).index
    share = pd.crosstab(x.astype(str), y.astype(str), normalize="index").loc[
        lambda d: d.index.isin(top)
    ]
    for i, label in enumerate(c for c in classes if c in share.columns):
        fig.add_trace(
            go.Bar(x=share.index, y=share[label], name=str(label), marker={"color": SERIES[i]})
        )
    fig.update_layout(barmode="stack", yaxis={"tickformat": ".0%"})
    return _layout(fig, f"Class share by {name} (training rows)", name, "Share of rows")


def correlation_fig(df: pd.DataFrame) -> go.Figure | None:
    """Pearson correlations between numeric columns (diverging colours, grey = none)."""
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return None
    if numeric.shape[1] > MAX_CORR_COLUMNS:
        numeric = numeric[numeric.var().sort_values(ascending=False).index[:MAX_CORR_COLUMNS]]
    corr = numeric.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(),
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#c0392b"], [0.5, "#e8e8e8"], [1, "#2a78d6"]],
            hovertemplate="%{y} vs %{x}: %{z}<extra></extra>",
        )
    )
    size = max(380, 26 * len(corr) + 160)
    return _layout(fig, "Correlation between number columns", "", "", height=size)


def strongest_pairs(df: pd.DataFrame, n: int = 5) -> list[tuple[str, str, float]]:
    """The most strongly correlated column pairs (|r|), for a plain-language note."""
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs().where(lambda c: np.triu(np.ones(c.shape, dtype=bool), k=1))
    pairs = corr.stack().sort_values(ascending=False).head(n)
    return [(str(a), str(b), float(numeric[a].corr(numeric[b]))) for (a, b), _ in pairs.items()]
