"""Pure Plotly figure builders (no Streamlit import).

Colours follow the validated reference palette (dataviz skill): categorical hues in a fixed
order, a single blue ramp for magnitude, neutral grey for reference lines. Backgrounds are
transparent so the host theme (light or dark) supplies the surface and text colour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
PRIMARY = SERIES[0]
EMPHASIS = "#184f95"  # darker step of the same blue ramp, used for "best"
REFERENCE = "#8c8b86"
BLUE_RAMP = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#104281"]]
MAX_SERIES = len(SERIES)
TOP_FEATURES = 15


def _layout(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"namelength": -1},
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def leaderboard_bar(
    board: pd.DataFrame, metric: str, label: str, lower_is_better: bool
) -> go.Figure:
    col = f"test_{metric}"
    data = board[board["status"] == "ok"].dropna(subset=[col])
    data = data.sort_values(col, ascending=not lower_is_better)
    best = data[col].min() if lower_is_better else data[col].max()
    colors = [EMPHASIS if v == best else PRIMARY for v in data[col]]
    fig = go.Figure(
        go.Bar(
            x=data[col],
            y=data["model"],
            orientation="h",
            marker={"color": colors},
            text=[f"{v:.3f}" for v in data[col]],
            textposition="outside",
            hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    direction = "lower is better" if lower_is_better else "higher is better"
    return _layout(
        fig, f"Test {label} by model ({direction})", x=label, height=max(260, 46 * len(data) + 90)
    )


def cv_boxplot(cv_scores: dict[str, Sequence[float]], label: str) -> go.Figure:
    fig = go.Figure()
    for name, scores in cv_scores.items():
        fig.add_trace(
            go.Box(
                y=list(scores),
                name=name,
                marker={"color": PRIMARY},
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                line={"width": 2},
                showlegend=False,
            )
        )
    return _layout(fig, f"Cross-validation {label} per fold (spread = stability)", y=label)


def confusion_matrix_fig(y_true: Any, y_pred: Any, labels: Sequence[str]) -> go.Figure:
    idx = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    text = [[f"{cm[i, j]}<br>{row_pct[i, j]:.0%}" for j in idx] for i in idx]
    fig = go.Figure(
        go.Heatmap(
            z=row_pct,
            x=list(labels),
            y=list(labels),
            text=text,
            texttemplate="%{text}",
            colorscale=BLUE_RAMP,
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Confusion matrix (count and % of actual class)", "Predicted", "Actual")


def _class_curves(y_true: Any, proba: np.ndarray, labels: Sequence[str]) -> list[tuple[str, int]]:
    if proba.shape[1] == 2:
        return [(labels[1], 1)]
    return [(lab, i) for i, lab in enumerate(labels)][:MAX_SERIES]


def roc_fig(y_true: Any, proba: np.ndarray, labels: Sequence[str]) -> go.Figure:
    y_true = np.asarray(y_true)
    fig = go.Figure()
    for k, (label, i) in enumerate(_class_curves(y_true, proba, labels)):
        fpr, tpr, _ = roc_curve(y_true == i, proba[:, i])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} (AUC {auc(fpr, tpr):.3f})",
                line={"color": SERIES[k], "width": 2},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"color": REFERENCE, "dash": "dash", "width": 1},
        )
    )
    return _layout(fig, "ROC curve (one-vs-rest)", "False positive rate", "True positive rate")


def pr_fig(y_true: Any, proba: np.ndarray, labels: Sequence[str]) -> go.Figure:
    y_true = np.asarray(y_true)
    fig = go.Figure()
    for k, (label, i) in enumerate(_class_curves(y_true, proba, labels)):
        precision, recall, _ = precision_recall_curve(y_true == i, proba[:, i])
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=str(label),
                line={"color": SERIES[k], "width": 2},
            )
        )
        if proba.shape[1] == 2:
            base = float(np.mean(y_true == i))
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[base, base],
                    mode="lines",
                    name=f"Chance ({base:.2f})",
                    line={"color": REFERENCE, "dash": "dash", "width": 1},
                )
            )
    return _layout(fig, "Precision-recall curve", "Recall", "Precision")


def pred_vs_actual_fig(y_true: Any, y_pred: Any) -> go.Figure:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig = go.Figure(
        [
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Rows",
                marker={
                    "color": PRIMARY,
                    "size": 8,
                    "opacity": 0.6,
                    "line": {"width": 1, "color": "rgba(255,255,255,0.8)"},
                },
                hovertemplate="Actual %{x:.3f}<br>Predicted %{y:.3f}<extra></extra>",
            ),
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect prediction",
                line={"color": REFERENCE, "dash": "dash", "width": 1},
            ),
        ]
    )
    return _layout(fig, "Predicted vs actual", "Actual", "Predicted")


def residuals_fig(y_true: Any, y_pred: Any) -> go.Figure:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    fig = go.Figure(
        go.Scatter(
            x=y_pred,
            y=y_true - y_pred,
            mode="markers",
            name="Residual",
            marker={"color": PRIMARY, "size": 8, "opacity": 0.6},
            hovertemplate="Predicted %{x:.3f}<br>Residual %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": REFERENCE, "dash": "dash", "width": 1})
    return _layout(fig, "Residuals (actual - predicted)", "Predicted", "Residual")


def importance_fig(importance: pd.DataFrame, metric_label: str) -> go.Figure:
    top = importance.head(TOP_FEATURES).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["importance_mean"],
            y=top["feature"],
            orientation="h",
            marker={"color": PRIMARY},
            error_x={"type": "data", "array": top["importance_std"], "color": REFERENCE},
            hovertemplate="%{y}<br>Drop in " + metric_label + ": %{x:.4f}<extra></extra>",
        )
    )
    return _layout(
        fig,
        f"Permutation importance (drop in {metric_label} when shuffled)",
        x=f"Drop in {metric_label}",
        height=max(280, 28 * len(top) + 100),
    )


def class_distribution_fig(y: pd.Series, title: str = "Target classes") -> go.Figure:
    counts = y.astype(str).value_counts()
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker={"color": PRIMARY},
            text=[f"{v / counts.sum():.1%}" for v in counts.values],
            textposition="outside",
            hovertemplate="%{x}: %{y} rows<extra></extra>",
        )
    )
    return _layout(fig, title, y="Rows")


def target_histogram_fig(y: pd.Series, title: str = "Target distribution") -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=y, marker={"color": PRIMARY, "line": {"width": 1, "color": "white"}})
    )
    return _layout(fig, title, x=str(y.name or "target"), y="Rows")


def missingness_fig(missing: pd.Series) -> go.Figure:
    data = missing[missing > 0].sort_values()
    fig = go.Figure(
        go.Bar(
            x=data.values * 100,
            y=data.index,
            orientation="h",
            marker={"color": PRIMARY},
            text=[f"{v:.1%}" for v in data.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>",
        )
    )
    return _layout(
        fig, "Missing values by column", x="% missing", height=max(220, 26 * len(data) + 100)
    )


def probability_fig(probabilities: dict[str, float]) -> go.Figure:
    labels = list(probabilities)
    values = [probabilities[k] for k in labels]
    best = max(values)
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": [EMPHASIS if v == best else PRIMARY for v in values]},
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
    return _layout(fig, "Predicted probability by class", height=max(200, 50 * len(labels) + 80))
