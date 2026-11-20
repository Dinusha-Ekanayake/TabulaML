"""Step 5 tab: the Random Forest dashboard (RF vs boosting, inside the forest).

Uses cross-validation results only, so it is available before the test set is revealed.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from tabulaml.core.types import TrainingRun
from tabulaml.evaluation.plots import PRIMARY, SERIES, _layout
from tabulaml.evaluation.rf_insights import (
    RIVALS,
    forest_summary,
    head_to_head,
    impurity_importance,
    verdict,
)
from tabulaml.ui.helpers import metric_label


def has_rf(run: TrainingRun) -> bool:
    return any(r.key == "rf" and r.status in ("ok", "incomplete") for r in run.results)


def _kpis(run: TrainingRun) -> None:
    rf = run.result_for("rf")
    scores = np.asarray(rf.cv_scores.get(run.primary_metric, ()), dtype=float)
    summary = forest_summary(rf.pipeline) or {}
    cells = [
        (f"CV {metric_label(run.primary_metric)}", f"{np.nanmean(scores):.4f}"),
        ("Spread across folds (std)", f"{np.nanstd(scores):.4f}"),
        ("Trees", f"{summary.get('trees', 0):.0f}" if summary else "-"),
        ("Average tree depth", f"{summary.get('mean_depth', 0):.1f}" if summary else "-"),
        ("Fit time", f"{rf.fit_seconds:.1f} s"),
    ]
    for col, (label, value) in zip(st.columns(len(cells)), cells, strict=True):
        with col.container(border=True, key=f"rf_kpi_{label[:6]}"):
            st.metric(label, value)


def _fold_chart(run: TrainingRun) -> go.Figure | None:
    metric = run.primary_metric
    lines = {}
    for key in ("rf", *RIVALS):
        try:
            r = run.result_for(key)
        except KeyError:
            continue
        if r.status in ("ok", "incomplete") and metric in r.cv_scores:
            lines[r.name] = r.cv_scores[metric]
    if len(lines) < 2:
        return None
    fig = go.Figure()
    for i, (name, scores) in enumerate(lines.items()):
        is_rf = i == 0
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(scores) + 1)),
                y=list(scores),
                mode="lines+markers",
                name=name,
                line={"width": 3 if is_rf else 1.5, "color": PRIMARY if is_rf else SERIES[i % 8]},
                opacity=1.0 if is_rf else 0.75,
            )
        )
    fig.update_xaxes(dtick=1)
    return _layout(
        fig, "Fold by fold: Random Forest vs the others", "CV fold", metric_label(metric)
    )


def rf_dashboard(run: TrainingRun) -> None:
    st.caption(
        "Random Forest = many decision trees, each trained on a bootstrap sample of the rows "
        "with a random subset of features at every split; their votes are averaged. Bagging "
        "lowers the variance of single trees. Boosting instead builds trees one after another, "
        "each fixing the previous errors. All numbers here come from cross-validation."
    )
    _kpis(run)
    table = head_to_head(run)
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        fig = _fold_chart(run)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Train Random Forest together with XGBoost, LightGBM or CatBoost to compare.")
    with right, st.container(border=True, key="rf_h2h"):
        st.markdown("**Head-to-head (same CV folds)**")
        sentence = verdict(table)
        if sentence:
            st.write(sentence)
            st.caption(
                "RF advantage = how much better RF's mean CV score is (positive = RF better)."
            )
            shown = table[["rival", "rf_advantage", "rf_wins", "rf_losses", "ties"]].rename(
                columns={
                    "rival": "Rival",
                    "rf_advantage": "RF advantage",
                    "rf_wins": "RF wins",
                    "rf_losses": "RF losses",
                    "ties": "Ties",
                }
            )
            st.dataframe(shown.round(4), hide_index=True, width="stretch")
        else:
            st.caption("No boosting or tree rival was trained in this run.")
    _importance(run)


def _importance(run: TrainingRun) -> None:
    imp = impurity_importance(run.result_for("rf").pipeline)
    if imp.empty:
        return
    fig = go.Figure(
        go.Bar(
            x=imp["importance"][::-1],
            y=imp["feature"][::-1],
            orientation="h",
            marker={"color": PRIMARY},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig = _layout(fig, "Inside the forest: built-in importance", "Mean decrease in impurity", "")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Built-in importance is quick but favours columns with many distinct values. "
        "The permutation importance in the final test tab is the fairer measure."
    )
