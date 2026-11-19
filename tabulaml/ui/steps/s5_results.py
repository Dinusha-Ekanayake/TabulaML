"""Step 5: leaderboard, comparison charts and per-model diagnostics."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from tabulaml.core.types import LOWER_IS_BETTER, ProblemType, TrainingRun
from tabulaml.evaluation import plots
from tabulaml.evaluation.importance import permutation_importance_df
from tabulaml.ui.components import nav_buttons
from tabulaml.ui.helpers import metric_label, rf_vs_boosting, run_report
from tabulaml.ui.state import AppState, set_state

TILE_METRICS = {
    ProblemType.CLASSIFICATION: ("accuracy", "f1", "f1_macro", "roc_auc", "pr_auc"),
    ProblemType.REGRESSION: ("rmse", "mae", "r2"),
}


def _tiles(run: TrainingRun) -> None:
    best = run.best()
    st.success(f"Best model by cross-validated {metric_label(run.primary_metric)}: **{best.name}**")
    shown = [m for m in TILE_METRICS[run.problem_type] if m in best.test_metrics][:4]
    for col, metric in zip(st.columns(len(shown) or 1), shown, strict=False):
        col.metric(f"Test {metric_label(metric)}", f"{best.test_metrics[metric]:.4f}")


def _leaderboard(run: TrainingRun, name: str) -> None:
    board = run.leaderboard()
    st.dataframe(board.drop(columns=["key"]), hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.download_button(
        "Download leaderboard (CSV)",
        board.to_csv(index=False).encode(),
        "leaderboard.csv",
        "text/csv",
        key="dl_board",
    )
    report = json.dumps(run_report(run, name), indent=2, default=str).encode()
    right.download_button(
        "Download run report (JSON)", report, "run_report.json", "application/json", key="dl_report"
    )
    for note in run.notes:
        st.caption(note)


def _comparison(run: TrainingRun) -> None:
    metric = run.primary_metric
    label = metric_label(metric)
    board = run.leaderboard()
    if f"test_{metric}" in board:
        st.plotly_chart(
            plots.leaderboard_bar(board, metric, label, metric in LOWER_IS_BETTER), width="stretch"
        )
    cv = {
        r.name: r.cv_scores[metric]
        for r in run.results
        if r.status == "ok" and metric in r.cv_scores
    }
    if cv:
        st.plotly_chart(plots.cv_boxplot(cv, label), width="stretch")
    sentence = rf_vs_boosting(run)
    if sentence:
        st.info(f"**Random Forest vs gradient boosting:** {sentence}", icon="\U0001f332")


def _classification_detail(run: TrainingRun, result) -> None:  # noqa: ANN001
    labels = list(run.class_labels or [])
    st.plotly_chart(plots.confusion_matrix_fig(run.y_test, result.y_pred, labels), width="stretch")
    if result.y_proba is not None:
        left, right = st.columns(2)
        left.plotly_chart(
            plots.roc_fig(run.y_test, np.asarray(result.y_proba), labels), width="stretch"
        )
        right.plotly_chart(
            plots.pr_fig(run.y_test, np.asarray(result.y_proba), labels), width="stretch"
        )


def _regression_detail(run: TrainingRun, result) -> None:  # noqa: ANN001
    left, right = st.columns(2)
    left.plotly_chart(plots.pred_vs_actual_fig(run.y_test, result.y_pred), width="stretch")
    right.plotly_chart(plots.residuals_fig(run.y_test, result.y_pred), width="stretch")


def _importance(state: AppState, result) -> None:  # noqa: ANN001
    run = state.run
    if result.key not in state.importance:
        if not st.button("Compute feature importance", key=f"imp_{result.key}"):
            return
        with st.spinner("Shuffling each column on the test set..."):
            imp = permutation_importance_df(
                result.pipeline, run.X_test, run.y_test, run.problem_type, run.primary_metric
            )
        set_state(importance={**state.importance, result.key: imp})
        st.rerun()
    st.plotly_chart(
        plots.importance_fig(state.importance[result.key], metric_label(run.primary_metric)),
        width="stretch",
    )


def _details(state: AppState) -> None:
    run = state.run
    ok = [r for r in run.results if r.status == "ok"]
    names = {r.name: r for r in ok}
    choice = st.selectbox(
        "Inspect model", list(names), index=list(names).index(run.best().name), key="inspect_model"
    )
    result = names[choice]
    if run.problem_type == ProblemType.CLASSIFICATION:
        _classification_detail(run, result)
    else:
        _regression_detail(run, result)
    _importance(state, result)


def render(state: AppState) -> None:
    st.header("5. Results")
    run = state.run
    failed = [r for r in run.results if r.status != "ok"]
    for r in failed:
        st.error(f"{r.name} failed: {r.error}")
    if len(failed) == len(run.results):
        nav_buttons(back=4, next_step=None)
        return
    name = state.card.title if state.card else (state.filename or "uploaded data")
    _tiles(run)
    board_tab, compare_tab, detail_tab = st.tabs(["Leaderboard", "Comparison", "Model details"])
    with board_tab:
        _leaderboard(run, name)
    with compare_tab:
        _comparison(run)
    with detail_tab:
        _details(state)
    nav_buttons(back=4, next_step=6, next_label="Use a model")
