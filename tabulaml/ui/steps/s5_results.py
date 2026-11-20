"""Step 5: results with a HOLDOUT LOCK.

Leaderboard, comparison and diagnostics use cross-validation on the training (dev) rows
only. The held-out test set stays hidden until the user deliberately reveals it, so model
choices are not tuned against the test set.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from tabulaml.core.types import LOWER_IS_BETTER, ModelResult, ProblemType, TrainingRun
from tabulaml.evaluation import plots
from tabulaml.evaluation.importance import permutation_importance_df
from tabulaml.evaluation.metrics import best_threshold, threshold_curve
from tabulaml.models.params import param_specs
from tabulaml.ui.components import nav_buttons
from tabulaml.ui.helpers import display_board, metric_label, rf_vs_boosting, run_report
from tabulaml.ui.state import AppState, get_state, set_state
from tabulaml.ui.steps.s5_forecast import forecast_test_view
from tabulaml.ui.steps.s5_rf import has_rf, rf_dashboard

TILE_METRICS = {
    ProblemType.CLASSIFICATION: (
        "accuracy",
        "f1",
        "f1_macro",
        "subset_accuracy",
        "roc_auc",
        "pr_auc",
    ),
    ProblemType.REGRESSION: ("rmse", "mae", "r2"),
}


def _cv_tiles(run: TrainingRun) -> None:
    best = run.best()
    st.success(
        f"Best model by cross-validated {metric_label(run.primary_metric)}: **{best.name}** "
        "(chosen on the training rows only).",
        icon=":material/emoji_events:",
    )
    shown = [m for m in TILE_METRICS[run.problem_type] if m in best.cv_scores][:4]
    for col, metric in zip(st.columns(len(shown) or 1), shown, strict=False):
        scores = np.asarray(best.cv_scores[metric], dtype=float)
        col.metric(
            f"CV {metric_label(metric)}",
            f"{np.nanmean(scores):.4f}",
            help=f"Mean over folds; spread (std) {np.nanstd(scores):.4f}.",
        )


def _leaderboard(run: TrainingRun, name: str, revealed: bool) -> None:
    board = run.leaderboard().drop(columns=["key"])
    if not revealed:
        board = board[[c for c in board.columns if not c.startswith("test_")]]
        st.caption(
            ":material/lock: Test-set columns are hidden until you reveal the final "
            "test results (last tab)."
        )
    st.dataframe(display_board(board), hide_index=True, width="stretch")
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
    cv = {
        r.name: r.cv_scores[metric]
        for r in run.results
        if r.status in ("ok", "incomplete") and metric in r.cv_scores
    }
    if cv:
        st.plotly_chart(plots.cv_boxplot(cv, label), width="stretch")
        st.caption(
            "Each dot is one cross-validation fold. Overlapping boxes mean the models "
            "are about as good as each other on this data."
        )


# ---------------------------------------------------------------- dev diagnostics (OOF)
def _oof(run: TrainingRun, r: ModelResult) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.asarray(r.oof_mask, dtype=bool)
    proba = None if r.oof_proba is None else np.asarray(r.oof_proba)[mask]
    return np.asarray(run.y_dev)[mask], np.asarray(r.oof_pred)[mask], proba


def _threshold_tool(state: AppState, run: TrainingRun, r: ModelResult, y, proba) -> None:  # noqa: ANN001
    pos = run.positive_index
    label = run.class_labels[pos]
    st.markdown(f"**Decision threshold for '{label}'**")
    st.caption(
        "A row is predicted as the positive class when its probability is at least the "
        "threshold. Lower it to catch more positives (recall), raise it for fewer false "
        "alarms (precision). Tuned on cross-validation predictions, not the test set."
    )
    current = state.thresholds.get(r.key, 0.5)
    left, right = st.columns([3, 1])
    if right.button("Best F1", key=f"thr_best_{r.key}", width="stretch"):
        current = best_threshold(y, proba[:, pos], pos, "f1")
        set_state(thresholds={**state.thresholds, r.key: current})
        st.rerun()
    chosen = left.slider("Threshold", 0.05, 0.95, float(current), 0.01, key=f"thr_{r.key}")
    if chosen != state.thresholds.get(r.key, 0.5):
        set_state(thresholds={**get_state().thresholds, r.key: chosen})
    curve = threshold_curve(y, proba[:, pos], pos)
    st.plotly_chart(plots.threshold_fig(curve, chosen), width="stretch")
    st.plotly_chart(plots.calibration_fig(y, proba[:, pos], pos, label), width="stretch")


def _target_slice(run: TrainingRun, y, pred):  # noqa: ANN001, ANN202
    """With several targets, diagnostics show one chosen target at a time."""
    if not run.is_multi_output:
        return y, pred, list(run.class_labels or [])
    names = list(run.config.targets)
    name = st.selectbox("Target", names, key="diag_target")
    j = names.index(name)
    return y[:, j], pred[:, j], list(run.target_labels.get(name, ()))


def _dev_details(state: AppState, run: TrainingRun, r: ModelResult) -> None:
    y, pred, proba = _oof(run, r)
    if not len(y):
        st.info("No cross-validation predictions are available for this model.")
        return
    st.caption(
        f"Out-of-fold predictions on {len(y):,} training rows: every row is predicted "
        "by a model that did not see it."
    )
    y, pred, labels = _target_slice(run, y, pred)
    if run.problem_type == ProblemType.REGRESSION:
        left, right = st.columns(2)
        left.plotly_chart(plots.pred_vs_actual_fig(y, pred), width="stretch")
        right.plotly_chart(plots.residuals_fig(y, pred), width="stretch")
        return
    st.plotly_chart(plots.confusion_matrix_fig(y, pred.astype(int), labels), width="stretch")
    if proba is not None:
        left, right = st.columns(2)
        left.plotly_chart(plots.roc_fig(y, proba, labels, run.positive_index), width="stretch")
        right.plotly_chart(plots.pr_fig(y, proba, labels, run.positive_index), width="stretch")
        if run.positive_index is not None:
            _threshold_tool(state, run, r, y, proba)


def _settings(r: ModelResult) -> None:
    if r.params:
        specs = {s.name: s.help for s in param_specs(r.key)}
        st.dataframe(
            pd.DataFrame(
                [
                    {"parameter": k, "value": str(v), "meaning": specs.get(k, "")}
                    for k, v in r.params.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    if r.tuning_history is not None:
        st.markdown("**Hyperparameter search: combinations tried (best first)**")
        st.dataframe(r.tuning_history.astype(str), hide_index=True, width="stretch")
    if r.effective_imbalance:
        st.caption(f"Class imbalance handling actually used: {r.effective_imbalance}.")


def _details(state: AppState) -> None:
    run = state.run
    usable = {r.name: r for r in run.results if r.status in ("ok", "incomplete")}
    names = list(usable)
    choice = st.selectbox(
        "Inspect model", names, index=names.index(run.best().name), key="inspect_model"
    )
    r = usable[choice]
    if r.status == "incomplete":
        st.warning(f"{r.name}: {r.error} It is excluded from best-model selection.")
    _dev_details(state, run, r)
    with st.expander("Settings, tuning and imbalance handling"):
        _settings(r)


# ---------------------------------------------------------------- final test (locked)
def _final_test(state: AppState, name: str) -> None:
    run = state.run
    if not state.test_revealed:
        st.info(
            "The test set was held out from every decision. Look at it once, after you have "
            "chosen your model, so the test score stays an honest estimate of real-world "
            "performance.",
            icon=":material/lock:",
        )
        if st.button("Reveal final test results", type="primary", key="reveal_test"):
            set_state(test_revealed=True)
            st.rerun()
        return
    best = run.best()
    shown = [m for m in TILE_METRICS[run.problem_type] if m in best.test_metrics][:4]
    for col, metric in zip(st.columns(len(shown) or 1), shown, strict=False):
        col.metric(f"Test {metric_label(metric)} ({best.name})", f"{best.test_metrics[metric]:.4f}")
    board = run.leaderboard()
    metric = run.primary_metric
    if f"test_{metric}" in board:
        st.plotly_chart(
            plots.leaderboard_bar(board, metric, metric_label(metric), metric in LOWER_IS_BETTER),
            width="stretch",
        )
    if best.per_target:
        st.markdown(f"**Per target ({best.name}, test set)**")
        table = pd.DataFrame(best.per_target).T.rename_axis("target").reset_index()
        st.dataframe(table, hide_index=True, width="stretch")
    forecast_test_view(run, best)
    sentence = rf_vs_boosting(run)
    if sentence:
        st.info(f"**Random Forest vs gradient boosting:** {sentence}", icon="\U0001f332")
    _test_importance(state, best)


def _test_importance(state: AppState, result: ModelResult) -> None:
    run = state.run
    if result.key not in state.importance:
        if not st.button(
            f"Compute feature importance ({result.name}, test set)", key=f"imp_{result.key}"
        ):
            return
        with st.spinner("Shuffling each column on the test set..."):
            imp = permutation_importance_df(
                result.pipeline,
                run.X_test,
                run.y_test,
                run.problem_type,
                run.primary_metric,
                positive=run.positive_index,
            )
        set_state(importance={**state.importance, result.key: imp})
        st.rerun()
    st.plotly_chart(
        plots.importance_fig(state.importance[result.key], metric_label(run.primary_metric)),
        width="stretch",
    )


def _keep_tab() -> None:
    """Tab selection is tracked in session state, so re-runs keep the open tab."""


def render(state: AppState) -> None:
    st.header("5. Results")
    run = state.run
    if run.cancelled:
        st.warning(
            "Training was cancelled: only the models that finished are shown. "
            "Go back to step 4 to train the rest.",
            icon=":material/stop_circle:",
        )
    for r in run.results:
        if r.status == "failed":
            st.error(f"{r.name} failed: {r.error}")
    if not any(r.status == "ok" for r in run.results):
        st.error("No model was fully validated. Check the messages above and go back.")
        nav_buttons(back=4, next_step=None)
        return
    name = state.card.title if state.card else (state.filename or "uploaded data")
    _cv_tiles(run)
    names = ["Leaderboard (CV)", "Comparison"]
    if has_rf(run):
        names.append("Random Forest dashboard")
    names += ["Model details (CV)", "Final test (held out)"]  # labels never change: a
    # changed label makes Streamlit re-create the tabs and jump back to the first one
    tabs = dict(zip(names, st.tabs(names, key="results_tabs", on_change=_keep_tab), strict=True))
    with tabs["Leaderboard (CV)"]:
        _leaderboard(run, name, state.test_revealed)
    with tabs["Comparison"]:
        _comparison(run)
    if "Random Forest dashboard" in tabs:
        with tabs["Random Forest dashboard"]:
            rf_dashboard(run)
    with tabs["Model details (CV)"]:
        _details(state)
    with tabs[names[-1]]:
        _final_test(get_state(), name)
    nav_buttons(back=4, next_step=6, next_label="Use a model")
