"""Step 5 additions for forecasting: naive-baseline comparison and the test period chart."""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import ModelResult, TrainingRun
from tabulaml.evaluation.forecast_plots import test_period_fig


def _naive(run: TrainingRun) -> ModelResult | None:
    try:
        naive = run.result_for("naive")
    except KeyError:
        return None
    return naive if naive.status == "ok" else None


def naive_sentence(run: TrainingRun, best: ModelResult) -> str | None:
    """'Beats the naive baseline by X % RMSE' (or an honest warning that it does not)."""
    naive = _naive(run)
    if naive is None:
        return None
    if best.key == "naive":
        return (
            "The naive baseline (last known value) is the best model: the other models did "
            "not learn anything useful beyond the most recent value."
        )
    ours, base = best.test_metrics.get("rmse"), naive.test_metrics.get("rmse")
    if ours is None or not base:
        return None
    gain = (base - ours) / base
    if gain > 0:
        return (
            f"{best.name} beats the naive baseline (last known value) by {gain:.1%} RMSE "
            f"on the held-out test period ({ours:,.2f} vs {base:,.2f})."
        )
    return (
        f"{best.name} does NOT beat the naive baseline on the test period "
        f"({ours:,.2f} vs {base:,.2f} RMSE): the features add little beyond the last value."
    )


def forecast_test_view(run: TrainingRun, best: ModelResult) -> None:
    if run.config.forecast is None or run.test_order is None:
        return
    sentence = naive_sentence(run, best)
    if sentence and best.key == "naive":
        st.warning(sentence)
    elif sentence:
        st.info(sentence, icon=":material/timeline:")
    naive = _naive(run)
    st.plotly_chart(
        test_period_fig(
            run.test_order,
            run.y_test,
            best.y_pred,
            best.name,
            naive.y_pred if naive is not None and naive.key != best.key else None,
        ),
        width="stretch",
    )
