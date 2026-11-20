"""Step 6 (forecasting mode): forecast the next periods from the history."""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import ModelResult, TrainingRun
from tabulaml.data.loader import DataLoadError, read_tabular_file
from tabulaml.evaluation.forecast_plots import future_fig
from tabulaml.forecasting.future import forecast_future
from tabulaml.training.splits import order_key
from tabulaml.ui.state import AppState

UPLOAD_TYPES = ["csv", "txt", "xlsx", "xls", "json", "parquet"]


def _history(state: AppState):  # noqa: ANN202 - DataFrame or None
    source = st.radio(
        "History to forecast from",
        ["The data used for training", "Upload newer history"],
        horizontal=True,
        key="fc_source",
    )
    if source.startswith("The data"):
        return state.raw_df
    file = st.file_uploader(
        "History file (same columns, including the target)", type=UPLOAD_TYPES, key="fc_file"
    )
    if file is None:
        return None
    try:
        return read_tabular_file(file, file.name)
    except DataLoadError as exc:
        st.error(str(exc))
        return None


def forecast_tab(state: AppState, run: TrainingRun, result: ModelResult) -> None:
    fc = run.config.forecast
    st.caption(
        f"The chosen model is refitted on ALL the history (its settings were already tested), "
        f"then forecasts the next {fc.horizon} period(s) for every series."
    )
    history = _history(state)
    if history is None:
        return
    missing = [c for c in (run.config.target, run.config.order_column) if c not in history]
    if missing:
        st.error(f"The history needs these columns: {', '.join(missing)}.")
        return
    if not st.button(f"Forecast the next {fc.horizon} period(s)", type="primary", key="fc_go"):
        return
    try:
        with st.spinner("Refitting on all history and forecasting..."):
            table, _ = forecast_future(result.pipeline, history, run.config, run.feature_columns)
    except ValueError as exc:
        st.error(f"Could not forecast: {exc}")
        return
    st.dataframe(table, hide_index=True, width="stretch")
    if not fc.series_column:
        shown = history.assign(
            **{run.config.order_column: order_key(history[run.config.order_column])}
        )
        st.plotly_chart(
            future_fig(shown, table, run.config.order_column, run.config.target),
            width="stretch",
        )
    st.download_button(
        "Download forecast (CSV)",
        table.to_csv(index=False).encode(),
        "forecast.csv",
        "text/csv",
        key="dl_forecast",
    )
