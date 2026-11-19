"""Step 2: upload a dataset or load a Sri Lankan sample, then choose the target."""

from __future__ import annotations

import streamlit as st

from tabulaml.config import MAX_FILE_MB
from tabulaml.data.loader import DataLoadError, list_excel_sheets, read_tabular_file
from tabulaml.data.samples import list_samples, load_sample
from tabulaml.data.validation import suggest_problem_type, validate_target
from tabulaml.ui.components import dataset_card_view, issues_view, nav_buttons
from tabulaml.ui.state import AppState, reset_from

PREVIEW_ROWS = 50
EXCEL = (".xlsx", ".xlsm", ".xls")


def _upload_tab() -> None:
    file = st.file_uploader(
        f"CSV or Excel file (up to {MAX_FILE_MB} MB)",
        type=["csv", "txt", "xlsx", "xls"],
        key="upload_file",
    )
    if file is None:
        return
    sheet = None
    if file.name.lower().endswith(EXCEL):
        try:
            sheet = st.selectbox("Sheet", list_excel_sheets(file), key="upload_sheet")
        except DataLoadError as exc:
            st.error(str(exc))
            return
    if st.button("Load this file", key="load_upload", type="primary"):
        try:
            df = read_tabular_file(file, file.name, sheet)
        except DataLoadError as exc:
            st.error(str(exc))
            return
        reset_from(
            2, raw_df=df, source="upload", filename=file.name, card=None, target=str(df.columns[-1])
        )
        st.rerun()


def _sample_tab(state: AppState) -> None:
    cards = list_samples(state.problem_type)
    if not cards:
        st.info("No bundled sample for this problem type.")
        return
    titles = {c.title: c for c in cards}
    title = st.selectbox("Sample dataset", list(titles), key="sample_choice")
    card = titles[title]
    dataset_card_view(card)
    if st.button("Load this sample", key="load_sample", type="primary"):
        df, card = load_sample(card.key)
        reset_from(
            2, raw_df=df, source="sample", filename=f"{card.key}.csv", card=card, target=card.target
        )
        st.rerun()


def _target_section(state: AppState) -> bool:
    df = state.raw_df
    st.subheader("Loaded data")
    st.caption(
        f"{state.filename}: {len(df):,} rows x {df.shape[1]} columns, "
        f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB in memory"
    )
    st.dataframe(df.head(PREVIEW_ROWS), width="stretch", hide_index=True)
    columns = list(map(str, df.columns))
    index = columns.index(state.target) if state.target in columns else len(columns) - 1
    target = st.selectbox("Target column (what to predict)", columns, index=index, key="target_col")
    if target != state.target:
        reset_from(3, target=target)
        st.rerun()
    issues = validate_target(df, target, state.problem_type, cv_folds=5)
    suggested = suggest_problem_type(df[target])
    if suggested != state.problem_type:
        st.info(
            f"'{target}' looks more like a {suggested.value} target. You can go back to "
            "step 1 to change the problem type.",
            icon="ℹ️",
        )
    issues_view(issues)
    return not any(i.level == "error" for i in issues)


def render(state: AppState) -> None:
    st.header("2. Choose your data")
    upload, sample = st.tabs(["Upload my dataset", "Use a Sri Lankan sample dataset"])
    with upload:
        _upload_tab()
    with sample:
        _sample_tab(state)
    ok = _target_section(state) if state.raw_df is not None else False
    nav_buttons(back=1, next_step=3, next_enabled=ok)
