"""Step 3 card: what to do with duplicate rows, rows with missing values and mostly-empty
columns. Row choices change which rows exist, so they re-create the held-out test set."""

from __future__ import annotations

import streamlit as st

from tabulaml.training.prepare import incomplete_rows
from tabulaml.ui.imputation import exclude_mostly_missing
from tabulaml.ui.state import AppState, reset_from

ROW_CHOICES = {
    "impute": "Keep them and fill the gaps (imputation, recommended)",
    "drop": "Drop rows that have any missing value",
}


def _restart() -> None:
    reset_from(3)  # a new set of rows means a new holdout and new recommendations


def row_settings() -> tuple[bool, str]:
    """(drop duplicates?, what to do with incomplete rows) from the widgets' state."""
    return (
        bool(st.session_state.get("rs_dups", True)),
        str(st.session_state.get("rs_missing_rows", "impute")),
    )


def _rows(state: AppState) -> None:
    n_dup = state.schema.n_duplicate_rows if state.schema is not None else 0
    st.toggle(
        f"Drop exact duplicate rows ({n_dup:,} found)",
        value=True,
        key="rs_dups",
        on_change=_restart,
        help="Identical rows add no information and can leak between training and test. "
        "Keep them only if repeats are genuine, separate events.",
    )
    if state.forecasting:
        st.caption("Forecasting keeps every period, so missing values are always filled.")
        return
    n_missing = int(incomplete_rows(state.raw_df, state.config).sum())
    st.radio(
        f"Rows with missing values ({n_missing:,} of {len(state.raw_df):,} rows)",
        list(ROW_CHOICES),
        format_func=ROW_CHOICES.get,
        key="rs_missing_rows",
        on_change=_restart,
        help="Dropping is simple but loses data and can bias the sample when values are "
        "not missing at random. Filling keeps every row. New data at prediction time is "
        "always filled.",
    )
    if st.session_state.get("rs_missing_rows") == "drop" and n_missing > len(state.raw_df) / 2:
        st.warning(
            "More than half of the rows would be dropped. Consider excluding the "
            "mostly-empty columns below and filling the rest instead.",
            icon=":material/warning:",
        )


def _columns(state: AppState) -> None:
    if state.schema is None:
        return
    missing = {c.name: c.missing_ratio for c in state.schema.columns}
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    percent = left.slider(
        "Exclude columns with more missing values than",
        5,
        95,
        60,
        5,
        format="%d%%",
        key="rs_col_limit",
        help="Columns that are mostly empty carry little information. Excluded columns "
        "can be re-included in the per-column table.",
    )
    if right.button("Apply", key="rs_col_apply", width="stretch"):
        new, names = exclude_mostly_missing(state.config, missing, percent / 100)
        if names:
            reset_from(4, config=new)
            st.toast(f"Excluded {len(names)} column(s): {', '.join(names)}")
            st.rerun()
        else:
            st.toast("No included column is above that limit.")


def rows_card(state: AppState) -> None:
    """Duplicates, incomplete rows and mostly-empty columns, as one card."""
    with st.container(border=True, key="rows_card"):
        st.markdown("##### Rows and missing values")
        st.caption(
            "Choose how duplicates and gaps are handled. Changing a row option re-creates "
            "the held-out test set, because the set of rows changes."
        )
        _rows(state)
        _columns(state)
