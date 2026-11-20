"""Step 3 sub-section: choose how missing values are filled, per column."""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import FeatureType
from tabulaml.ui.imputation import imputable_columns, impute_label, impute_options, with_imputation
from tabulaml.ui.state import AppState, reset_from

_BULK_TYPES = (FeatureType.NUMERIC, FeatureType.CATEGORICAL)


def _missing_ratios(state: AppState) -> dict[str, float]:
    if state.schema is None:
        return {}
    return {c.name: c.missing_ratio for c in state.schema.columns}


def _bulk_controls(state: AppState, names: dict[FeatureType, list[str]]) -> None:
    cols = st.columns(len(_BULK_TYPES))
    for slot, ftype in zip(cols, _BULK_TYPES, strict=True):
        if not names[ftype]:
            continue
        options = impute_options(ftype)
        method = slot.selectbox(
            f"All {ftype.value} columns",
            options,
            format_func=lambda m, t=ftype: impute_label(m, t),
            key=f"imp_bulk_{ftype.value}",
        )
        label = f"Apply to {len(names[ftype])} {ftype.value} columns"
        if slot.button(label, key=f"imp_apply_{ftype.value}"):
            new = with_imputation(state.config, {n: method for n in names[ftype]})
            for name in names[ftype]:  # let the per-column boxes show the new choice
                st.session_state.pop(f"imp_{name}", None)
            if new is not state.config:
                reset_from(4, config=new)
                st.rerun()


def missing_values_section(state: AppState) -> None:
    """Per-column imputation choices. Every method is fitted on training folds only."""
    st.markdown("##### Missing values (imputation)")
    st.caption(
        "Choose how empty cells are filled. The fill value is learned from the training "
        "rows only, and is also used for empty cells in files you predict on later."
    )
    columns = imputable_columns(state.config)
    if not columns:
        st.info("No numeric or category columns are included, so there is nothing to choose.")
        return
    ratios = _missing_ratios(state)
    names = {t: [c.name for c in columns if c.feature_type == t] for t in _BULK_TYPES}
    _bulk_controls(state, names)
    with_missing = [c for c in columns if ratios.get(c.name, 0.0) > 0]
    show_all = st.toggle(
        "Show all columns",
        value=not with_missing,
        key="imp_show_all",
        help="By default only columns that have empty cells in this dataset are listed.",
    )
    shown = columns if show_all else with_missing
    if not shown:
        st.caption(
            "No included column has empty cells. Turn on *Show all columns* to set methods anyway."
        )
        return
    choices: dict[str, str] = {}
    for col in shown:
        left, right = st.columns([2, 3], vertical_alignment="center")
        left.markdown(
            f"`{col.name}` · {col.feature_type.value} · {ratios.get(col.name, 0.0):.1%} missing"
        )
        options = impute_options(col.feature_type)
        choices[col.name] = right.selectbox(
            f"Method for {col.name}",
            options,
            index=options.index(col.impute) if col.impute in options else 0,
            format_func=lambda m, t=col.feature_type: impute_label(m, t),
            key=f"imp_{col.name}",
            label_visibility="collapsed",
        )
    new = with_imputation(state.config, choices)
    if new is not state.config:
        reset_from(4, config=new)
        st.rerun()
