"""Step 3 tab: explore the data before preprocessing (distributions and relationships).

Column distributions use all rows (no target involved). Relationships with the target use
the TRAINING rows only, so the held-out test set is never looked at.
"""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import FeatureType
from tabulaml.evaluation.eda_plots import (
    column_summary,
    correlation_fig,
    distribution_fig,
    strongest_pairs,
    target_relation_fig,
)
from tabulaml.ui.state import AppState

STAT_COLUMNS = 4
SKIP_TYPES = {FeatureType.IDENTIFIER, FeatureType.CONSTANT}


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _stat_cards(summary: dict[str, object]) -> None:
    items = list(summary.items())
    for start in range(0, len(items), STAT_COLUMNS):
        cols = st.columns(STAT_COLUMNS)
        for col, (label, value) in zip(cols, items[start : start + STAT_COLUMNS], strict=False):
            with col.container(border=True, key=f"eda_kpi_{label.split()[0]}_{start}"):
                st.metric(label[:1].upper() + label[1:], _fmt(value))


def _column_view(state: AppState, clean, types: dict[str, FeatureType]) -> None:  # noqa: ANN001
    targets = set(state.config.targets)
    options = [c for c in clean.columns if c not in targets and types.get(c) not in SKIP_TYPES]
    options = list(state.config.targets) + options  # the target(s) first
    name = st.selectbox(
        "Column to explore",
        options,
        key="eda_column",
        help="The target is listed first. ID-like and constant columns are left out.",
    )
    ftype = types.get(name, FeatureType.CATEGORICAL)
    st.caption(f"Detected type: **{ftype.value}** (you can change it in Preprocessing → Custom).")
    _stat_cards(column_summary(clean[name], ftype))
    left, right = st.columns(2)
    left.plotly_chart(distribution_fig(clean[name], ftype, name), width="stretch")
    if name in targets or state.auto.dev_index is None:
        right.info("Pick a feature column to see how it relates to the target.")
        return
    dev = clean.loc[clean.index.intersection(state.auto.dev_index)]
    right.plotly_chart(
        target_relation_fig(
            dev[name], dev[state.target], ftype, state.problem_type, name, state.target
        ),
        width="stretch",
    )


def _correlations(clean, types: dict[str, FeatureType]) -> None:  # noqa: ANN001
    numbers = [c for c in clean.columns if types.get(c) == FeatureType.NUMERIC]
    fig = correlation_fig(clean[numbers]) if numbers else None
    if fig is None:
        st.caption("Fewer than two number columns, so there are no correlations to show.")
        return
    st.plotly_chart(fig, width="stretch")
    pairs = strongest_pairs(clean[numbers])
    if pairs:
        lines = [f"- `{a}` and `{b}`: r = {r:+.2f}" for a, b, r in pairs]
        st.markdown("**Most strongly related pairs**\n" + "\n".join(lines))
        st.caption(
            "Blue = rise together, red = one rises as the other falls, grey = unrelated. "
            "Very strong pairs carry almost the same information; tree models cope well, "
            "linear models can become unstable."
        )


def explore_tab(state: AppState) -> None:
    clean = state.auto.clean_all if state.auto is not None else None
    if clean is None or state.schema is None:
        st.info("The data has not been analysed yet.")
        return
    st.caption(
        "Understand the data before choosing how to prepare it. Distributions use every "
        "row; relationships with the target use the training rows only, so the test set "
        "stays unseen."
    )
    types = {c.name: c.inferred_type for c in state.schema.columns}
    column, correlations = st.tabs(["One column", "Correlations"])
    with column:
        _column_view(state, clean, types)
    with correlations:
        _correlations(clean, types)
