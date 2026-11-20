"""Step 3 card: which features the models will use, which are excluded and why, and how
strongly each used feature relates to the target (training rows only)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tabulaml.core.types import FeatureType
from tabulaml.evaluation.feature_relevance import feature_relevance, relevance_fig
from tabulaml.ui.state import AppState

CACHE_KEY = "tml_relevance_cache"


def _used_and_excluded(state: AppState) -> tuple[list, list]:
    cfg = state.config
    roles = {*cfg.targets, cfg.group_column, cfg.order_column}
    used = [c for c in cfg.columns if c.include and c.name not in roles]
    excluded = [c for c in cfg.columns if not c.include and c.name not in cfg.targets]
    return used, excluded


def _relevance(state: AppState, names: list[str]) -> pd.DataFrame:
    """Mutual information / correlation on the training rows, cached per analysis + feature set."""
    auto = state.auto
    clean = auto.clean_all
    available = [n for n in names if n in clean.columns]
    cache_id = (id(auto), tuple(available), state.target)
    cache = st.session_state.get(CACHE_KEY)
    if cache and cache[0] == cache_id:
        return cache[1]
    dev = (
        clean.loc[clean.index.intersection(auto.dev_index)] if auto.dev_index is not None else clean
    )
    types = {c.name: c.inferred_type for c in state.schema.columns}
    for c in state.config.columns:  # the user's type choices win over the detected types
        types[c.name] = c.feature_type
    table = feature_relevance(dev, available, types, state.target, state.problem_type)
    st.session_state[CACHE_KEY] = (cache_id, table)
    return table


def _kpis(used: list, excluded: list) -> None:
    numeric = sum(c.feature_type == FeatureType.NUMERIC for c in used)
    categorical = sum(c.feature_type == FeatureType.CATEGORICAL for c in used)
    cells = [
        ("Features used", len(used)),
        ("Excluded", len(excluded)),
        ("Number features", numeric),
        ("Category features", categorical),
    ]
    for col, (label, value) in zip(st.columns(len(cells)), cells, strict=True):
        with col.container(border=True, key=f"feat_kpi_{label.split()[0]}"):
            st.metric(label, value)


def features_card(state: AppState) -> None:
    used, excluded = _used_and_excluded(state)
    with st.container(border=True, key="features_card"):
        st.markdown("##### Features the model will use")
        _kpis(used, excluded)
        if excluded:
            with st.expander(f"Excluded columns ({len(excluded)}) and why"):
                st.dataframe(
                    pd.DataFrame([{"column": c.name, "reason": c.reason} for c in excluded]),
                    hide_index=True,
                    width="stretch",
                )
        if not used or state.auto is None or state.auto.clean_all is None:
            return
        table = _relevance(state, [c.name for c in used])
        if table.empty:
            return
        left, right = st.columns([3, 2])
        left.plotly_chart(relevance_fig(table, state.target), width="stretch")
        shown = table.rename(
            columns={
                "feature": "Feature",
                "type": "Type",
                "mutual_information": "Mutual information",
                "correlation": "Correlation with target",
            }
        )
        right.dataframe(shown.round(3), hide_index=True, width="stretch", height=360)
        st.caption(
            "Computed on the training rows only. Mutual information measures any kind of link "
            "(0 = unrelated); correlation (number features) measures a straight-line link from "
            "-1 to +1. A low score does not always mean a feature is useless, because models "
            "can combine features."
        )
