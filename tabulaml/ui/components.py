"""Shared Streamlit widgets."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from tabulaml.core.types import ValidationIssue
from tabulaml.ui.state import STEPS, AppState, can_enter, go_to

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.tml-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
            font-weight:600; border:1px solid currentColor; margin-right:6px;}
.tml-real {color:#0c7a0c;} .tml-synthetic {color:#b26a00;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_stepper(state: AppState) -> None:
    with st.sidebar:
        st.markdown("### TabulaML")
        st.caption("Interactive ML workbench")
        for number, label in STEPS:
            enabled = can_enter(state, number)
            prefix = "▶ " if number == state.step else ("✓ " if number < state.step else "")
            if st.button(
                f"{prefix}{number}. {label}",
                key=f"nav_{number}",
                disabled=not enabled,
                width="stretch",
                type="primary" if number == state.step else "secondary",
            ):
                go_to(number)
        st.divider()
        if st.button("Start over", key="nav_reset", width="stretch"):
            st.session_state.clear()
            st.rerun()


def issues_view(issues: Sequence[ValidationIssue]) -> None:
    for issue in issues:
        if issue.level == "error":
            st.error(issue.message, icon="⛔")
        elif issue.level == "warning":
            st.warning(issue.message, icon="⚠️")
        else:
            st.info(issue.message, icon="ℹ️")


def provenance_badge(is_synthetic: bool) -> None:
    if is_synthetic:
        html = '<span class="tml-badge tml-synthetic">Synthetic Sri Lankan-context data</span>'
    else:
        html = '<span class="tml-badge tml-real">Real public Sri Lankan data</span>'
    st.markdown(html, unsafe_allow_html=True)


def dataset_card_view(card) -> None:  # noqa: ANN001 - DatasetCard
    provenance_badge(card.is_synthetic)
    st.markdown(f"**{card.title}**  \n{card.description}")
    with st.expander("Dataset card: source, licence, features, limitations"):
        st.markdown(
            f"- **Provenance:** {card.provenance}\n"
            f"- **Context:** {card.context}\n"
            f"- **Licence:** {card.license}\n"
            f"- **Period:** {card.collection_period}\n"
            f"- **Target:** `{card.target}`: {card.target_definition}\n"
            f"- **Anonymisation:** {card.anonymisation}"
        )
        st.markdown("**Features**")
        st.dataframe(
            [{"column": k, "description": v} for k, v in card.features.items()],
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Known limitations**\n" + "\n".join(f"- {x}" for x in card.known_limitations))


def nav_buttons(
    back: int | None, next_step: int | None, next_enabled: bool = True, next_label: str = "Next"
) -> None:
    left, _, right = st.columns([1, 4, 1])
    if back is not None and left.button("← Back", key=f"back_{back}", width="stretch"):
        go_to(back)
    if next_step is not None and right.button(
        f"{next_label} →",
        key=f"next_{next_step}",
        type="primary",
        disabled=not next_enabled,
        width="stretch",
    ):
        go_to(next_step)
