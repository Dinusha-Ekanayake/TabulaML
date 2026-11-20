"""Step 1: choose the kind of problem (classification, regression or forecasting)."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from tabulaml.core.types import ProblemType
from tabulaml.ui.state import AppState, go_to, reset_from


@dataclass(frozen=True)
class Choice:
    key: str
    problem_type: ProblemType
    forecasting: bool
    title: str
    icon: str
    text: str
    examples: str


CHOICES = (
    Choice(
        "classification",
        ProblemType.CLASSIFICATION,
        False,
        "Classification",
        ":material/category:",
        "Predict a category.",
        "Tea lot market catalogue; customer churn (Yes / No). Several yes/no columns at "
        "once = multi-label.",
    ),
    Choice(
        "regression",
        ProblemType.REGRESSION,
        False,
        "Regression",
        ":material/trending_up:",
        "Predict a number from the other columns of the same row.",
        "Tea auction price in LKR; several numbers at once = multi-output regression.",
    ),
    Choice(
        "forecasting",
        ProblemType.REGRESSION,
        True,
        "Forecasting (time series)",
        ":material/timeline:",
        "Predict future values of a series from its own past.",
        "Next week's dengue cases in Colombo from past cases and weather.",
    ),
)
WORKFLOW = (
    "Define the problem",
    "Load data",
    "Check quality + hold out test set",
    "Preprocess (inside the pipeline)",
    "Compare models with CV",
    "Test once",
    "Predict / export",
)


def _workflow_strip() -> None:
    steps = "".join(
        f'<span class="tml-flow-step"><b>{i}</b> {name}</span>'
        for i, name in enumerate(WORKFLOW, start=1)
    )
    st.markdown(f'<div class="tml-flow">{steps}</div>', unsafe_allow_html=True)


def _is_chosen(state: AppState, choice: Choice) -> bool:
    return state.problem_type == choice.problem_type and state.forecasting == choice.forecasting


def render(state: AppState) -> None:
    st.header("1. What do you want to predict?")
    st.write(
        "The problem type decides which targets, models, validation and metrics are offered "
        "next. TabulaML follows the standard workflow used in practice:"
    )
    _workflow_strip()
    columns = st.columns(len(CHOICES))
    for col, choice in zip(columns, CHOICES, strict=True):
        with col.container(border=True, key=f"problem_{choice.key}"):
            st.markdown(f"### {choice.icon} {choice.title}")
            st.write(choice.text)
            st.caption(f"**Examples:** {choice.examples}")
            chosen = _is_chosen(state, choice)
            if st.button(
                "Selected ✓" if chosen else f"Choose {choice.key}",
                key=f"choose_{choice.key}",
                type="primary" if chosen else "secondary",
                width="stretch",
            ):
                if not chosen:
                    reset_from(2, problem_type=choice.problem_type, forecasting=choice.forecasting)
                go_to(2)
