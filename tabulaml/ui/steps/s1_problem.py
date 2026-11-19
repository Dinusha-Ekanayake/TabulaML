"""Step 1: choose classification or regression."""

from __future__ import annotations

import streamlit as st

from tabulaml.core.types import ProblemType
from tabulaml.ui.state import AppState, go_to, reset_from

CHOICES = (
    (
        ProblemType.CLASSIFICATION,
        "Classification",
        "Predict a category, for example whether a customer will churn (Yes / No) "
        "or which risk group a patient belongs to.",
    ),
    (
        ProblemType.REGRESSION,
        "Regression",
        "Predict a number, for example weekly dengue cases, length of hospital stay "
        "or a price in LKR.",
    ),
)


def render(state: AppState) -> None:
    st.header("1. What do you want to predict?")
    st.write(
        "The problem type decides which target columns, models and evaluation metrics are "
        "offered in the next steps."
    )
    columns = st.columns(2)
    for col, (ptype, title, text) in zip(columns, CHOICES, strict=True):
        with col.container(border=True):
            st.subheader(title)
            st.write(text)
            chosen = state.problem_type == ptype
            if st.button(
                "Selected ✓" if chosen else f"Choose {title.lower()}",
                key=f"choose_{ptype.value}",
                type="primary" if chosen else "secondary",
                width="stretch",
            ):
                if not chosen:
                    reset_from(2, problem_type=ptype)
                go_to(2)
