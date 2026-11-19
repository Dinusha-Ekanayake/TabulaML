"""TabulaML Streamlit entry point: `streamlit run app.py`."""

from __future__ import annotations

import logging

import streamlit as st

from tabulaml.ui.components import inject_css, sidebar_stepper
from tabulaml.ui.state import can_enter, get_state, set_state
from tabulaml.ui.steps import (
    s1_problem,
    s2_data,
    s3_preprocess,
    s4_models,
    s5_results,
    s6_predict,
)

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="TabulaML", page_icon="\U0001f4ca", layout="wide")
inject_css()

PAGES = {
    1: s1_problem.render,
    2: s2_data.render,
    3: s3_preprocess.render,
    4: s4_models.render,
    5: s5_results.render,
    6: s6_predict.render,
}

state = get_state()
if not can_enter(state, state.step):  # e.g. stale step after a reset
    state = set_state(step=1)
sidebar_stepper(state)
try:
    PAGES[state.step](state)
except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
    logging.getLogger("tabulaml").exception("UI step failed")
    st.error(f"Something went wrong: {type(exc).__name__}: {exc}")
    st.caption("Try going back a step, or use 'Start over' in the sidebar.")
