"""Wizard state: one immutable AppState in st.session_state, replaced on every change.

Changing an earlier choice invalidates everything after it (``reset_from``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import streamlit as st

from tabulaml.core.types import PreprocessingConfig, ProblemType, TrainingRun, ValidationIssue

STATE_KEY = "tml_state"
STEPS = (
    (1, "Problem type"),
    (2, "Data"),
    (3, "Preprocessing"),
    (4, "Models"),
    (5, "Results"),
    (6, "Predict"),
)

# Which fields belong to (are produced by) each step.
_STEP_FIELDS = {
    2: ("raw_df", "source", "card", "target", "filename"),
    3: ("clean_df", "schema", "leakage", "supported", "config"),
    4: ("selected_models", "ensemble_bases", "primary_metric", "run"),
    5: ("importance",),
    6: ("input_spec",),
}


@dataclass(frozen=True)
class AppState:
    step: int = 1
    problem_type: ProblemType | None = None
    raw_df: Any = None
    source: str | None = None
    filename: str | None = None
    card: Any = None
    target: str | None = None
    clean_df: Any = None
    schema: Any = None
    leakage: tuple[ValidationIssue, ...] = ()
    supported: tuple[ValidationIssue, ...] = ()
    config: PreprocessingConfig | None = None
    selected_models: tuple[str, ...] = ()
    ensemble_bases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    primary_metric: str | None = None
    run: TrainingRun | None = None
    importance: dict[str, Any] = field(default_factory=dict)
    input_spec: tuple[dict, ...] = ()


def get_state() -> AppState:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = AppState()
    return st.session_state[STATE_KEY]


def set_state(**changes: Any) -> AppState:
    new = replace(get_state(), **changes)
    st.session_state[STATE_KEY] = new
    return new


def cleared_fields(from_step: int) -> dict[str, Any]:
    """Default values for every field produced at ``from_step`` or later."""
    defaults = {f.name: f for f in fields(AppState)}
    out: dict[str, Any] = {}
    for step, names in _STEP_FIELDS.items():
        if step < from_step:
            continue
        for name in names:
            f = defaults[name]
            out[name] = f.default_factory() if callable(f.default_factory) else f.default
    return out


def reset_from(from_step: int, **changes: Any) -> AppState:
    """Clear everything produced at ``from_step`` or later, then apply ``changes``."""
    return set_state(**{**cleared_fields(from_step), **changes})


def can_enter(state: AppState, step: int) -> bool:
    checks = {
        1: True,
        2: state.problem_type is not None,
        3: state.raw_df is not None and state.target is not None,
        4: state.config is not None,
        5: state.run is not None,
        6: state.run is not None and any(r.status == "ok" for r in state.run.results),
    }
    return bool(checks.get(step, False))


def go_to(step: int) -> None:
    if can_enter(get_state(), step):
        set_state(step=step)
        st.rerun()
