"""Step 4 helper: hyperparameters per selected model (Default / Manual / Auto-tune)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabulaml.models.params import ParamSpec, param_specs
from tabulaml.models.registry import get_model

MODES = ("Default", "Manual", "Auto-tune")
MODE_HELP = {
    "Default": "Sensible library defaults. Fastest and a fair baseline.",
    "Manual": "Set the values yourself. Each one is explained.",
    "Auto-tune": "Random search: tries N combinations with the same cross-validation on the "
    "training rows only, then keeps the best. The test set is never used.",
}
FAST_TREES = 100


def _widget(key: str, spec: ParamSpec) -> Any:
    wkey = f"p_{key}_{spec.name}"
    label = spec.name.replace("_", " ")
    if spec.kind == "choice":
        options = list(spec.choices)
        default = options.index(spec.default) if spec.default in options else 0
        return st.selectbox(
            label,
            options,
            index=default,
            key=wkey,
            help=spec.help,
            format_func=lambda v: "None (unlimited)" if v is None else str(v),
        )
    if spec.kind == "int":
        return int(
            st.number_input(
                label,
                int(spec.low),
                int(spec.high),
                int(spec.default),
                step=1,
                key=wkey,
                help=spec.help,
            )
        )
    return float(
        st.number_input(
            label,
            float(spec.low),
            float(spec.high),
            float(spec.default),
            key=wkey,
            help=spec.help,
            format="%.4g",
        )
    )


def _model_panel(key: str) -> tuple[dict[str, Any] | None, int]:
    specs = param_specs(key)
    if not specs:
        st.caption("This ensemble is configured through its base models.")
        return None, 0
    mode = st.radio("Settings", MODES, horizontal=True, key=f"mode_{key}")
    st.caption(MODE_HELP[mode])
    if mode == "Manual":
        cols = st.columns(min(3, len(specs)))
        return {s.name: _widget_in(cols[i % len(cols)], key, s) for i, s in enumerate(specs)}, 0
    if mode == "Auto-tune":
        n = st.slider(
            "Combinations to try",
            5,
            50,
            15,
            key=f"tune_{key}",
            help="More combinations can find better settings but take longer.",
        )
        return None, int(n)
    return None, 0


def _widget_in(col, key: str, spec: ParamSpec) -> Any:  # noqa: ANN001 - Streamlit column
    with col:
        return _widget(key, spec)


def apply_preset(name: str, selected: list[str]) -> None:
    """Fast = fewer trees; Balanced = defaults; Thorough = auto-tune 20 combinations."""
    for key in selected:
        if not param_specs(key):
            continue
        if name == "Thorough":
            st.session_state[f"mode_{key}"] = "Auto-tune"
            st.session_state[f"tune_{key}"] = 20
        elif name == "Fast" and any(
            s.name in ("n_estimators", "iterations") for s in param_specs(key)
        ):
            st.session_state[f"mode_{key}"] = "Manual"
            for s in param_specs(key):
                if s.name in ("n_estimators", "iterations"):
                    st.session_state[f"p_{key}_{s.name}"] = max(int(s.low), FAST_TREES)
        else:
            st.session_state[f"mode_{key}"] = "Default"


def settings_panel(selected: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Render settings for the selected models; return (manual params, tuning budgets)."""
    manual: dict[str, dict[str, Any]] = {}
    tuning: dict[str, int] = {}
    tunable = [k for k in selected if param_specs(k)]
    if not tunable:
        return manual, tuning
    with st.expander(":material/tune: Model settings (hyperparameters)", expanded=False):
        cols = st.columns([1, 1, 1, 3])
        for col, preset in zip(cols[:3], ("Fast", "Balanced", "Thorough"), strict=True):
            if col.button(preset, key=f"preset_{preset}", width="stretch"):
                apply_preset(preset, selected)
                st.rerun()
        cols[3].caption("Fast: fewer trees. Balanced: defaults. Thorough: auto-tune each model.")
        for tab, key in zip(st.tabs([get_model(k).name for k in tunable]), tunable, strict=True):
            with tab:
                params, budget = _model_panel(key)
                if params:
                    manual[key] = params
                if budget:
                    tuning[key] = budget
    return manual, tuning
