"""Step 4 helper: hyperparameters per selected model.

Modes: Default, Manual, Random search, Grid search. No ML logic here: the trainer runs the
searches (``training.tuning``); this module only collects the user's choices.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabulaml.core.types import ProblemType
from tabulaml.models.params import (
    ParamSpec,
    grid_size,
    grid_values,
    model_defaults,
    param_specs,
)
from tabulaml.models.registry import get_model
from tabulaml.training.tuning import MAX_GRID

MODES = ("Default", "Manual", "Random search", "Grid search")
MODE_HELP = {
    "Default": "The model's standard settings. Fastest, and a fair baseline.",
    "Manual": "Set each value yourself. Every setting is explained.",
    "Random search": "Tries N random combinations from sensible ranges with the same "
    "cross-validation on the training rows, then keeps the best. Cheap when there are "
    "many settings.",
    "Grid search": "Tries EVERY combination of a few values per chosen setting, with the "
    "same cross-validation on the training rows. Exhaustive and easy to explain, but the "
    "number of combinations multiplies quickly.",
}
FAST_TREES = 100
GRID_DEFAULT_SETTINGS = 2


def _label(v: Any) -> str:
    return "None (unlimited)" if v is None else str(v)


def _widget(key: str, spec: ParamSpec, default: Any) -> Any:
    wkey, label = f"p_{key}_{spec.name}", spec.name.replace("_", " ")
    if spec.kind == "choice":
        options = list(spec.choices)
        index = options.index(default) if default in options else 0
        return st.selectbox(
            label, options, index=index, key=wkey, help=spec.help, format_func=_label
        )
    if spec.kind == "int":
        value = int(min(max(int(default), spec.low), spec.high))
        return int(
            st.number_input(
                label, int(spec.low), int(spec.high), value, step=1, key=wkey, help=spec.help
            )
        )
    value = float(min(max(float(default), spec.low), spec.high))
    return float(
        st.number_input(
            label,
            float(spec.low),
            float(spec.high),
            value,
            key=wkey,
            help=spec.help,
            format="%.4g",
        )
    )


def _manual(key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    specs = param_specs(key)
    cols = st.columns(min(3, len(specs)))
    out = {}
    for i, spec in enumerate(specs):
        with cols[i % len(cols)]:
            out[spec.name] = _widget(key, spec, defaults[spec.name])
    return out


def _grid(key: str, defaults: dict[str, Any], folds: int) -> dict[str, tuple[Any, ...]] | None:
    specs = [s for s in param_specs(key) if s.tunable]
    names = [s.name for s in specs]
    chosen = st.multiselect(
        "Settings to search",
        names,
        default=names[:GRID_DEFAULT_SETTINGS],
        key=f"grid_params_{key}",
        help="Each extra setting multiplies the number of combinations.",
    )
    n = st.select_slider(
        "Values per number setting",
        options=[2, 3, 4, 5],
        value=3,
        key=f"grid_n_{key}",
        help="Values are centred on the model's default. Choice settings use every option.",
    )
    grid = {s.name: tuple(grid_values(s, n, defaults[s.name])) for s in specs if s.name in chosen}
    if not grid:
        st.caption("Choose at least one setting to search.")
        return None
    rows = [{"setting": k, "values tried": ", ".join(map(_label, v))} for k, v in grid.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    size = grid_size({k: list(v) for k, v in grid.items()})
    if size > MAX_GRID:
        st.error(
            f"{size} combinations is too many (limit {MAX_GRID}). Search fewer settings or "
            "use fewer values.",
            icon=":material/block:",
        )
        return None
    st.caption(
        f":material/grid_on: **{size} combinations** x {folds} folds = **{size * folds} fits** "
        "for this model."
    )
    return grid


def _model_panel(
    key: str, problem_type: ProblemType, folds: int
) -> tuple[dict[str, Any] | None, int, dict[str, tuple[Any, ...]] | None]:
    """(manual params, random-search size, grid) chosen for one model."""
    if not param_specs(key):
        st.caption("This ensemble is configured through its base models.")
        return None, 0, None
    with st.container(border=True, key=f"params_card_{key}"):
        mode = st.radio("How to set the hyperparameters", MODES, horizontal=True, key=f"mode_{key}")
        st.caption(MODE_HELP[mode])
        defaults = model_defaults(key, problem_type)
        if mode == "Manual":
            return _manual(key, defaults), 0, None
        if mode == "Random search":
            n = st.slider(
                "Combinations to try",
                5,
                50,
                15,
                key=f"tune_{key}",
                help="More combinations can find better settings but take longer.",
            )
            st.caption(f":material/casino: {n} combinations x {folds} folds = {n * folds} fits.")
            return None, int(n), None
        if mode == "Grid search":
            return None, 0, _grid(key, defaults, folds)
    return None, 0, None


def apply_preset(name: str, selected: list[str]) -> None:
    """Fast = fewer trees; Balanced = defaults; Thorough = random search, 20 combinations."""
    for key in selected:
        if not param_specs(key):
            continue
        if name == "Thorough":
            st.session_state[f"mode_{key}"] = "Random search"
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


def settings_panel(
    selected: list[str], problem_type: ProblemType, folds: int
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, dict[str, tuple[Any, ...]]]]:
    """Render settings for the selected models; return (manual, random sizes, grids)."""
    manual: dict[str, dict[str, Any]] = {}
    tuning: dict[str, int] = {}
    grids: dict[str, dict[str, tuple[Any, ...]]] = {}
    tunable = [k for k in selected if param_specs(k)]
    if not tunable:
        return manual, tuning, grids
    with st.expander(":material/tune: Model settings (hyperparameters)", expanded=False):
        cols = st.columns([1, 1, 1, 3], vertical_alignment="center")
        for col, preset in zip(cols[:3], ("Fast", "Balanced", "Thorough"), strict=True):
            if col.button(preset, key=f"preset_{preset}", width="stretch"):
                apply_preset(preset, selected)
                st.rerun()
        cols[3].caption("Fast: fewer trees. Balanced: defaults. Thorough: random search.")
        for tab, key in zip(st.tabs([get_model(k).name for k in tunable]), tunable, strict=True):
            with tab:
                params, budget, grid = _model_panel(key, problem_type, folds)
                if params:
                    manual[key] = params
                if budget:
                    tuning[key] = budget
                if grid:
                    grids[key] = grid
    return manual, tuning, grids
