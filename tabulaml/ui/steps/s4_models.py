"""Step 4: model selection dashboard and the Train & Predict action."""

from __future__ import annotations

from dataclasses import replace
from itertools import groupby

import streamlit as st

from tabulaml.core.types import ModelProfile
from tabulaml.evaluation.metrics import default_primary_metric, scoring_map
from tabulaml.models.params import grid_size
from tabulaml.models.registry import (
    ENSEMBLE_KEYS,
    FORECAST_KEYS,
    RECOMMENDED,
    RECOMMENDED_FORECAST,
    list_models,
)
from tabulaml.training.splits import describe
from tabulaml.training.trainer import prepare_data
from tabulaml.ui.components import nav_buttons
from tabulaml.ui.helpers import metric_label
from tabulaml.ui.model_meta import GROUP_ICONS, SPEED_CLASS, meta_for
from tabulaml.ui.state import AppState, get_state, set_state
from tabulaml.ui.steps.s4_params import settings_panel
from tabulaml.ui.steps.s4_training import (
    estimate_card,
    estimate_for,
    is_training,
    progress_panel,
    show_notice,
    start_training,
)

PROFILE_BADGE = {
    ModelProfile.TREE: "tree pipeline",
    ModelProfile.SCALE_SENSITIVE: "scaled pipeline",
}
MIN_BASES = 2


def _init_selection(state: AppState) -> None:
    """Seed widget state once so cards never mix `value=` with session-state updates."""
    for spec in list_models(state.problem_type, state.forecasting):
        st.session_state.setdefault(f"m_{spec.key}", spec.key in state.selected_models)


def _tags(spec) -> str:  # noqa: ANN001 - ModelSpec
    speed, _ = meta_for(spec.key)
    pipeline = "ensemble" if spec.key in ENSEMBLE_KEYS else PROFILE_BADGE[spec.profile]
    return (
        f'<span class="tml-tag">{pipeline}</span>'
        f'<span class="tml-tag {SPEED_CLASS[speed]}">{speed}</span>'
    )


def _card(spec, icon: str) -> None:  # noqa: ANN001 - ModelSpec
    _, best_for = meta_for(spec.key)
    with st.container(border=True, key=f"card_{spec.key}"):
        st.markdown(f"{icon} **{spec.name}**")
        st.markdown(_tags(spec), unsafe_allow_html=True)
        st.caption(f"**Best for:** {best_for}  \n{spec.description}")
        st.toggle("Include", key=f"m_{spec.key}")


def _group_header(group: str, members: list, icon: str) -> None:
    chosen = sum(bool(st.session_state.get(f"m_{s.key}")) for s in members)
    left, mid, right = st.columns([6, 1, 1])
    left.markdown(
        f"#### {icon} {group} <small>· {chosen} of {len(members)} selected</small>",
        unsafe_allow_html=True,
    )
    if mid.button("All", key=f"all_{group}", width="stretch"):
        for s in members:
            st.session_state[f"m_{s.key}"] = True
        st.rerun()
    if right.button("None", key=f"none_{group}", width="stretch"):
        for s in members:
            st.session_state[f"m_{s.key}"] = False
        st.rerun()


def _model_cards(state: AppState) -> list[str]:
    _init_selection(state)
    specs = list_models(state.problem_type, state.forecasting)
    for group, members_iter in groupby(specs, key=lambda s: s.group):
        members = list(members_iter)
        icon = GROUP_ICONS.get(group, ":material/category:")
        _group_header(group, members, icon)
        cols = st.columns(3)
        for i, spec in enumerate(members):
            with cols[i % 3]:
                _card(spec, icon)
    return [s.key for s in specs if st.session_state.get(f"m_{s.key}")]


def _selection_bar(state: AppState) -> None:
    names = {s.key: s.name for s in list_models(state.problem_type, state.forecasting)}
    chosen = [names[k] for k in names if st.session_state.get(f"m_{k}")]
    with st.container(border=True):
        left, b1, b2 = st.columns([4.6, 1.7, 1], vertical_alignment="center")
        if chosen:
            chips = "".join(f'<span class="tml-chip">{n}</span>' for n in chosen)
            left.markdown(f"**{len(chosen)} selected** &nbsp; {chips}", unsafe_allow_html=True)
        else:
            left.markdown(
                "**No models selected yet.** Pick cards below or use the recommended set."
            )
        if b1.button("Recommended set", key="rec_set", width="stretch"):
            _set_recommended(state)
            st.rerun()
        if b2.button("Clear", key="clear_set", width="stretch"):
            for key in names:
                st.session_state[f"m_{key}"] = False
            st.rerun()


def _ensemble_bases(state: AppState, selected: list[str]) -> dict[str, tuple[str, ...]]:
    bases: dict[str, tuple[str, ...]] = {}
    candidates = [
        s
        for s in list_models(state.problem_type, state.forecasting)
        if s.key not in ENSEMBLE_KEYS | FORECAST_KEYS
    ]
    names = {s.key: s.name for s in candidates}
    for key in ("voting", "stacking"):
        if key not in selected:
            continue
        default = state.ensemble_bases.get(key) or ("rf", "xgb", "lgbm")
        chosen = st.multiselect(
            f"Base models for {key}",
            list(names),
            default=list(default),
            format_func=names.get,
            key=f"bases_{key}",
        )
        if len(chosen) < MIN_BASES:
            st.warning(f"{key.title()} needs at least {MIN_BASES} base models.")
        bases[key] = tuple(chosen)
    return bases


def _set_recommended(state: AppState) -> None:
    for spec in list_models(state.problem_type, state.forecasting):
        recommended = RECOMMENDED_FORECAST if state.forecasting else RECOMMENDED[state.problem_type]
        st.session_state[f"m_{spec.key}"] = spec.key in recommended


def _primary_metric(state: AppState, y) -> str:  # noqa: ANN001
    options = list(scoring_map(state.problem_type, y))
    default = state.primary_metric or default_primary_metric(state.problem_type, y)
    return st.selectbox(
        "Primary metric (ranks the leaderboard)",
        options,
        index=options.index(default) if default in options else 0,
        format_func=metric_label,
        key="primary_metric",
    )


def render(state: AppState) -> None:
    st.header("4. Choose models")
    st.caption(
        "Compare Random Forest with boosting, linear baselines and ensembles. Each "
        "model gets the preprocessing pipeline that suits it."
    )
    show_notice()
    if is_training():
        progress_panel()
        return
    _init_selection(state)
    _selection_bar(state)
    selected = _model_cards(state)
    bases = _ensemble_bases(state, selected)
    manual, tuning, grids = settings_panel(selected, state.problem_type, state.config.cv_folds)
    effort = {
        **tuning,
        **{k: grid_size({n: list(v) for n, v in g.items()}) for k, g in grids.items()},
    }
    _, y, _, _ = prepare_data(state.raw_df, state.config)
    primary = _primary_metric(state, y)
    if tuple(selected) != state.selected_models:
        set_state(selected_models=tuple(selected))
    ready = bool(selected) and all(len(b) >= MIN_BASES for b in bases.values())
    _summary(state, selected, effort)
    if selected:
        estimate_card(estimate_for(state, selected, effort, bases))
    if st.button("Train & Predict", type="primary", disabled=not ready, key="train"):
        config = replace(get_state().config, model_params=manual, tuning=tuning, grids=grids)
        start_training(get_state(), selected, bases, primary, config)
    nav_buttons(back=3, next_step=5 if state.run else None)


def _summary(state: AppState, selected: list[str], tuning: dict[str, int]) -> None:
    """Plain summary of what will happen when Train & Predict is pressed."""
    cfg = state.config
    fits = sum((tuning.get(k, 0) + 1) * cfg.cv_folds + 1 for k in selected)
    tuned = f", {len(tuning)} with hyperparameter search" if tuning else ""
    st.info(
        f"**{len(selected)} model(s){tuned}** \u2192 about {fits} model fits. "
        f"{describe(cfg)} The held-out test set ({cfg.test_size:.0%}) is scored once at the "
        "end and stays locked until you reveal it.",
        icon=":material/schedule:",
    )
