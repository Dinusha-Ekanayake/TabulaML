"""Step 4: model selection dashboard and the Train & Predict action."""

from __future__ import annotations

from itertools import groupby

import streamlit as st

from tabulaml.core.types import ModelProfile
from tabulaml.evaluation.metrics import default_primary_metric, scoring_map
from tabulaml.models.registry import ENSEMBLE_KEYS, RECOMMENDED, list_models
from tabulaml.training.trainer import prepare_data, train_models
from tabulaml.ui.components import nav_buttons
from tabulaml.ui.helpers import metric_label
from tabulaml.ui.state import AppState, get_state, reset_from, set_state

PROFILE_BADGE = {
    ModelProfile.TREE: "tree pipeline",
    ModelProfile.SCALE_SENSITIVE: "scaled pipeline",
}
MIN_BASES = 2


def _model_checkboxes(state: AppState) -> list[str]:
    selected = []
    specs = list_models(state.problem_type)
    for group, members in groupby(specs, key=lambda s: s.group):
        st.markdown(f"**{group}**")
        cols = st.columns(3)
        for i, spec in enumerate(members):
            with cols[i % 3]:
                badge = "ensemble" if spec.key in ENSEMBLE_KEYS else PROFILE_BADGE[spec.profile]
                if st.checkbox(
                    spec.name,
                    value=spec.key in state.selected_models,
                    key=f"m_{spec.key}",
                    help=f"{spec.description} ({badge})",
                ):
                    selected.append(spec.key)
    return selected


def _ensemble_bases(state: AppState, selected: list[str]) -> dict[str, tuple[str, ...]]:
    bases: dict[str, tuple[str, ...]] = {}
    candidates = [s for s in list_models(state.problem_type) if s.key not in ENSEMBLE_KEYS]
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
    for spec in list_models(state.problem_type):
        st.session_state[f"m_{spec.key}"] = spec.key in RECOMMENDED[state.problem_type]


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


def _train(state: AppState, selected: list[str], bases: dict, primary: str) -> None:
    bar = st.progress(0.0, text="Starting...")

    def progress(key: str, i: int, n: int) -> None:
        text = (
            "Finished"
            if key == "done"
            else f"Training {key} ({i + 1}/{n}): cross-validation, fit, test"
        )
        bar.progress(i / n, text=text)

    with st.status(
        "Training models (cross-validation + held-out test)...", expanded=False
    ) as status:
        run = train_models(
            state.raw_df,
            state.config,
            selected,
            primary_metric=primary,
            progress=progress,
            ensemble_bases=bases,
        )
        failed = [r for r in run.results if r.status != "ok"]
        status.update(
            label=f"Done: {len(run.results) - len(failed)} trained, {len(failed)} failed",
            state="complete" if not failed else "error",
        )
    reset_from(
        5,
        run=run,
        selected_models=tuple(selected),
        ensemble_bases=bases,
        primary_metric=primary,
        step=5,
    )
    st.rerun()


def render(state: AppState) -> None:
    st.header("4. Choose models")
    top = st.columns([1, 1, 4])
    if top[0].button("Recommended set", key="rec_set"):
        _set_recommended(state)
        st.rerun()
    if top[1].button("Clear", key="clear_set"):
        for spec in list_models(state.problem_type):
            st.session_state[f"m_{spec.key}"] = False
        st.rerun()
    selected = _model_checkboxes(state)
    bases = _ensemble_bases(state, selected)
    _, y, _, _ = prepare_data(state.raw_df, state.config)
    primary = _primary_metric(state, y)
    if tuple(selected) != state.selected_models:
        set_state(selected_models=tuple(selected))
    ready = bool(selected) and all(len(b) >= MIN_BASES for b in bases.values())
    st.caption(
        f"{len(selected)} model(s) selected. Each is cross-validated "
        f"({state.config.cv_folds} folds) on the training split, then scored once on "
        f"the held-out test set ({state.config.test_size:.0%})."
    )
    if st.button("Train & Predict", type="primary", disabled=not ready, key="train"):
        _train(get_state(), selected, bases, primary)
    nav_buttons(back=3, next_step=5 if state.run else None)
