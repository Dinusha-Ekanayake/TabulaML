"""Step 4 training panel: time estimate, background training, progress and Cancel.

Training runs in a ``TrainingJob`` thread. A fragment refreshes the progress card every
second; when the job ends, ``finish_training`` (called on every page run from app.py)
stores the run and moves to step 5. Nothing here contains ML logic.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from tabulaml.core.types import PreprocessingConfig
from tabulaml.training.estimate import TrainingEstimate, estimate_training
from tabulaml.training.jobs import TrainingJob
from tabulaml.ui.history import save_run
from tabulaml.ui.state import AppState, get_state, reset_from

JOB_KEY = "tml_training_job"
NOTICE_KEY = "tml_training_notice"
LEVEL_STYLE = {
    "quick": ("tml-speed-fast", ":material/bolt:"),
    "moderate": ("tml-speed-medium", ":material/schedule:"),
    "long": ("tml-speed-slow", ":material/hourglass_top:"),
}
REFRESH_SECONDS = 1.0


def current_job() -> dict[str, Any] | None:
    return st.session_state.get(JOB_KEY)


def is_training() -> bool:
    job = current_job()
    return job is not None and job["job"].snapshot().state == "running"


def estimate_for(
    state: AppState, selected: list[str], tuning: dict[str, int], bases: dict
) -> TrainingEstimate:
    cfg = state.config
    rows = len(state.raw_df) if not cfg.max_rows else min(cfg.max_rows, len(state.raw_df))
    dev_rows = int(rows * (1 - cfg.test_size))
    n_features = sum(1 for c in cfg.included() if c.name != cfg.target)
    return estimate_training(
        dev_rows,
        n_features,
        selected,
        cfg.cv_folds,
        tuning=tuning,
        ensemble_bases=bases,
        knn_imputation=any(c.impute == "knn" for c in cfg.included()),
        n_targets=len(getattr(cfg, "targets", (cfg.target,))),
    )


def estimate_card(estimate: TrainingEstimate) -> None:
    css, icon = LEVEL_STYLE[estimate.level]
    with st.container(border=True, key="estimate_card"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown(f"{icon} **Estimated training time: {estimate.label}**")
        right.markdown(
            f'<span class="tml-tag {css}">{estimate.level}</span>', unsafe_allow_html=True
        )
        for note in estimate.warnings:
            st.caption(f":material/warning: {note}")
        st.caption(
            "A rough range for this machine. Training runs in the background; you can "
            "cancel it at any time and keep the models that already finished."
        )


def start_training(
    state: AppState,
    selected: list[str],
    bases: dict,
    primary: str,
    config: PreprocessingConfig,
) -> None:
    job = TrainingJob(
        df=state.raw_df,
        config=config,
        model_keys=list(selected),
        primary_metric=primary,
        ensemble_bases=bases,
    ).start()
    st.session_state[JOB_KEY] = {
        "job": job,
        "base_config": state.config,
        "selected": tuple(selected),
        "bases": bases,
        "primary": primary,
        "dataset": state.card.title if state.card else (state.filename or "upload"),
    }
    st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def progress_panel() -> None:
    """Live progress card with a Cancel button (re-runs itself every second)."""
    job = current_job()
    if job is None:
        return
    snap = job["job"].snapshot()
    if snap.state != "running":
        st.rerun()  # full page run: finish_training picks the result up
        return
    with st.container(border=True, key="training_card"):
        st.markdown(":material/model_training: **Training in progress**")
        st.progress(min(snap.fraction, 1.0), text=snap.label)
        left, mid, right = st.columns([2, 2, 1.4], vertical_alignment="center")
        left.metric("Elapsed", _clock(snap.elapsed))
        mid.metric("Remaining (approx.)", _clock(snap.remaining) if snap.remaining else "...")
        if snap.stopping:
            right.caption("Stopping after the current fold...")
        elif right.button("Cancel", key="cancel_training", icon=":material/stop_circle:"):
            job["job"].cancel()
            st.rerun(scope="fragment")
        st.caption("You can keep this page open. Cancelling keeps every model that has finished.")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def finish_training() -> None:
    """Called on every page run: store a finished job's result exactly once."""
    job = current_job()
    if job is None:
        return
    snap = job["job"].snapshot()
    if snap.state == "running":
        return
    del st.session_state[JOB_KEY]
    state = get_state()
    if snap.state == "failed":
        st.session_state[NOTICE_KEY] = ("error", f"Training failed: {snap.error}")
        return
    if snap.run is None:
        st.session_state[NOTICE_KEY] = (
            "warning",
            "Training was cancelled before any model finished. Nothing was saved.",
        )
        return
    if state.config != job["base_config"]:
        st.session_state[NOTICE_KEY] = (
            "warning",
            "Settings changed while training, so those results were discarded. "
            "Press Train & Predict again.",
        )
        return
    save_run(snap.run, job["dataset"])
    reset_from(
        5,
        run=snap.run,
        selected_models=job["selected"],
        ensemble_bases=job["bases"],
        primary_metric=job["primary"],
        step=5,
    )


def show_notice() -> None:
    notice = st.session_state.pop(NOTICE_KEY, None)
    if notice:
        level, text = notice
        (st.error if level == "error" else st.warning)(text)
