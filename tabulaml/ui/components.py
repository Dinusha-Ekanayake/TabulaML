"""Shared Streamlit widgets."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

import streamlit as st

from tabulaml.core.types import ValidationIssue
from tabulaml.ui.history import list_runs
from tabulaml.ui.state import STEPS, AppState, can_enter, go_to

CSS = "<style>" + (Path(__file__).with_name("style.css")).read_text(encoding="utf-8") + "</style>"

REPO_URL = "https://github.com/Dinusha-Ekanayake/TabulaML"


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("tabulaml")
    except Exception:  # noqa: BLE001 - footer must never break the page
        return "dev"


def footer(state: AppState) -> None:
    """Persistent footer: app info, attribution for the loaded dataset, tech and privacy."""
    card = state.card
    if card is not None:
        url = escape(card.source_url, quote=True)
        link = f' (<a href="{url}" target="_blank">source</a>)' if url else ""
        data = f"<b>Data:</b> {escape(card.title)}{link}. Licence: {escape(card.license)}."
    elif state.raw_df is not None:  # user-controlled filename: always escape
        data = (
            f"<b>Data:</b> your upload ({escape(str(state.filename))}). It stays in this session."
        )
    else:
        data = "<b>Data:</b> no dataset loaded."
    html = f"""
<div class="tml-footer">
  <div><b>TabulaML</b> v{_version()}<br>Generalised preprocessing and ensemble model comparison
  for tabular data.<br><a href="{REPO_URL}" target="_blank">GitHub repository</a></div>
  <div>{data}<br>Sample datasets are public Sri Lankan data; please cite the sources shown on
  each dataset card.</div>
  <div>Built with Streamlit · scikit-learn · XGBoost · LightGBM · CatBoost<br>
  Your data is processed only in this session and is not sent to external services.<br>
  <a href="{REPO_URL}/blob/main/docs/USER_GUIDE.md" target="_blank">User guide</a> ·
  <a href="{REPO_URL}/blob/main/docs/REPORT.md" target="_blank">Report</a></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


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
        _history()
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


def _history() -> None:
    """Past runs saved on this machine (config + metrics, no models)."""
    runs = list_runs()
    with st.expander(f":material/history: Past runs ({len(runs)})"):
        if not runs:
            st.caption("Runs appear here after you train models.")
            return
        for report in runs[:10]:
            board = report.get("leaderboard") or []
            best = board[0] if board else {}
            metric = report.get("primary_metric", "")
            score = best.get(f"cv_{metric}_mean")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            dataset = report.get("dataset", "?")
            when = str(report.get("saved_at", ""))[:15]
            st.markdown(
                f"**{dataset}**  \n{when} · best {best.get('model', '?')} · "
                f"CV {metric} {score_text}"
            )


def results_will_reset_warning(state: AppState) -> None:
    """Shown on steps 2-4 when trained results exist: edits here clear them."""
    if state.run is not None and state.step in (2, 3, 4):
        st.warning(
            "You already have trained results. Changing settings on this step clears "
            "them, and you will need to train again.",
            icon=":material/restart_alt:",
        )
