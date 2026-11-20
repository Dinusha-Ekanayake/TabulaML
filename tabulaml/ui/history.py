"""Experiment history: every training run is kept as a small JSON report.

Locally, reports are saved in ``artifacts/experiments/`` (git-ignored; set
TABULAML_ARTIFACTS to move them). In cloud mode (a public deployment shared by many
visitors) nothing is written to the server: the history lives only in the visitor's own
browser session and disappears when the session ends. Reports hold the configuration,
decisions, metrics and library versions, not the data or the models.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tabulaml.config as app_config
from tabulaml.core.types import TrainingRun
from tabulaml.ui.helpers import run_report

logger = logging.getLogger(__name__)
MAX_LISTED = 20
SESSION_KEY = "tml_session_history"


def _session_runs() -> list[dict[str, Any]]:
    import streamlit as st  # UI layer: the per-visitor store is the Streamlit session

    return st.session_state.setdefault(SESSION_KEY, [])


def _folder() -> Path:
    return Path(os.environ.get("TABULAML_ARTIFACTS", "artifacts")) / "experiments"


def save_run(run: TrainingRun, dataset: str) -> Path | None:
    """Keep the run report (session only in cloud mode); never let storage break the app."""
    if app_config.CLOUD_MODE:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        _session_runs().insert(0, {"saved_at": stamp, **run_report(run, dataset)})
        return None
    try:
        folder = _folder()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = folder / f"run_{stamp}.json"
        report = {"saved_at": stamp, **run_report(run, dataset)}
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path
    except OSError:
        logger.warning("Could not save the run history", exc_info=True)
        return None


def list_runs(limit: int = MAX_LISTED) -> list[dict[str, Any]]:
    """Most recent first; unreadable files are skipped. Cloud mode: this session only."""
    if app_config.CLOUD_MODE:
        return list(_session_runs()[:limit])
    folder = _folder()
    if not folder.exists():
        return []
    runs = []
    for path in sorted(folder.glob("run_*.json"), reverse=True)[:limit]:
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return runs
