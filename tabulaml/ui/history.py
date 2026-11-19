"""Experiment history: every training run is saved as a small JSON report.

Reports live in ``artifacts/experiments/`` (git-ignored; set TABULAML_ARTIFACTS to move
them). They hold the configuration, decisions, metrics and library versions, not the
models, so the history stays small. Download a model bundle in step 6 to keep a model.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tabulaml.core.types import TrainingRun
from tabulaml.ui.helpers import run_report

logger = logging.getLogger(__name__)
MAX_LISTED = 20


def _folder() -> Path:
    return Path(os.environ.get("TABULAML_ARTIFACTS", "artifacts")) / "experiments"


def save_run(run: TrainingRun, dataset: str) -> Path | None:
    """Write the run report; never let a disk problem break the app."""
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
    """Most recent first; unreadable files are skipped."""
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
