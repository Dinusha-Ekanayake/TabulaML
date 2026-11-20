"""Run training in a background thread so the UI can show progress and a Cancel button.

The job never touches Streamlit: the UI polls ``snapshot()``. Cancelling is cooperative:
the trainer checks the stop flag between cross-validation folds and between models, so a
fold that is already running finishes first. Finished models are kept.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tabulaml.core.types import TrainingRun
from tabulaml.training.trainer import TrainingCancelled, train_models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobSnapshot:
    state: str  # running | done | cancelled | failed
    label: str
    fraction: float
    elapsed: float
    remaining: float | None  # seconds, from the progress so far (None until known)
    run: TrainingRun | None
    error: str | None
    stopping: bool


class TrainingJob:
    """One background training run. Create it, call ``start()``, poll ``snapshot()``."""

    def __init__(self, train: Callable[..., TrainingRun] = train_models, **kwargs: Any) -> None:
        self._train, self._kwargs = train, kwargs
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, daemon=True, name="tabulaml-train")
        self._state, self._label, self._fraction = "running", "Starting...", 0.0
        self._run: TrainingRun | None = None
        self._error: str | None = None
        self._started = self._finished = 0.0

    @property
    def config(self) -> Any:
        return self._kwargs.get("config")

    def start(self) -> TrainingJob:
        self._started = time.monotonic()
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> JobSnapshot:
        self._thread.join(timeout)
        return self.snapshot()

    def _progress(self, label: str, done: int, total: int) -> None:
        with self._lock:
            self._label = label
            self._fraction = done / total if total else 0.0

    def _work(self) -> None:
        try:
            stop = self._stop.is_set
            run = self._train(**self._kwargs, progress=self._progress, should_stop=stop)
            state = "cancelled" if run.cancelled else "done"
            with self._lock:
                self._run, self._state, self._fraction = run, state, 1.0
        except TrainingCancelled:
            with self._lock:
                self._state = "cancelled"
        except Exception as exc:  # noqa: BLE001 - reported to the user, never crashes the app
            logger.exception("Training job failed")
            with self._lock:
                self._state, self._error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self._finished = time.monotonic()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            running = self._state == "running"
            end = time.monotonic() if running else self._finished
            elapsed = max(0.0, end - self._started)
            remaining = None
            if running and self._fraction > 0.02:
                remaining = elapsed * (1 - self._fraction) / self._fraction
            return JobSnapshot(
                self._state,
                self._label,
                self._fraction,
                elapsed,
                remaining,
                self._run,
                self._error,
                self._stop.is_set() and running,
            )
