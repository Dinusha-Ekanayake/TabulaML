"""Progress reporting and cooperative cancellation for the training loop."""

from __future__ import annotations

from collections.abc import Callable

Progress = Callable[[str, int, int], None]  # (label, steps done, total steps)
StopCheck = Callable[[], bool]


class TrainingCancelled(Exception):  # noqa: N818 - reads naturally at call sites
    """Raised when the user stops training before any model has finished."""


class _Tracker:
    """Fold-level progress reporting and cooperative cancellation (checked between folds
    and models; a single fold that is already running always completes)."""

    def __init__(self, total: int, progress: Progress | None, should_stop: StopCheck | None):
        self.total, self.done = total, 0
        self._progress, self._should_stop = progress, should_stop

    def report(self, label: str) -> None:
        if self._progress:
            self._progress(label, min(self.done, self.total), self.total)

    def step(self, label: str) -> None:
        self.done += 1
        self.report(label)

    def check(self) -> None:
        if self._should_stop is not None and self._should_stop():
            raise TrainingCancelled("Training was cancelled.")
