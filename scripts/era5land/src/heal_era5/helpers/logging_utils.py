"""Shared helpers for compact structured logging in the ERA5-Land workflow."""

import logging
import sys
from typing import Any

from dask.callbacks import Callback


def log_stage(logger: logging.Logger, stage: str, **fields: object) -> None:
    """Emit a compact structured log line for one workflow stage.

    Parameters
    ----------
    logger:
        Logger instance that should receive the message.
    stage:
        Stable stage identifier used by the colored formatter.
    **fields:
        Additional structured key/value pairs appended to the message.
    """

    tokens = [f"stage={stage}"]
    tokens.extend(f"{key}={value}" for key, value in fields.items())
    logger.info(" ".join(tokens))


class _TaskProgress(Callback):
    """Report bounded percentage updates while Dask computes a task graph."""

    def __init__(self, logger: logging.Logger, stage: str, label: str) -> None:
        super().__init__()
        self.logger = logger
        self.stage = stage
        self.label = label
        self.total_tasks = 1
        self.completed = 0
        self.next_percent = 5
        self.last_percent = -1
        self.live = sys.stderr.isatty()

    def _start(self, dsk: Any) -> None:
        # ``dsk`` is the optimized graph Dask will execute, unlike the larger
        # pre-optimization graph returned by ``to_zarr(compute=False)``.
        self.total_tasks = max(1, len(dsk))
        log_stage(self.logger, f"{self.stage}_start", label=self.label, tasks=self.total_tasks)

    def _render_live_progress(self, percent: int) -> None:
        width = 20
        filled = width * percent // 100
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r[{bar}] {percent:3d}% {self.completed}/{self.total_tasks} tasks")
        sys.stderr.flush()

    def _posttask(self, *args: object) -> None:
        self.completed += 1
        percent = min(100, self.completed * 100 // self.total_tasks)
        if self.live:
            if percent != self.last_percent:
                self._render_live_progress(percent)
                self.last_percent = percent
            return
        if percent >= self.next_percent:
            log_stage(
                self.logger,
                f"{self.stage}_progress",
                label=self.label,
                percent=percent,
                completed_tasks=self.completed,
                total_tasks=self.total_tasks,
            )
            self.next_percent = (percent // 5 + 1) * 5

    def _finish(self, dsk: Any, state: Any, errored: bool) -> None:
        if not errored:
            self.completed = self.total_tasks
            if self.live:
                self._render_live_progress(100)
                sys.stderr.write("\n")
                sys.stderr.flush()
            log_stage(
                self.logger,
                f"{self.stage}_done",
                label=self.label,
                completed_tasks=self.completed,
                total_tasks=self.total_tasks,
            )


def compute_with_task_progress(
    delayed: Any,
    *,
    logger: logging.Logger,
    stage: str,
    label: str,
) -> None:
    """Compute a Dask delayed object while logging progress every five percent."""

    with _TaskProgress(logger, stage, label):
        delayed.compute()
