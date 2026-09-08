"""Shared helpers for compact structured logging in the ERA5-Land workflow."""

import logging
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

    def __init__(self, logger: logging.Logger, stage: str, label: str, total_tasks: int) -> None:
        super().__init__()
        self.logger = logger
        self.stage = stage
        self.label = label
        self.total_tasks = max(1, total_tasks)
        self.completed = 0
        self.next_percent = 5

    def _start(self, dsk: Any) -> None:
        log_stage(self.logger, f"{self.stage}_start", label=self.label, tasks=self.total_tasks)

    def _posttask(self, *args: object) -> None:
        self.completed += 1
        percent = min(100, self.completed * 100 // self.total_tasks)
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
            log_stage(self.logger, f"{self.stage}_done", label=self.label, completed_tasks=self.completed)


def compute_with_task_progress(
    delayed: Any,
    *,
    logger: logging.Logger,
    stage: str,
    label: str,
) -> None:
    """Compute a Dask delayed object while logging progress every five percent."""

    total_tasks = len(delayed.dask)
    with _TaskProgress(logger, stage, label, total_tasks):
        delayed.compute()
