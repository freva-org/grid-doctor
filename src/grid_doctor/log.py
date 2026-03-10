"""Logging configuration for grid-doctor.

Provides a single knob that controls the log level of **all** loggers —
those that already exist and those created later.

Examples
--------
In a CLI entry point::

    from grid_doctor.log import setup_logging
    setup_logging(verbosity=2)          # WARNING → INFO → DEBUG

At runtime::

    from grid_doctor.log import increase, decrease, set_level
    increase()                          # one step more verbose
    decrease()                          # one step less verbose
    set_level("DEBUG")                  # explicit level
"""

from __future__ import annotations

import logging
from typing import Union, cast

#: Verbosity ladder from least to most verbose.
_LEVELS: list[int] = [
    logging.CRITICAL,  # 50
    logging.ERROR,  # 40
    logging.WARNING,  # 30  ← default
    logging.INFO,  # 20
    logging.DEBUG,  # 10
]

_DEFAULT_INDEX: int = _LEVELS.index(logging.WARNING)
_current_index: int = _DEFAULT_INDEX

_DEFAULT_FORMAT: str = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ── internal helpers ───────────────────────────────────────────────
def _apply_level(level: int) -> None:
    """Set *level* on the root logger and every existing logger."""
    logging.root.setLevel(level)
    for lgr in logging.Logger.manager.loggerDict.values():
        if isinstance(lgr, logging.Logger):
            lgr.setLevel(level)


class _LevelPropagatingManager(logging.Manager):
    """Manager that applies the current global level to new loggers."""

    def getLogger(self, name: str) -> logging.Logger:  # noqa: N802
        lgr = super().getLogger(name)
        lgr.setLevel(_LEVELS[_current_index])
        return lgr


# Install once on import.
logging.Logger.manager.__class__ = _LevelPropagatingManager


# ── public API: level manipulation ─────────────────────────────────
def set_level(level: Union[int, str]) -> None:
    """Set all loggers (existing and future) to *level*.

    Parameters
    ----------
    level : int or str
        A log level as an ``int`` (e.g. ``logging.DEBUG``) or a
        case-insensitive string (e.g. ``"debug"``).
    """
    global _current_index
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    level = cast(int, level)
    try:
        _current_index = _LEVELS.index(level)
    except ValueError:
        _current_index = min(
            range(len(_LEVELS)), key=lambda i: abs(_LEVELS[i] - level)
        )
    _apply_level(_LEVELS[_current_index])


def increase(steps: int = 1) -> int:
    """Make logging more verbose (e.g. WARNING → INFO → DEBUG).

    Parameters
    ----------
    steps : int, optional
        Number of verbosity steps to increase (default ``1``).

    Returns
    -------
    int
        The new log level.
    """
    global _current_index
    _current_index = min(_current_index + steps, len(_LEVELS) - 1)
    _apply_level(_LEVELS[_current_index])
    return _LEVELS[_current_index]


def decrease(steps: int = 1) -> int:
    """Make logging less verbose (e.g. DEBUG → INFO → WARNING).

    Parameters
    ----------
    steps : int, optional
        Number of verbosity steps to decrease (default ``1``).

    Returns
    -------
    int
        The new log level.
    """
    global _current_index
    _current_index = max(_current_index - steps, 0)
    _apply_level(_LEVELS[_current_index])
    return _LEVELS[_current_index]


def get_level() -> int:
    """Return the current global log level.

    Returns
    -------
    int
        One of ``logging.DEBUG``, ``logging.INFO``, etc.
    """
    return _LEVELS[_current_index]


# ── public API: initial setup ──────────────────────────────────────
def setup_logging(
    verbosity: int = 0,
    level: Union[int, str, None] = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE_FORMAT,
) -> None:
    """Configure the root handler and set the global log level.

    Call this **once** at program startup.

    Parameters
    ----------
    verbosity : int, optional
        Number of ``-v`` flags.  Each step moves one rung towards
        ``DEBUG`` on the verbosity ladder.
    level : int, str, or None, optional
        Explicit level — overrides *verbosity* when given.
    fmt : str, optional
        :mod:`logging` format string.
    datefmt : str, optional
        Date format string.
    """
    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.root.addHandler(handler)

    if level is not None:
        set_level(level)
    else:
        global _current_index
        _current_index = _DEFAULT_INDEX + verbosity
        _current_index = max(0, min(_current_index, len(_LEVELS) - 1))
        _apply_level(_LEVELS[_current_index])
