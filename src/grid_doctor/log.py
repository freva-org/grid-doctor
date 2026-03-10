"""
grid-doctor logging configuration.

Provides a single knob that controls the log level of **all** loggers —
those that already exist and those created later.

Usage::

    from grid_doctor.log import setup_logging, increase, decrease, set_level

    # In your CLI entry point:
    setup_logging()                     # defaults to WARNING
    setup_logging(verbosity=2)          # bumps 2 steps → DEBUG

    # At runtime:
    increase()                          # one step more verbose
    decrease()                          # one step less verbose
    set_level(logging.INFO)             # explicit level

    # In your argparse setup:
    from grid_doctor.log import add_log_args, setup_logging_from_args
    add_log_args(parser)
    args = parser.parse_args()
    setup_logging_from_args(args)
"""

from __future__ import annotations

import logging
from typing import Union

#: Verbosity ladder from least to most verbose.
_LEVELS = [
    logging.CRITICAL,  # 50
    logging.ERROR,  # 40
    logging.WARNING,  # 30  ← default
    logging.INFO,  # 20
    logging.DEBUG,  # 10
]

_DEFAULT_INDEX = _LEVELS.index(logging.WARNING)

_current_index: int = _DEFAULT_INDEX

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------
def _apply_level(level: int) -> None:
    """Set *level* on the root logger **and** every logger that has already
    been instantiated, so nothing escapes."""
    logging.root.setLevel(level)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)


class _LevelPropagatingManager(logging.Manager):
    """Drop-in replacement for :class:`logging.Manager` that automatically
    applies the current global level to every newly created logger."""

    def getLogger(self, name: str) -> logging.Logger:
        logger = super().getLogger(name)
        logger.setLevel(_LEVELS[_current_index])
        return logger


# Install once on import.
logging.Logger.manager.__class__ = _LevelPropagatingManager


# ------------------------------------------------------------------
# Public API — level manipulation
# ------------------------------------------------------------------
def set_level(level: Union[int, str]) -> None:
    """Set all loggers (existing and future) to *level*.

    Parameters
    ----------
    level:
        A log level as an int (``logging.DEBUG``, …) or a string
        (``"DEBUG"``, ``"info"``, …).
    """
    global _current_index
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    try:
        _current_index = _LEVELS.index(level)
    except ValueError:
        # Not one of the canonical five — pick the closest.
        _current_index = min(range(len(_LEVELS)), key=lambda i: abs(_LEVELS[i] - level))
    _apply_level(_LEVELS[_current_index])


def increase(steps: int = 1) -> int:
    """Make logging more verbose (e.g. WARNING → INFO → DEBUG).

    Returns the new level.
    """
    global _current_index
    _current_index = min(_current_index + steps, len(_LEVELS) - 1)
    _apply_level(_LEVELS[_current_index])
    return _LEVELS[_current_index]


def decrease(steps: int = 1) -> int:
    """Make logging less verbose (e.g. DEBUG → INFO → WARNING).

    Returns the new level.
    """
    global _current_index
    _current_index = max(_current_index - steps, 0)
    _apply_level(_LEVELS[_current_index])
    return _LEVELS[_current_index]


def get_level() -> int:
    """Return the current global log level."""
    return _LEVELS[_current_index]


# ------------------------------------------------------------------
# Public API — initial setup
# ------------------------------------------------------------------
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
    verbosity:
        Number of ``-v`` flags (each bumps one step towards DEBUG).
    level:
        Explicit level — overrides *verbosity* if given.
    fmt:
        Log format string.
    datefmt:
        Date format string.
    """
    # Ensure the root logger has at least one handler.
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
