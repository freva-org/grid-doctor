"""Library-wide settings.

Each setting has a built-in default, which the environment variable
``GRID_DOCTOR_<SETTING>`` overrides (read when ``grid_doctor.config`` is
imported), which assignment at runtime overrides in turn::

    import grid_doctor as gd

    gd.config.ellipsoid = "WGS84"

    with gd.config.set(ellipsoid="sphere"):  # restored after the block
        ...

Settings are process-wide: a ``with gd.config.set(...)`` block also affects
other threads while it is open.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import Any, get_args

from .types import Ellipsoid

ENV_VAR_PREFIX = "GRID_DOCTOR"
"""Prefix of the environment variables, ``<PREFIX>_<SETTING>``."""

ellipsoid: Ellipsoid = "sphere"
"""Ellipsoid on which new HEALPix grids are built, passed to ``healpix_geo``:
``"sphere"`` (consistent with ESMF's spherical area calculation) or
``"WGS84"``.  It is recorded in the output and in weight files; code reading
existing data uses the recorded one instead (a sphere if the HEALPix
metadata or weight file records none).  Only a dataset with no HEALPix
metadata at all, selected from with an explicit ``level=``, falls back to
this setting.  Environment variable: ``GRID_DOCTOR_ELLIPSOID``."""

weights_store: str | None = None
"""Location of the default
[`WeightsStore`][grid_doctor.weights.WeightsStore]: a local path or a
``file://`` or ``memory://`` URL.  ``None`` uses
``$XDG_CACHE_HOME/grid-doctor/weights`` (``~/.cache/grid-doctor/weights``).
Environment variable: ``GRID_DOCTOR_WEIGHTS_STORE``."""

SETTINGS: tuple[str, ...] = ("ellipsoid", "weights_store")
"""Names of all settings."""

_CHOICES: dict[str, tuple[object, ...]] = {"ellipsoid": get_args(Ellipsoid)}


def env_var(name: str) -> str:
    """Return the environment variable of setting *name*."""
    return f"{ENV_VAR_PREFIX}_{name.upper()}"


def _check(name: str, value: object) -> None:
    """Raise unless *value* is valid for setting *name*."""
    if name not in SETTINGS:
        raise AttributeError(f"Unknown setting {name!r}; settings are {SETTINGS}")
    choices = _CHOICES.get(name)
    if choices is not None and value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _read_env(environ: Mapping[str, str]) -> dict[str, Any]:
    """Return the settings given in *environ*, validated and stripped.

    An empty ``GRID_DOCTOR_WEIGHTS_STORE`` counts as unset.

    Raises
    ------
    ValueError
        If a variable holds an invalid value; the message names the variable.
    """
    values: dict[str, Any] = {}
    for name in SETTINGS:
        var = env_var(name)
        if var not in environ:
            continue
        value: str | None = environ[var].strip()
        if name == "weights_store" and not value:
            value = None
        try:
            _check(name, value)
        except ValueError as error:
            raise ValueError(f"Invalid {var}: {error}") from None
        values[name] = value
    return values


@contextmanager
def set(**settings: Any) -> Iterator[None]:
    """Change settings inside a ``with`` block, restoring them afterwards.

    Raises
    ------
    AttributeError
        If a setting does not exist.
    ValueError
        If a value is invalid; no setting is changed then.
    """
    for name, value in settings.items():
        _check(name, value)
    module = sys.modules[__name__]
    previous = {name: getattr(module, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


class _ConfigModule(ModuleType):
    """Module type that validates assignments to settings."""

    def __setattr__(self, name: str, value: object) -> None:
        if name in SETTINGS:
            _check(name, value)
        super().__setattr__(name, value)


globals().update(_read_env(os.environ))
sys.modules[__name__].__class__ = _ConfigModule
