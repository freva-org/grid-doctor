"""Library-wide settings.

Each setting has a built-in default.  Config files, environment variables
and changes at runtime override it, in this order (later ones win):

1. the environment's file ``<sys.prefix>/etc/grid-doctor/config.toml``,
   e.g. for everyone using a shared conda environment,
2. the user's file in the user config directory of
   [platformdirs](https://platformdirs.readthedocs.io)
   (``~/.config/grid-doctor/config.toml`` on Linux), or the file named by
   ``GRID_DOCTOR_CONFIG`` instead,
3. the environment variable ``GRID_DOCTOR_<SETTING>``,
4. changes at runtime, including files loaded with ``set``.

```python
import grid_doctor as gd

gd.config.set("project.toml", ellipsoid="WGS84")  # until changed

with gd.config.set(scratch_dir="/tmp/gd"):  # restored after the block
    ...
```

Files and environment variables are read when ``grid_doctor.config`` is
imported, and again by ``refresh()``.  ``python -m grid_doctor.config.template``
writes a commented config file to start from.

Settings are process-wide: a ``with gd.config.set(...)`` block also affects
other threads while it is open.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any, get_args

from platformdirs import user_config_path

from ..types import Ellipsoid

ENV_VAR_PREFIX = "GRID_DOCTOR"
"""Prefix of the environment variables, ``<PREFIX>_<SETTING>``."""

CONFIG_ENV_VAR = "GRID_DOCTOR_CONFIG"
"""Environment variable naming a user config file to read instead of
[`user_config_file()`][grid_doctor.config.user_config_file].  Set it to an
empty value to read no user file."""

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
``file://`` or ``memory://`` URL.  ``None`` uses ``grid-doctor/weights`` in
the user cache directory of platformdirs (``~/.cache/grid-doctor/weights``
on Linux).  Environment variable: ``GRID_DOCTOR_WEIGHTS_STORE``."""

esmf_regrid_weightgen: str | None = None
"""Name or path of the ``ESMF_RegridWeightGen`` executable for offline
weight generation, used when the ``esmf_regrid_weightgen=`` argument is not
given.  ``None`` looks up ``ESMF_RegridWeightGen`` in ``PATH``.
Environment variable: ``GRID_DOCTOR_ESMF_REGRID_WEIGHTGEN``."""

scratch_dir: str | None = None
"""Directory in which offline weight generation creates its temporary
working directories (for the intermediate mesh files, which get large for
fine grids), used when the ``workdir=`` argument is not given.  Each run
gets its own subdirectory, removed afterwards.  ``None`` uses the system's
temporary directory (``$TMPDIR``, usually ``/tmp``).  Environment variable:
``GRID_DOCTOR_SCRATCH_DIR``."""

SETTINGS: tuple[str, ...] = ("ellipsoid", "weights_store", "esmf_regrid_weightgen", "scratch_dir")
"""Names of all settings."""

loaded_files: tuple[Path, ...] = ()
"""Config files in effect, lowest priority first: those read at import or by
``refresh()``, then those loaded with ``set()``."""

_CHOICES: dict[str, tuple[object, ...]] = {"ellipsoid": get_args(Ellipsoid)}

# Settings that may be None; an empty value in a file or an environment
# variable unsets them.  Their values are paths, so "~" is expanded.
_OPTIONAL: frozenset[str] = frozenset({"weights_store", "esmf_regrid_weightgen", "scratch_dir"})


def env_var(name: str) -> str:
    """Return the environment variable of setting *name*."""
    return f"{ENV_VAR_PREFIX}_{name.upper()}"


def user_config_file() -> Path:
    r"""Return the default user config file.

    ``config.toml`` in the user config directory of platformdirs:
    ``~/.config/grid-doctor/config.toml`` on Linux (following
    ``XDG_CONFIG_HOME``), ``~/Library/Application Support/grid-doctor/config.toml``
    on macOS, ``%APPDATA%\grid-doctor\config.toml`` on Windows.  The file
    need not exist.
    """
    return user_config_path("grid-doctor", appauthor=False) / "config.toml"


def environment_config_file() -> Path:
    """Return the config file of the Python environment.

    ``<sys.prefix>/etc/grid-doctor/config.toml``, e.g.
    ``$CONDA_PREFIX/etc/grid-doctor/config.toml`` in a conda environment.
    The file need not exist.
    """
    return Path(sys.prefix) / "etc" / "grid-doctor" / "config.toml"


def config_files(environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Return the default config files, lowest priority first.

    These are the environment's file
    ([`environment_config_file()`][grid_doctor.config.environment_config_file])
    and the user's file: the one named by ``GRID_DOCTOR_CONFIG``, or else
    [`user_config_file()`][grid_doctor.config.user_config_file].  The files
    need not exist.

    Parameters
    ----------
    environ:
        Environment variables to use, ``os.environ`` by default.
    """
    environ = os.environ if environ is None else environ
    files = [environment_config_file()]
    user = environ.get(CONFIG_ENV_VAR)
    if user is None:
        files.append(user_config_file())
    elif user.strip():
        files.append(Path(user.strip()).expanduser())
    return tuple(files)


def _check(name: str, value: object) -> None:
    """Raise unless *value* is valid for setting *name*."""
    if name not in SETTINGS:
        raise AttributeError(f"Unknown setting {name!r}; settings are {SETTINGS}")
    choices = _CHOICES.get(name)
    if choices is not None and value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    if name in _OPTIONAL and not (value is None or isinstance(value, str | os.PathLike)):
        raise ValueError(f"{name} must be a string, a path or None, got {value!r}")


def _clean(name: str, value: str) -> str | None:
    """Return *value* stripped, ``~`` expanded, ``None`` if empty and optional."""
    value = value.strip()
    if name in _OPTIONAL:
        return os.path.expanduser(value) if value else None
    return value


def _read_env(environ: Mapping[str, str]) -> dict[str, Any]:
    """Return the settings given in *environ*, validated and stripped.

    An empty variable unsets an optional setting (e.g. ``weights_store``).

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
        value = _clean(name, environ[var])
        try:
            _check(name, value)
        except ValueError as error:
            raise ValueError(f"Invalid {var}: {error}") from None
        values[name] = value
    return values


def _read_file(path: Path) -> dict[str, Any]:
    """Return the settings in the TOML file *path*, validated.

    An empty string unsets an optional setting set by a file read before.

    Raises
    ------
    ValueError
        If the file is not valid TOML, has an unknown key or an invalid
        value; the message names the file.
    """
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid config file {path}: {error}") from None
    unknown = sorted(data.keys() - frozenset(SETTINGS))  # "set" is this module's own
    if unknown:
        raise ValueError(f"Unknown setting(s) {', '.join(map(repr, unknown))} in {path}; settings are {SETTINGS}")
    values: dict[str, Any] = {}
    for name, raw in data.items():
        if not isinstance(raw, str):
            raise ValueError(f"Invalid {name!r} in {path}: expected a string, got {raw!r}")
        value = _clean(name, raw)
        try:
            _check(name, value)
        except ValueError as error:
            raise ValueError(f"Invalid {name!r} in {path}: {error}") from None
        values[name] = value
    return values


def _load(environ: Mapping[str, str]) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Return the settings from the config files and *environ*, and the files read.

    Raises
    ------
    FileNotFoundError
        If ``GRID_DOCTOR_CONFIG`` names a file that does not exist.
    ValueError
        If a file or a variable holds an invalid setting.
    """
    values: dict[str, Any] = {}
    read: list[Path] = []
    files = config_files(environ)
    for path in files:
        if not path.is_file():
            if path == files[-1] and environ.get(CONFIG_ENV_VAR, "").strip():
                raise FileNotFoundError(f"{CONFIG_ENV_VAR} names {path}, which does not exist")
            continue
        values.update(_read_file(path))
        read.append(path)
    values.update(_read_env(environ))
    return values, tuple(read)


class set:  # noqa: N801  # a function to its users, like dask.config.set
    """Change settings, from a config file and/or one by one.

    The settings change at once and stay until changed again.  Used as a
    context manager, they are restored when the block ends.

    ```python
    gd.config.set("project.toml")                     # a file's settings
    gd.config.set(scratch_dir="/scratch/ab1234")      # single settings
    gd.config.set("project.toml", ellipsoid="WGS84")  # the file, then these

    with gd.config.set(ellipsoid="WGS84"):            # only in the block
        ...
    ```

    A file loaded here counts as a change at runtime: it overrides
    environment variables, unlike the files read at import.  Only the
    settings it names change; an empty string unsets one.  Everything is
    checked before anything changes.  Files loaded here are not seen by
    dask workers in other processes; use ``GRID_DOCTOR_CONFIG`` for those.

    Parameters
    ----------
    config_file:
        Config file (TOML) to apply first.  Positional only.
    **settings:
        Settings to apply after the file.

    Raises
    ------
    FileNotFoundError
        If *config_file* does not exist.
    AttributeError
        If a setting does not exist.
    ValueError
        If the file or a value is invalid; no setting is changed then.
    """

    def __init__(self, config_file: str | os.PathLike[str] | None = None, /, **settings: Any) -> None:  # noqa: D107
        global loaded_files
        values: dict[str, Any] = {}
        files: tuple[Path, ...] = ()
        if config_file is not None:
            path = Path(config_file).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Config file {path} does not exist")
            values.update(_read_file(path))
            files = (path,)
        for name, value in settings.items():
            _check(name, value)
        values.update(settings)
        module = sys.modules[__name__]
        self._previous = {name: getattr(module, name) for name in values}
        self._previous_files = loaded_files
        for name, value in values.items():
            setattr(module, name, value)
        loaded_files = (*loaded_files, *files)

    def __enter__(self) -> set:
        """Return this object; the settings are already applied."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the settings and files in effect before this ``set``."""
        global loaded_files
        module = sys.modules[__name__]
        for name, value in self._previous.items():
            setattr(module, name, value)
        loaded_files = self._previous_files


def get(name: str | None = None) -> Any:
    """Return the setting *name*, or all settings as a dict if ``None``.

    Raises
    ------
    AttributeError
        If the setting does not exist.
    """
    module = sys.modules[__name__]
    if name is None:
        return {setting: getattr(module, setting) for setting in SETTINGS}
    if name not in SETTINGS:
        raise AttributeError(f"Unknown setting {name!r}; settings are {SETTINGS}")
    return getattr(module, name)


def refresh() -> None:
    """Read the default config files and environment variables again.

    Every setting is reset to its built-in default first, so the result is
    the same as at import: changes made at runtime and files loaded with
    ``set()`` are dropped.  Useful after editing a config file.

    Raises
    ------
    FileNotFoundError
        If ``GRID_DOCTOR_CONFIG`` names a file that does not exist.
    ValueError
        If a file or a variable holds an invalid setting; no setting is
        changed then.
    """
    global loaded_files
    values, files = _load(os.environ)
    module = sys.modules[__name__]
    for name in SETTINGS:
        setattr(module, name, values.get(name, _DEFAULTS[name]))
    loaded_files = files


class _ConfigModule(ModuleType):
    """Module type that validates assignments to settings."""

    def __setattr__(self, name: str, value: object) -> None:
        if name in SETTINGS:
            _check(name, value)
        super().__setattr__(name, value)


_DEFAULTS: dict[str, Any] = {name: globals()[name] for name in SETTINGS}

refresh()
sys.modules[__name__].__class__ = _ConfigModule
