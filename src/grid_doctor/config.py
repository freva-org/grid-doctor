"""Library-wide settings.

Each setting has a default that can be overridden by an environment variable
``GRID_DOCTOR_<NAMESPACE>__<KEY>`` (read at import) or by assignment, e.g.
``grid_doctor.config.remap.method = "conservative"``. An explicit function
argument always wins.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    get_args,
    get_origin,
    get_type_hints,
)

from .types import RemapMethod

_HINTS: dict[type, dict[str, Any]] = {}


def _type_hints(cls: type) -> dict[str, Any]:
    """Return the resolved annotations of ``cls``, computed once per class."""
    if cls not in _HINTS:
        _HINTS[cls] = get_type_hints(cls)
    return _HINTS[cls]


class _Namespace:
    """Base for a group of settings; subclasses are slotted dataclasses.

    ``Literal``-typed fields only accept their listed values. Values from the
    environment are read as stripped strings.
    """

    __slots__ = ()
    namespace: ClassVar[str]

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate ``Literal`` settings before assigning them."""
        hint = _type_hints(type(self)).get(name)
        if get_origin(hint) is Literal and value not in get_args(hint):
            raise ValueError(f"{self.namespace}.{name} must be one of {get_args(hint)}, got {value!r}")
        object.__setattr__(self, name, value)

    @contextmanager
    def set(self, **settings: Any) -> Iterator[None]:
        """Change settings inside a ``with`` block, restoring them afterwards.

        Raises
        ------
        ValueError
            If a value is invalid; no setting is changed then.
        """
        previous = {name: getattr(self, name) for name in settings}
        try:
            for name, value in settings.items():
                setattr(self, name, value)
            yield
        finally:
            for name, value in previous.items():
                object.__setattr__(self, name, value)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Self:
        """Return the defaults, overridden by ``GRID_DOCTOR_<NAMESPACE>__<KEY>`` variables.

        Raises
        ------
        ValueError
            If a variable holds an invalid value; the message names the variable.
        """
        settings = cls()
        for field in fields(settings):  # type: ignore[arg-type]
            var = f"GRID_DOCTOR_{cls.namespace.upper()}__{field.name.upper()}"
            if var in environ:
                try:
                    setattr(settings, field.name, environ[var].strip())
                except ValueError as error:
                    raise ValueError(f"Invalid {var}: {error}") from None
        return settings


@dataclass(slots=True)
class RemapConfig(_Namespace):
    """Remap settings.

    Attributes
    ----------
    method:
        Weight-generation method used when a function is called without
        ``method``. Environment variable: ``GRID_DOCTOR_REMAP__METHOD``.
    """

    namespace: ClassVar[str] = "remap"
    method: RemapMethod = "nearest"


remap = RemapConfig.from_env()
"""Active remap settings, e.g. ``grid_doctor.config.remap.method``."""


@contextmanager
def set(
    namespaces: Mapping[str, _Namespace | Mapping[str, Any]] | None = None,
    /,
    **kwargs: _Namespace | Mapping[str, Any],
) -> Iterator[None]:
    """Change settings of several namespaces inside a ``with`` block.

    Each namespace is given by name, either as a mapping of the settings to
    change or as a whole namespace object, whose every setting is applied::

        with grid_doctor.config.set(remap={"method": "conservative"}):
            ...
        with grid_doctor.config.set({"remap": RemapConfig()}):
            ...

    The namespace objects themselves are kept; only their values change and
    are restored afterwards.

    Raises
    ------
    AttributeError
        If a namespace does not exist.
    TypeError
        If a namespace object is of the wrong type.
    ValueError
        If a value is invalid; no setting is changed then.
    """
    with ExitStack() as stack:
        for name, values in {**(namespaces or {}), **kwargs}.items():
            target = globals().get(name)
            if not isinstance(target, _Namespace):
                raise AttributeError(f"Unknown config namespace {name!r}")
            if isinstance(values, _Namespace):
                if type(values) is not type(target):
                    raise TypeError(
                        f"{name} expects {type(target).__name__}, "
                        f"got {type(values).__name__}"
                    )
                values = {f.name: getattr(values, f.name) for f in fields(values)}  # type: ignore[arg-type]
            stack.enter_context(target.set(**values))
        yield
