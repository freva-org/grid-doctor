"""Library-wide settings.

Each setting has a default that can be overridden by an environment variable
``GRID_DOCTOR_<NAMESPACE>__<KEY>`` (read at import) or by assignment, e.g.
``grid_doctor.config.remap.method = "conservative"``. An explicit function
argument always wins.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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


class _Namespace:
    """Base for a group of settings; subclasses are slotted dataclasses.

    ``Literal``-typed fields only accept their listed values. Values from the
    environment are read as stripped strings.
    """

    __slots__ = ()
    namespace: ClassVar[str]

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate ``Literal`` settings before assigning them."""
        hint = get_type_hints(type(self)).get(name)
        if get_origin(hint) is Literal and value not in get_args(hint):
            raise ValueError(f"{self.namespace}.{name} must be one of {get_args(hint)}, got {value!r}")
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
