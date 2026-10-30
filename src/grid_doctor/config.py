"""Library-wide settings.

Each setting has a default that can be overridden by an environment variable
``GRID_DOCTOR_<NAMESPACE>__<KEY>`` (read at import) or by assignment, e.g.
``grid_doctor.config.healpix.ellipsoid = "WGS84"``.
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

ENV_VAR_PREFIX = "GRID_DOCTOR"
"""Prefix of the environment variables, ``<PREFIX>_<NAMESPACE>__<KEY>``."""


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
            var = f"{ENV_VAR_PREFIX}_{cls.namespace.upper()}__{field.name.upper()}"
            if var in environ:
                try:
                    setattr(settings, field.name, environ[var].strip())
                except ValueError as error:
                    raise ValueError(f"Invalid {var}: {error}") from None
        return settings


@dataclass(slots=True)
class HealpixConfig(_Namespace):
    """HEALPix geometry settings.

    Attributes
    ----------
    ellipsoid:
        Ellipsoid on which HEALPix cell centres and vertices are computed,
        passed to ``healpix_geo``: ``"sphere"`` (consistent with ESMF's
        spherical area calculation) or a named ellipsoid such as
        ``"WGS84"``. Environment variable: ``GRID_DOCTOR_HEALPIX__ELLIPSOID``.
    """

    namespace: ClassVar[str] = "healpix"
    ellipsoid: str = "sphere"


healpix = HealpixConfig.from_env()
"""Active HEALPix settings, e.g. ``grid_doctor.config.healpix.ellipsoid``."""
