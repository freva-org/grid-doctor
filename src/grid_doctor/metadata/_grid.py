"""Description of a HEALPix grid, independent of how it is stored."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Final, Literal, cast, get_args

import numpy as np

from ..types import Ellipsoid

IndexingScheme = Literal["nested", "ring"]
"""HEALPix indexing schemes grid-doctor produces.

CF also defines ``nuniq`` and ``zuniq``; grid-doctor does not write
them, so they are rejected rather than half supported."""

INDEXING_SCHEMES: Final = frozenset(get_args(IndexingScheme))

MAX_LEVEL: Final = 29
"""Deepest HEALPix level addressable with 64-bit nested indices."""

ELLIPSOIDS: Final = frozenset(get_args(Ellipsoid))


def _as_level(value: object) -> int:
    """Return *value* as a plain ``int`` level or raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"HEALPix level must be an integer, not {value!r}.")
    level = int(value)
    if not 0 <= level <= MAX_LEVEL:
        raise ValueError(f"HEALPix level must be within [0, {MAX_LEVEL}], got {level}.")
    return level


def _as_radius(value: object) -> float | None:
    """Return *value* as a positive finite radius in metres, or ``None``."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"earth_radius must be a number in metres, not {value!r}.")
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"earth_radius must be positive and finite, got {radius}.")
    return radius


def _as_ellipsoid(value: object) -> Ellipsoid | None:
    """Return *value* as a supported ellipsoid name, or ``None``."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in ELLIPSOIDS:
        raise ValueError(f"ellipsoid must be one of {sorted(ELLIPSOIDS)} or None, not {value!r}.")
    return cast(Ellipsoid, str(value))


@dataclass(frozen=True, slots=True)
class HealpixGrid:
    """A HEALPix grid: refinement level, indexing scheme and figure of the earth.

    Holds facts only; how they are spelled in a dataset is up to
    [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata].  Values
    are validated and normalised on construction, so a ``HealpixGrid``
    is always a valid grid (NumPy integers from Zarr attributes are
    accepted and converted).

    Parameters
    ----------
    level:
        Refinement level (CF ``refinement_level``), ``0`` to ``29``.
        ``nside = 2**level``.
    indexing_scheme:
        ``"nested"`` (default) or ``"ring"``.
    earth_radius:
        Radius of the spherical earth in metres, if it should be
        recorded.  Only valid on a sphere.
    ellipsoid:
        Ellipsoid the cell geometry is computed on, as a ``healpix_geo``
        name: ``"sphere"`` or ``"WGS84"``.  ``None`` leaves it
        unspecified; geometry then uses
        ``grid_doctor.config.healpix.ellipsoid``.

    Examples
    --------
    >>> grid = HealpixGrid(level=3)
    >>> grid.nside, grid.npix, grid.nest
    (8, 768, True)
    """

    level: int
    indexing_scheme: IndexingScheme = "nested"
    earth_radius: float | None = None
    ellipsoid: Ellipsoid | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the fields."""
        object.__setattr__(self, "level", _as_level(self.level))
        scheme = str(self.indexing_scheme)
        if scheme not in INDEXING_SCHEMES:
            raise ValueError(
                f"indexing_scheme must be one of {sorted(INDEXING_SCHEMES)}, not {self.indexing_scheme!r}."
            )
        object.__setattr__(self, "indexing_scheme", scheme)
        object.__setattr__(self, "earth_radius", _as_radius(self.earth_radius))
        object.__setattr__(self, "ellipsoid", _as_ellipsoid(self.ellipsoid))
        if self.earth_radius is not None and self.ellipsoid not in (None, "sphere"):
            raise ValueError(
                f"earth_radius describes a sphere and cannot be combined with ellipsoid={self.ellipsoid!r}."
            )

    @classmethod
    def from_nest(cls, level: int, *, nest: bool, ellipsoid: Ellipsoid | None = None) -> HealpixGrid:
        """Build a grid from the healpy-style ``nest`` flag."""
        return cls(level=level, indexing_scheme="nested" if nest else "ring", ellipsoid=ellipsoid)

    @property
    def nside(self) -> int:
        """Number of cells along a base-cell side, ``2**level``."""
        return 1 << self.level

    @property
    def npix(self) -> int:
        """Number of cells on the full sphere, ``12 * nside**2``."""
        return 12 << (2 * self.level)

    @property
    def nest(self) -> bool:
        """``True`` for nested indexing (healpy's ``nest`` flag)."""
        return self.indexing_scheme == "nested"
