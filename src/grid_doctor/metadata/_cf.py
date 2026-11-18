"""CF spelling of a [`HealpixGrid`][grid_doctor.metadata.HealpixGrid].

Follows the ``healpix`` grid mapping of CF-1.13: a scalar grid mapping
variable with ``grid_mapping_name``, ``indexing_scheme`` and
``refinement_level`` (plus optional ``earth_radius``), and an integer
index coordinate identified by ``standard_name = "healpix_index"``.

A WGS84 grid also carries the CF figure-of-the-earth attributes
``reference_ellipsoid_name``, ``semi_major_axis`` and
``inverse_flattening``; a grid mapping without them is spherical.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Hashable, Mapping

import numpy as np

from ..types import Ellipsoid
from ._grid import WGS84_INVERSE_FLATTENING, WGS84_SEMI_MAJOR_AXIS, HealpixGrid
from ._keys import (
    CF_VERSION,
    GRID_MAPPING_NAME,
    INDEX_STANDARD_NAME,
    WGS84_CF_NAME,
    WGS84_CF_NAMES,
)

_CF_TOKEN = re.compile(r"^CF-\d+(\.\d+)*$")

INDEX_ATTRS: Mapping[str, str] = {
    "standard_name": INDEX_STANDARD_NAME,
    "long_name": "HEALPix cell index",
    "units": "1",
}
"""Attributes of the HEALPix index coordinate."""

LAT_ATTRS: Mapping[str, str] = {
    "standard_name": "latitude",
    "long_name": "latitude of HEALPix cell centre",
    "units": "degrees_north",
}
"""Attributes of the cell-centre latitude coordinate."""

LON_ATTRS: Mapping[str, str] = {
    "standard_name": "longitude",
    "long_name": "longitude of HEALPix cell centre",
    "units": "degrees_east",
}
"""Attributes of the cell-centre longitude coordinate."""


def grid_mapping_attrs(grid: HealpixGrid) -> dict[str, str | int | float]:
    """Return the attributes of the CF grid mapping variable for *grid*.

    ``refinement_level`` is always written: CF requires it for the
    ``nested`` and ``ring`` schemes, including level 0.
    """
    attrs: dict[str, str | int | float] = {
        "grid_mapping_name": GRID_MAPPING_NAME,
        "indexing_scheme": grid.indexing_scheme,
        "refinement_level": grid.level,
    }
    if grid.earth_radius is not None:
        attrs["earth_radius"] = grid.earth_radius
    if grid.ellipsoid == "WGS84":
        attrs["reference_ellipsoid_name"] = WGS84_CF_NAME
        attrs["semi_major_axis"] = WGS84_SEMI_MAJOR_AXIS
        attrs["inverse_flattening"] = WGS84_INVERSE_FLATTENING
    return attrs


def _matches(value: object, expected: float) -> bool:
    """Whether the attribute *value* is *expected*, at the precision it was stored in."""
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return False
        if value.dtype.itemsize < 8:
            return bool(value == value.dtype.type(expected))
        return math.isclose(float(value), expected)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(value) and math.isclose(float(value), expected)


def _ellipsoid_from_mapping(attrs: Mapping[Hashable, object]) -> Ellipsoid:
    """Return the ellipsoid described by a grid mapping's figure-of-the-earth attributes.

    No such attributes means a sphere.  WGS84 is recognised by its
    ``reference_ellipsoid_name`` (any parameters given must agree) or,
    without a name, by both of its parameters.
    """
    name = attrs.get("reference_ellipsoid_name")
    axis = attrs.get("semi_major_axis")
    flattening = attrs.get("inverse_flattening")
    if name is None and axis is None and flattening is None:
        return "sphere"
    parameters = ((axis, WGS84_SEMI_MAJOR_AXIS), (flattening, WGS84_INVERSE_FLATTENING))
    given = [(value, expected) for value, expected in parameters if value is not None]
    agree = all(_matches(value, expected) for value, expected in given)
    if isinstance(name, str) and name in WGS84_CF_NAMES and agree:
        return "WGS84"
    if name is None and len(given) == 2 and agree:
        return "WGS84"
    raise ValueError(
        "Unsupported figure of the earth: "
        f"reference_ellipsoid_name={name!r}, semi_major_axis={axis!r}, inverse_flattening={flattening!r}."
    )


def is_healpix_grid_mapping(attrs: Mapping[Hashable, object]) -> bool:
    """Whether *attrs* belong to a CF HEALPix grid mapping variable."""
    return attrs.get("grid_mapping_name") == GRID_MAPPING_NAME


def grid_from_mapping(attrs: Mapping[Hashable, object]) -> HealpixGrid:
    """Parse a CF HEALPix grid mapping variable's attributes.

    Raises
    ------
    ValueError
        When *attrs* are not a HEALPix grid mapping or describe an
        invalid or unsupported grid.
    """
    if not is_healpix_grid_mapping(attrs):
        raise ValueError(f"Not a HEALPix grid mapping: grid_mapping_name={attrs.get('grid_mapping_name')!r}.")
    if "refinement_level" not in attrs:
        raise ValueError("HEALPix grid mapping has no `refinement_level`.")
    return HealpixGrid(
        level=attrs["refinement_level"],  # type: ignore[arg-type]
        indexing_scheme=str(attrs.get("indexing_scheme", "")),  # type: ignore[arg-type]
        earth_radius=attrs.get("earth_radius"),  # type: ignore[arg-type]
        ellipsoid=_ellipsoid_from_mapping(attrs),
    )


def merge_conventions(existing: object) -> str:
    """Merge the CF version into an existing ``Conventions`` attribute.

    Any ``CF-x.y`` token is replaced by the version grid-doctor writes;
    other conventions (``CMIP-6.2``, ``ACDD-1.3``, ...) are kept in
    order.
    """
    tokens = str(existing).replace(",", " ").split() if existing else []
    kept = [token for token in tokens if not _CF_TOKEN.match(token)]
    return " ".join([CF_VERSION, *kept])
