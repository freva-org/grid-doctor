"""CF spelling of a [`HealpixGrid`][grid_doctor.metadata.HealpixGrid].

Follows the ``healpix`` grid mapping of CF-1.13: a scalar grid mapping
variable with ``grid_mapping_name``, ``indexing_scheme`` and
``refinement_level`` (plus optional ``earth_radius``), and an integer
index coordinate identified by ``standard_name = "healpix_index"``.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping

from ._grid import HealpixGrid
from ._keys import CF_VERSION, GRID_MAPPING_NAME, INDEX_STANDARD_NAME

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
    return attrs


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
