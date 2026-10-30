"""Attributes of the root group of a multiscales datatree."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import xarray as xr

from ._keys import (
    COARSEN_MODE_ATTR,
    COARSENED_FROM_ATTR,
    IMPLICIT_COORDS_ATTR,
    MIN_VALID_FRACTION_ATTR,
    SPARSE_ATTR,
)
from ._multiscales import PyramidMetadata

_PER_LEVEL_ATTRS = frozenset(
    {
        COARSENED_FROM_ATTR,
        COARSEN_MODE_ATTR,
        MIN_VALID_FRACTION_ATTR,
        IMPLICIT_COORDS_ATTR,
        SPARSE_ATTR,
    }
)
"""Global attributes that describe one level and not the whole pyramid."""


def pyramid_root_attrs(pyramid: Mapping[int, xr.Dataset]) -> dict[Hashable, object]:
    """Return the attributes of a multiscales datatree's root group.

    The root carries everything that describes the whole pyramid: the
    finest level's global attributes (source attributes, ``Conventions``,
    grid-doctor version and method) without the per-level ones, plus the
    ``zarr_conventions`` and ``multiscales`` layout of
    [`PyramidMetadata`][grid_doctor.metadata.PyramidMetadata].

    Parameters
    ----------
    pyramid:
        ``{level: dataset}`` as written to the level groups.
    """
    finest = pyramid[max(pyramid)]
    attrs: dict[Hashable, object] = {key: value for key, value in finest.attrs.items() if key not in _PER_LEVEL_ATTRS}
    attrs.update(PyramidMetadata.from_pyramid(pyramid).to_attrs())
    return attrs
