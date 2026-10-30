"""Attributes of a multiscales pyramid stored as one datatree."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

from ._keys import (
    COARSEN_MODE_ATTR,
    COARSENED_FROM_ATTR,
    IMPLICIT_COORDS_ATTR,
    LEVEL_MAX_ATTR,
    LEVEL_MIN_ATTR,
    MIN_VALID_FRACTION_ATTR,
    SOURCE_ATTRS_ATTR,
    SPARSE_ATTR,
)

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


def pyramid_root_attrs(
    levels: Iterable[int],
    level_attrs: Mapping[Hashable, object],
) -> dict[str, object]:
    """Return the attributes of a multiscales datatree's root group.

    Parameters
    ----------
    levels:
        The pyramid's HEALPix levels.
    level_attrs:
        Global attributes of one level; everything that describes the
        whole pyramid (source attributes, conventions, version, method)
        is copied as strings, per-level attributes are left out.
    """
    resolved = [int(level) for level in levels]
    return {
        LEVEL_MIN_ATTR: min(resolved),
        LEVEL_MAX_ATTR: max(resolved),
        SOURCE_ATTRS_ATTR: {str(key): str(value) for key, value in level_attrs.items() if key not in _PER_LEVEL_ATTRS},
    }
