"""xarray accessors: ``ds.gd`` on datasets and ``dt.gd`` on pyramids.

Importing ``grid_doctor`` registers both, so that grid-doctor's functions
can be called as methods, e.g. ``ds.gd.regrid_to_healpix(level)`` or
``dt.gd.save_pyramid(path)``.  The methods are not there yet: this module
only registers the namespaces.
"""

from __future__ import annotations

import xarray as xr

ACCESSOR_NAME = "gd"
"""Attribute under which the accessors are registered."""


@xr.register_dataset_accessor(ACCESSOR_NAME)
class DatasetAccessor:
    """grid-doctor methods on an ``xarray.Dataset`` (``ds.gd``)."""

    def __init__(self, ds: xr.Dataset) -> None:  # noqa: D107
        self._ds = ds


@xr.register_datatree_accessor(ACCESSOR_NAME)
class DataTreeAccessor:
    """grid-doctor methods on an ``xarray.DataTree`` pyramid (``dt.gd``)."""

    def __init__(self, dt: xr.DataTree) -> None:  # noqa: D107
        self._dt = dt


__all__ = ["ACCESSOR_NAME", "DataTreeAccessor", "DatasetAccessor"]
