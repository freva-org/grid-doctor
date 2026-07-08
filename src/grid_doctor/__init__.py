"""grid-doctor.

Convert regular, curvilinear, and unstructured geoscience grids to HEALPix
pyramids and reusable HEALPix weight files.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "2604.0.0"

_SUBMODULES: dict[str, str] = {
    "helpers": ".helpers",
    "log": ".log",
    "remap": ".remap",
    "select": ".select",
    "swath": ".swath",
    "utils": ".utils",
    "cli": ".cli",
}


_ATTRS: dict[str, str] = {
    "apply_weight_file": ".remap",
    "attach_cell_coords": ".select",
    "bin_to_healpix": ".swath",
    "compute_healpix_weights": ".remap",
    "chunk_for_target_store_size": ".utils",
    "coarsen_healpix": ".helpers",
    "create_healpix_pyramid": ".helpers",
    "cached_open_dataset": ".utils",
    "cached_weights": ".utils",
    "get_latlon_resolution": ".helpers",
    "get_s3_options": ".utils",
    "latlon_to_healpix_pyramid": ".helpers",
    "regrid_to_healpix": ".remap",
    "regrid_unstructured_to_healpix": ".remap",
    "resolution_to_healpix_level": ".helpers",
    "save_pyramid": ".helpers",
    "select_bbox": ".select",
    "select_cells": ".select",
    "select_cone": ".select",
    "setup_logging": ".log",
    "sparse_to_dense": ".swath",
}


if TYPE_CHECKING:
    from . import helpers, log, remap, select, swath, utils
    from .helpers import (
        coarsen_healpix,
        create_healpix_pyramid,
        get_latlon_resolution,
        latlon_to_healpix_pyramid,
        resolution_to_healpix_level,
        save_pyramid,
    )
    from .log import setup_logging
    from .remap import (
        apply_weight_file,
        compute_healpix_weights,
        regrid_to_healpix,
        regrid_unstructured_to_healpix,
    )
    from .select import (
        attach_cell_coords,
        select_bbox,
        select_cells,
        select_cone,
    )
    from .swath import bin_to_healpix, sparse_to_dense
    from .utils import (
        cached_open_dataset,
        cached_weights,
        chunk_for_target_store_size,
        get_s3_options,
    )


def __getattr__(name: str) -> Any:
    """Lazily load public submodules and exported attributes."""
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name], __name__)
        globals()[name] = module
        return module

    if name in _ATTRS:
        module = import_module(_ATTRS[name], __name__)
        obj = getattr(module, name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public module attributes for introspection."""
    return sorted(set(globals()) | set(_SUBMODULES) | set(_ATTRS))


__all__ = [
    "__version__",
    "helpers",
    "log",
    "remap",
    "select",
    "swath",
    "utils",
    "apply_weight_file",
    "attach_cell_coords",
    "bin_to_healpix",
    "cached_open_dataset",
    "cached_weights",
    "chunk_for_target_store_size",
    "coarsen_healpix",
    "compute_healpix_weights",
    "create_healpix_pyramid",
    "get_latlon_resolution",
    "get_s3_options",
    "latlon_to_healpix_pyramid",
    "regrid_to_healpix",
    "regrid_unstructured_to_healpix",
    "resolution_to_healpix_level",
    "save_pyramid",
    "select_bbox",
    "select_cells",
    "select_cone",
    "setup_logging",
    "sparse_to_dense",
]
