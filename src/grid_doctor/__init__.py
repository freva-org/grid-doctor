"""grid-doctor.

Convert regular, curvilinear, and unstructured geoscience grids to HEALPix
pyramids and reusable HEALPix weight files.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import xarray_accessor  # noqa: F401  registers on import

__version__ = "2604.0.0"

_SUBMODULES: dict[str, str] = {
    "cli": ".cli",
    "config": ".config",
    "io": ".io",
    "metadata": ".metadata",
    "pyramid": ".pyramid",
    "remap": ".remap",
    "select": ".select",
    "swath": ".swath",
    "types": ".types",
    "weights": ".weights",
    "xarray_accessor": ".xarray_accessor",
}


_ATTRS: dict[str, str] = {
    "apply_weight_file": ".remap",
    "attach_cell_coords": ".select",
    "bin_to_healpix": ".swath",
    "cached_weights": ".weights",
    "check_pyramid_store": ".io",
    "chunk_for_target_store_size": ".io",
    "coarsen_healpix": ".pyramid",
    "compute_healpix_weights": ".remap",
    "create_healpix_pyramid": ".pyramid",
    "DatasetAccessor": ".xarray_accessor",
    "get_latlon_resolution": "._geometry.coords",
    "get_s3_options": ".io",
    "init_pyramid_store": ".io",
    "is_current": ".io",
    "is_task_done": ".io",
    "load_manifest": ".io",
    "regrid_to_healpix": ".remap",
    "resolution_to_healpix_level": "._geometry.healpix",
    "save_pyramid": ".io",
    "select_bbox": ".select",
    "select_cells": ".select",
    "select_cone": ".select",
    "setup_logging": "._log",
    "sparse_to_dense": ".swath",
    "write_pyramid_task": ".io",
    "PyramidManifest": ".io",
    "StoreReport": ".io",
    "WriteTask": ".io",
}


if TYPE_CHECKING:
    from . import (
        cli,
        config,
        io,
        metadata,
        pyramid,
        remap,
        select,
        swath,
        types,
        weights,
    )
    from ._geometry.coords import (
        get_latlon_resolution,
    )
    from ._geometry.healpix import (
        resolution_to_healpix_level,
    )
    from ._log import (
        setup_logging,
    )
    from .io import (
        PyramidManifest,
        StoreReport,
        WriteTask,
        check_pyramid_store,
        chunk_for_target_store_size,
        get_s3_options,
        init_pyramid_store,
        is_current,
        is_task_done,
        load_manifest,
        save_pyramid,
        write_pyramid_task,
    )
    from .pyramid import (
        coarsen_healpix,
        create_healpix_pyramid,
    )
    from .remap import (
        apply_weight_file,
        compute_healpix_weights,
        regrid_to_healpix,
    )
    from .select import (
        attach_cell_coords,
        select_bbox,
        select_cells,
        select_cone,
    )
    from .swath import (
        bin_to_healpix,
        sparse_to_dense,
    )
    from .weights import (
        cached_weights,
    )


def __getattr__(name: str) -> object:
    """Lazily load public submodules and exported attributes.

    Type checkers read the real types from ``__init__.pyi``.
    """
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
    "cli",
    "config",
    "io",
    "metadata",
    "pyramid",
    "remap",
    "select",
    "swath",
    "types",
    "weights",
    "xarray_accessor",
    "apply_weight_file",
    "attach_cell_coords",
    "bin_to_healpix",
    "cached_weights",
    "check_pyramid_store",
    "chunk_for_target_store_size",
    "coarsen_healpix",
    "compute_healpix_weights",
    "create_healpix_pyramid",
    "get_latlon_resolution",
    "get_s3_options",
    "init_pyramid_store",
    "is_current",
    "is_task_done",
    "load_manifest",
    "regrid_to_healpix",
    "resolution_to_healpix_level",
    "save_pyramid",
    "select_bbox",
    "select_cells",
    "select_cone",
    "setup_logging",
    "sparse_to_dense",
    "write_pyramid_task",
    "PyramidManifest",
    "StoreReport",
    "WriteTask",
]
