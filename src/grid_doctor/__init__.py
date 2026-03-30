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
    "utils": ".utils",
}


_ATTRS: dict[str, str] = {
    "apply_weight_file": ".remap",
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
    "save_pyramid_to_s3": ".helpers",
    "setup_logging": ".log",
}


if TYPE_CHECKING:
    from . import helpers, log, remap, utils
    from .helpers import (
        coarsen_healpix,
        create_healpix_pyramid,
        get_latlon_resolution,
        latlon_to_healpix_pyramid,
        resolution_to_healpix_level,
        save_pyramid_to_s3,
    )
    from .log import setup_logging
    from .remap import (
        apply_weight_file,
        compute_healpix_weights,
        regrid_to_healpix,
        regrid_unstructured_to_healpix,
    )
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
    "utils",
    "apply_weight_file",
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
    "save_pyramid_to_s3",
    "setup_logging",
]
