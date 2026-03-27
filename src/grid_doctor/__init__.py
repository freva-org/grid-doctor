"""grid-doctor.

Convert regular, curvilinear, and unstructured geoscience grids to HEALPix
pyramids and reusable HEALPix weight files.
"""

from __future__ import annotations

from typing import Any

__version__ = "2603.1.0"

_LAZY_IMPORTS: dict[str, str] = {
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


def __getattr__(name: str) -> Any:
    """Lazily import heavy modules to keep `import grid_doctor` fast."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
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
