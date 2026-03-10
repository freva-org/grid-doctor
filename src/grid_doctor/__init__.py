"""
grid-doctor: Convert lat/lon xarray datasets to HEALPix pyramids.

Your lat/lon data goes to rehab and comes out HEALed.
"""

from typing import Any

__version__ = "2603.0.0"

_LAZY_IMPORTS = {
    "compute_weights_delaunay": ".helpers",
    "get_latlon_resolution": ".helpers",
    "latlon_to_healpix_pyramid": ".helpers",
    "regrid_unstructured_to_healpix": ".helpers",
    "resolution_to_healpix_level": ".helpers",
    "save_pyramid_to_s3": ".helpers",
    "cached_open_dataset": ".utils",
    "cached_weights": ".utils",
    "get_s3_options": ".utils",
    "setup_logging": ".log",
}


def __getattr__(name: str) -> Any:
    """Lazy import xarray dependent modules."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )  # pragma: no cover


__all__ = [
    "save_pyramid_to_s3",
    "latlon_to_healpix_pyramid",
    "cached_open_dataset",
    "compute_weights_delaunay",
    "regrid_unstructured_to_healpix",
    "cached_weights",
    "get_s3_options",
    "get_latlon_resolution",
    "resolution_to_healpix_level",
    "setup_logging",
    "__version__",
]
