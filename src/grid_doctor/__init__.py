"""grid-doctor — Convert lat/lon xarray datasets to HEALPix pyramids.

Your lat/lon data goes to rehab and comes out HEALed.
"""

from __future__ import annotations

from typing import Any

__version__: str = "2603.0.0"

_LAZY_IMPORTS: dict[str, str] = {
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
    """Lazily import heavy modules to keep ``import grid_doctor`` fast."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__: list[str] = [
    "__version__",
    "cached_open_dataset",
    "cached_weights",
    "compute_weights_delaunay",
    "get_latlon_resolution",
    "get_s3_options",
    "latlon_to_healpix_pyramid",
    "regrid_unstructured_to_healpix",
    "resolution_to_healpix_level",
    "save_pyramid_to_s3",
    "setup_logging",
]
