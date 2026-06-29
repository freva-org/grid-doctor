"""GRIB to HEALPix mapping boundary for ERA5/ERA5-Land.

The implementation will use grid_doctor to build remapping weights and convert
opened GRIB datasets into HEALPix pyramids. Batch execution hooks, including
reflow decorators, should attach around functions in this module once the
processing graph is fixed.
"""

from typing import Any


def map_grib_to_healpix(*args: Any, **kwargs: Any) -> Any:
    """Placeholder for the grid_doctor-backed GRIB to HEALPix mapper."""

    raise NotImplementedError("GRIB to HEALPix mapping is not defined yet.")
