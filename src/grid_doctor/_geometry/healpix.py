"""HEALPix geometry: the ``healpix_geo`` backend, cell centres and coordinates."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import xarray as xr

from .. import config
from ..metadata import (
    CELL_DIM,
    LAT_VAR,
    LON_VAR,
    HealpixGrid,
    HealpixMetadata,
    Provenance,
)
from ..types import Ellipsoid, FloatArray
from .normalise import _canonical_lon


def _ellipsoid_or_config(ellipsoid: Ellipsoid | None) -> Ellipsoid:
    """Return *ellipsoid*, or ``grid_doctor.config.ellipsoid`` if ``None``."""
    return config.ellipsoid if ellipsoid is None else ellipsoid


def _require_healpix_geo_module(
    grid: HealpixGrid,
) -> tuple[ModuleType, dict[str, str]]:
    """Import and return the appropriate ``healpix_geo`` sub-module.

    HEALPix geometry is defined on ``grid.ellipsoid``.  The default, a perfect
    sphere, is consistent with ESMF's internal area calculation
    (``CoordSys.SPH_DEG``) and with the geocentric coordinates used
    by climate models.  The maximum latitude discrepancy relative to
    WGS84 geodetic coordinates is ~0.19° at 45° latitude, which is
    well within the tolerance of conservative remapping.

    Args:
        grid: Grid whose indexing scheme selects `healpix_geo.nested`
            or `healpix_geo.ring` and whose ellipsoid is used.

    Returns:
        ``(module, kwargs)`` where *kwargs* should be forwarded to
        ``module.vertices`` and ``module.healpix_to_lonlat``.

    Raises:
        ImportError: When `healpix-geo` is not installed.
    """
    try:
        if grid.nest:
            from healpix_geo import nested as module
        else:
            from healpix_geo import ring as module
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "healpix-geo is required to construct HEALPix polygons."
        ) from exc
    return module, {"ellipsoid": grid.ellipsoid}


# ===================================================================
# HEALPix cell centres
# ===================================================================


def _healpix_centres(grid: HealpixGrid) -> tuple[FloatArray, FloatArray]:
    """Return HEALPix cell-centre coordinates in degrees.

    Args:
        grid: HEALPix grid; its ellipsoid sets the geometry.

    Returns:
        ``(lat_deg, lon_deg)`` arrays.
    """
    module, kwargs = _require_healpix_geo_module(grid)
    ipix = np.arange(grid.npix, dtype=np.int64)
    lon_deg, lat_deg = module.healpix_to_lonlat(ipix, grid.level, **kwargs)
    return (
        np.asarray(lat_deg, dtype=np.float64),
        _canonical_lon(np.asarray(lon_deg, dtype=np.float64)),
    )


# ===================================================================
# HEALPix coordinate attachment
# ===================================================================


def _attach_healpix_coords(
    ds_hp: xr.Dataset,
    grid: HealpixGrid,
    *,
    method: str | None = None,
) -> xr.Dataset:
    """Attach dense HEALPix cell coordinates and grid-doctor metadata.

    Adds the ``cell`` index and cell-centre ``latitude``/``longitude``
    coordinates; everything else (``crs``, CF attributes, provenance)
    is written by [`HealpixMetadata.apply`][grid_doctor.metadata.HealpixMetadata.apply].
    """
    lat_deg, lon_deg = _healpix_centres(grid)
    ds_hp = ds_hp.assign_coords(
        {
            CELL_DIM: np.arange(grid.npix, dtype=np.int64),
            LAT_VAR: (CELL_DIM, lat_deg),
            LON_VAR: (CELL_DIM, lon_deg),
        }
    )
    return HealpixMetadata(grid, Provenance(method=method)).apply(ds_hp)


def resolution_to_healpix_level(resolution_deg: float) -> int:
    """Convert an approximate source resolution to a HEALPix level.

    The heuristic uses a characteristic HEALPix pixel spacing of about
    ``58.6° / 2**level``.

    Parameters
    ----------
    resolution_deg:
        Approximate source resolution in degrees.

    Returns
    -------
    int
        Suggested HEALPix level.

    Examples
    --------
    ```python
    level = resolution_to_healpix_level(0.25)
    ```
    """
    if resolution_deg <= 0.0:
        raise ValueError("resolution_deg must be positive.")
    level = int(np.round(np.log2(58.6 / resolution_deg)))
    return max(0, level)
