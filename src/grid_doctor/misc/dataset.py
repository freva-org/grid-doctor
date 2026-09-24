"""Xarray Dataset utilities.

INTERNAL: NOT TO BE EXPOSED!!!
"""

from typing import Any

import numpy as np
import xarray as xr

from ..types import (
    FloatArray,
    SourceUnits,
)
from .constants import (
    _LAT_NAMES,
    _LON_NAMES,
    _UNSTRUCTURED_DIMS,
    _X_CANDIDATES,
    _Y_CANDIDATES,
)

# ===================================================================
# Low-level coordinate helpers (fully vectorised)
# ===================================================================


def _to_float64(values: Any) -> FloatArray:
    """Cast *values* to a contiguous float64 array.

    Args:
        values: Anything accepted by :func:`numpy.asarray`.

    Returns:
        Float64 NumPy array.
    """
    return np.asarray(values, dtype=np.float64)


def _canonical_lon(lon_deg: FloatArray) -> FloatArray:
    """Map longitudes into the range ``[-180, 180)``.

    Args:
        lon_deg: Longitude values in degrees.

    Returns:
        Canonicalised longitude array (same shape as input).
    """
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _looks_like_radians(values: FloatArray) -> bool:
    """Heuristic test whether *values* are in radians.

    The check passes when the maximum absolute finite value is at most
    ``2 * pi + 1e-6``.

    Args:
        values: Coordinate array to inspect.

    Returns:
        *True* when the values appear to be in radians.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(float(np.nanmax(np.abs(finite))) <= (2.0 * np.pi + 1e-6))


def _normalise_angle_units(
    values: FloatArray,
    units: SourceUnits,
) -> FloatArray:
    """Convert *values* to degrees according to *units*.

    When ``units="auto"`` the function applies
    [`_looks_like_radians`][grid_doctor.remap_backend._looks_like_radians]
    and converts if the heuristic fires.

    Args:
        values: Coordinate array.
        units: Unit convention (``"deg"``, ``"rad"``, or ``"auto"``).

    Returns:
        Coordinate array guaranteed to be in degrees.
    """
    if units == "deg":
        return values.astype(np.float64, copy=False)
    if units == "rad":
        return np.rad2deg(values)
    if _looks_like_radians(values):
        return np.rad2deg(values)
    return values.astype(np.float64, copy=False)


# ===================================================================
# Dataset coordinate introspection
# ===================================================================


def _get_latlon_arrays(ds: xr.Dataset) -> tuple[FloatArray, FloatArray]:
    """Extract latitude and longitude arrays from *ds*.

    The function searches coordinates and data variables using the
    priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor.remap_backend._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor.remap_backend._LON_NAMES].

    Args:
        ds: Source dataset.

    Returns:
        ``(lat, lon)`` as float64 NumPy arrays.

    Raises:
        ValueError: When no recognised coordinate names are found.
    """
    lat: FloatArray | None = None
    lon: FloatArray | None = None

    for name in _LAT_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lat = _to_float64(ds[name].values)
            break
    for name in _LON_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lon = _to_float64(ds[name].values)
            break

    if lat is None or lon is None:
        available = sorted({*map(str, ds.coords), *map(str, ds.data_vars)})
        raise ValueError(
            "Could not locate latitude/longitude coordinates. "
            f"Available names are: {available}."
        )
    return lat, lon


def _is_unstructured(ds: xr.Dataset) -> bool:
    """Check whether *ds* looks like an unstructured grid.

    The test succeeds when any dimension name is in
    [`_UNSTRUCTURED_DIMS`][grid_doctor.remap_backend._UNSTRUCTURED_DIMS]
    or when any variable carries the ``CDI_grid_type=unstructured``
    attribute.

    Args:
        ds: Dataset to test.

    Returns:
        *True* when the dataset appears to be unstructured.
    """
    if _UNSTRUCTURED_DIMS & {str(dim) for dim in ds.dims}:
        return True
    return any(
        var.attrs.get("CDI_grid_type") == "unstructured"
        for var in ds.data_vars.values()
    )


def _get_unstructured_dim(ds: xr.Dataset) -> str:
    """Return the name of the unstructured cell dimension in *ds*.

    Args:
        ds: Unstructured dataset.

    Returns:
        Dimension name.

    Raises:
        ValueError: When the cell dimension cannot be determined.
    """
    for dim in _UNSTRUCTURED_DIMS:
        if dim in ds.dims:
            return dim
    lat, _ = _get_latlon_arrays(ds)
    if lat.ndim == 1:
        for name in _LAT_NAMES:
            if name in ds and ds[name].ndim == 1:
                return str(ds[name].dims[0])
    raise ValueError(
        "Could not determine the source cell dimension for the unstructured grid."
    )


def _get_spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    """Return the ``(y_dim, x_dim)`` spatial dimension names.

    For curvilinear grids where the dimension names are not standard,
    the function falls back to inspecting 2-D coordinate shapes.

    Args:
        ds: Source dataset.

    Returns:
        ``(y_dim, x_dim)`` names.

    Raises:
        ValueError: When the spatial dimensions cannot be identified.
    """
    y_dim: str | None = None
    x_dim: str | None = None

    for dim in ds.dims:
        dim_name = str(dim)
        dim_lower = dim_name.lower()
        if y_dim is None and dim_lower in _Y_CANDIDATES:
            y_dim = dim_name
        elif x_dim is None and dim_lower in _X_CANDIDATES:
            x_dim = dim_name

    if y_dim is None or x_dim is None:
        lat, _ = _get_latlon_arrays(ds)
        if lat.ndim == 2:
            for coord in ds.coords.values():
                if coord.ndim == 2 and coord.shape == lat.shape:
                    dims = tuple(map(str, coord.dims))
                    if len(dims) == 2:
                        return dims[0], dims[1]

    if y_dim is None or x_dim is None:
        raise ValueError(
            f"Could not determine spatial dimensions from {list(ds.dims)}."
        )
    return y_dim, x_dim


def _get_vertex_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the names of the per-cell vertex coordinates of an unstructured grid.

    Raises:
        ValueError: When no vertex coordinates are present.
    """
    lat_name = "clat_vertices" if "clat_vertices" in ds else "lat_vertices"
    lon_name = "clon_vertices" if "clon_vertices" in ds else "lon_vertices"
    if lat_name not in ds or lon_name not in ds:
        raise ValueError(
            "Unstructured grids require per-cell vertex "
            "coordinates such as "
            "'clat_vertices'/'clon_vertices'/."
        )
    return lat_name, lon_name


def _replace_values(ds: xr.Dataset, name: str, values: np.ndarray) -> xr.Dataset:
    """Return *ds* with the values of variable ``name`` replaced, keeping dims and attrs."""
    new = ds[name].copy(data=values)
    if name in ds.coords:
        return ds.assign_coords({name: new})
    return ds.assign({name: new})
