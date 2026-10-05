"""Xarray Dataset utilities.

INTERNAL: NOT TO BE EXPOSED!!!
"""

from collections.abc import Mapping
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

# CF-convention unit strings written by the normalisers.
_LAT_UNITS = "degrees_north"
_LON_UNITS = "degrees_east"

_MICRO_SCALE = 1_000_000

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


def _normalize_angle_units(
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
        raise ValueError(f"Could not locate latitude/longitude coordinates. Available names are: {available}.")
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
    return any(var.attrs.get("CDI_grid_type") == "unstructured" for var in ds.data_vars.values())


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
    raise ValueError("Could not determine the source cell dimension for the unstructured grid.")


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
        raise ValueError(f"Could not determine spatial dimensions from {list(ds.dims)}.")
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
            "Unstructured grids require per-cell vertex coordinates such as 'clat_vertices'/'clon_vertices'/."
        )
    return lat_name, lon_name


def _replace_values(ds: xr.Dataset, mapping: Mapping[str, np.ndarray | xr.DataArray]) -> xr.Dataset:
    """Return *ds* with the values of variable ``name`` replaced, keeping dims and attrs."""
    for name, values in mapping.items():
        new = values if isinstance(values, xr.DataArray) else ds[name].copy(data=values)
        if name in ds.coords:
            ds = ds.assign_coords({name: new})
            continue
        ds = ds.assign({name: new})
    return ds


def _to_micro(a: FloatArray, scale: int) -> FloatArray:
    a = _to_float64(a)
    if np.isinf(a).any():
        raise ValueError("NaN/inf in input")
    return np.rint(a * scale).astype(np.int64)


def _has_degree_unit(da: xr.DataArray) -> bool:
    unit = da.attrs.get("units")
    return unit is not None and unit.lower().startswith("deg")


def _norm_lon(lon: FloatArray, *, scale: int = _MICRO_SCALE) -> FloatArray:
    """Convention lon: [-180, 180)°."""
    q = _to_micro(lon, scale)
    half = 180 * scale
    q = (q + half) % (2 * half) - half  # exact wrap to [-180, 180)
    return q / scale


def normalize_lon(lon: xr.DataArray) -> xr.DataArray:
    """Convert longiture to expected [-180, 180) range.

    Attribute `unit` is expected to be `degree`.

    Args:
        lon: xr.DataArray - longitude coordinate in degree
    Return:
        xr.DataArray with values and `unit` attribute adjusted (`degree_west`)
    """
    if not _has_degree_unit(lon) and _looks_like_radians(lon.values):
        raise ValueError(f"`{lon.name}` DataArray is not in degrees (try applying `normalize_degrees()`.")

    return xr.DataArray(_norm_lon(lon.values), dims=lon.dims, attrs=lon.attrs | {"units": _LON_UNITS})


def _norm_lat(lat: FloatArray, *, scale: int = _MICRO_SCALE) -> FloatArray:
    """Convention lat: [-90, 90]°."""
    q = _to_micro(lat, scale)
    if np.any(np.abs(q) > 90 * scale):
        raise ValueError("latitude outside [-90, 90]")
    return q / scale


def normalize_lat(lat: xr.DataArray) -> xr.DataArray:
    """Convert latitudes to expected [-90, 90] range.

    Attribute `unit` is expected to be `degree`.

    Args:
        lat: xr.DataArray - latidude coordinate in degree
    Return:
        xr.DataArray with values and `unit` attribute adjusted (`degree_north`)
    """
    if not _has_degree_unit(lat) and _looks_like_radians(lat.values):
        raise ValueError(f"`{lat.name}` DataArray is not in degrees (try applying `normalize_degrees()`.")

    return xr.DataArray(_norm_lat(lat.values), dims=lat.dims, attrs=lat.attrs | {"units": _LAT_UNITS})


def normalize_degrees(array: xr.DataArray) -> xr.DataArray:
    """Extract `units` from `xarray.DataArray` attributes and apply angle normalization (degree)."""
    d_array = array.copy(deep=True)
    if _has_degree_unit(array):
        return d_array

    source_unit: SourceUnits = "auto"
    if units := d_array.attrs.get("units"):
        if units.startswith("rad"):
            source_unit = "rad"

        elif units.startswith("deg"):
            source_unit = "deg"

    d_array = d_array.copy(data=_normalize_angle_units(d_array.values, source_unit))
    d_array.attrs["units"] = "degree"
    return d_array


def normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Return a normalized copy of the input dataset.

    Spatial dimensions and respective units are inferred and normatised.

    Args:
        ds: Source geometry dataset.

    Returns:
        ``xr.Dataset`` .

    Raises:
        ValueError: When the grid type cannot be handled or required
            vertex coordinates are missing.
    """
    lat_name, lon_name = _get_vertex_names(ds) if _is_unstructured(ds) else _get_spatial_dims(ds)


    return _replace_values(
        ds,
        {
            lat_name: normalize_lat(normalize_degrees(ds[lat_name])),
            lon_name: normalize_lon(normalize_degrees(ds[lon_name])),
        },
    )
