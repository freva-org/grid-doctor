"""Xarray Dataset utilities.

INTERNAL: NOT TO BE EXPOSED!!!
"""

import logging
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

logger = logging.getLogger(__name__)

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

def _get_latlon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the names latitude and longitude from *ds*.

    The function searches coordinates and data variables using the
    priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor.misc._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor.misc._LON_NAMES].

    Args:
        ds: Source dataset.

    Returns:
        ``(lat_name, lon_name)`` as strings.
    Raises:
        ValueError: When no latidure/longitude coordinates are present.
    """
    lat_name: str | None = None
    lon_name: str | None = None
    for name in _LAT_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lat_name = name
            break

    for name in _LON_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        available = sorted({*map(str, ds.coords), *map(str, ds.data_vars)})
        raise ValueError(f"Could not locate latitude/longitude coordinates. Available names are: {available}.")

    return lat_name, lon_name


def _get_latlon_arrays(ds: xr.Dataset) -> tuple[FloatArray, FloatArray]:
    """Extract latitude and longitude arrays from *ds*.

    The function searches coordinates and data variables using the
    priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor.misc._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor.misc._LON_NAMES].

    Args:
        ds: Source dataset.

    Returns:
        ``(lat, lon)`` as float64 NumPy arrays.

    Raises:
        ValueError: When no recognised coordinate names are found.
    """
    lat_name, lon_name = _get_latlon_names(ds)
    return _to_float64(ds[lat_name].values), _to_float64(ds[lon_name].values)


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


def _to_micro(a: FloatArray, scale: int) -> tuple[FloatArray, FloatArray]:
    """Return *a* as integer multiples of ``1 / scale`` plus its NaN mask.

    NaN (e.g. padding vertices) cannot be cast to an integer, so it is
    returned as 0 and must be restored by the caller via the mask.
    """
    a = _to_float64(a)
    if np.isinf(a).any():
        raise ValueError("inf in input")
    nan = np.isnan(a)
    return np.rint(np.where(nan, 0.0, a) * scale).astype(np.int64), nan


def _has_degree_unit(da: xr.DataArray) -> bool:
    unit = da.attrs.get("units")
    return unit is not None and unit.lower().startswith("deg")


def _norm_lon(lon: FloatArray, *, scale: int = _MICRO_SCALE) -> FloatArray:
    """Convention lon: [-180, 180)°."""
    q, nan = _to_micro(lon, scale)
    half = 180 * scale
    q = (q + half) % (2 * half) - half  # exact wrap to [-180, 180)
    return np.where(nan, np.nan, q / scale)


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
    q, nan = _to_micro(lat, scale)
    if np.any(np.abs(q) > 90 * scale):
        raise ValueError("latitude outside [-90, 90]")
    return np.where(nan, np.nan, q / scale)


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


def normalize_degrees(array: xr.DataArray, source_units: SourceUnits = "auto") -> xr.DataArray:
    """Return a copy of *array* in degrees.

    Args:
        array: Angular coordinate.
        source_units: Angular unit convention of *array*. With ``"auto"``
            the ``units`` attribute decides: values are converted only when
            it says radians. Missing or unrecognised units are assumed to be
            degrees, with a warning.

    Returns:
        ``xr.DataArray`` in degrees.
    """
    if source_units == "auto":
        units = str(array.attrs.get("units", "")).lower()
        if units.startswith("rad"):
            source_units = "rad"
        else:
            if not units.startswith("deg"):
                logger.warning(
                    "No angular units found for `%s` (units=%r); assuming degrees.",
                    array.name,
                    array.attrs.get("units"),
                )
            source_units = "deg"

    d_array = array.copy(deep=True)
    if source_units == "rad":
        d_array = d_array.copy(data=np.rad2deg(_to_float64(d_array.values)))
    d_array.attrs["units"] = "degree"
    return d_array


def normalize_dataset(ds: xr.Dataset, source_units: SourceUnits = "auto") -> xr.Dataset:
    """Return a normalized copy of the input dataset.

    Latitude/longitude coordinates and their units are inferred and
    normalised. On unstructured grids, the cell vertices are normalised
    as well when present.

    Args:
        ds: Source geometry dataset.
        source_units: Angular unit convention of the coordinates; see
            ``normalize_degrees``.

    Returns:
        ``xr.Dataset`` .

    Raises:
        ValueError: When no latitude/longitude coordinates are present.
    """
    names = [_get_latlon_names(ds)]
    if _is_unstructured(ds):
        try:
            names.append(_get_vertex_names(ds))
        except ValueError:
            pass  # vertices are optional

    mapping: dict[str, xr.DataArray] = {}
    for lat_name, lon_name in names:
        mapping[lat_name] = normalize_lat(normalize_degrees(ds[lat_name], source_units))
        mapping[lon_name] = normalize_lon(normalize_degrees(ds[lon_name], source_units))
    return _replace_values(ds, mapping)
