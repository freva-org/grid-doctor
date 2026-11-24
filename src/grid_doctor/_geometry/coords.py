"""Find the latitude, longitude, vertex and spatial dimensions of a dataset.

Holds the well-known coordinate and dimension names that grid-doctor
recognises, and the functions that look them up in a dataset.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr

from ..types import FloatArray

# ---------------------------------------------------------------------------
# Well-known coordinate and dimension names
# ---------------------------------------------------------------------------

_UNSTRUCTURED_DIMS: frozenset[str] = frozenset({"cell", "ncells", "ncell", "nCells"})
"""Dimension names that signal an unstructured source grid."""


_LAT_NAMES: tuple[str, ...] = (
    "clat",
    "lat",
    "latitude",
    "LAT",
    "LATITUDE",
    "Latitude",
    "XLAT",
    "XLAT_M",
    "XLAT_U",
    "XLAT_V",
    "nav_lat",
    "nav_lat_rho",
    "lat_rho",
    "lat_u",
    "lat_v",
    "lat_psi",
    "gridlat_0",
    "g0_lat_0",
    "yt_ocean",
    "yu_ocean",
    "geolat",
    "geolat_t",
    "geolat_c",
)
"""Priority-ordered latitude variable names recognised by the backend."""


_LON_NAMES: tuple[str, ...] = (
    "clon",
    "lon",
    "longitude",
    "LON",
    "LONGITUDE",
    "Longitude",
    "XLONG",
    "XLONG_M",
    "XLONG_U",
    "XLONG_V",
    "nav_lon",
    "nav_lon_rho",
    "lon_rho",
    "lon_u",
    "lon_v",
    "lon_psi",
    "gridlon_0",
    "g0_lon_0",
    "xt_ocean",
    "xu_ocean",
    "geolon",
    "geolon_t",
    "geolon_c",
)
"""Priority-ordered longitude variable names recognised by the backend."""


_Y_CANDIDATES: tuple[str, ...] = (
    "rlat",
    "lat",
    "latitude",
    "y",
    "j",
    "nj",
    "south_north",
    "south_north_stag",
    "eta_rho",
    "eta_u",
    "eta_v",
    "eta_psi",
    "yh",
    "yq",
    "njp1",
)
"""Lower-cased dimension names considered as latitude / y axes."""


_X_CANDIDATES: tuple[str, ...] = (
    "rlon",
    "lon",
    "longitude",
    "x",
    "i",
    "ni",
    "west_east",
    "west_east_stag",
    "xi_rho",
    "xi_u",
    "xi_v",
    "xi_psi",
    "xh",
    "xq",
    "nip1",
)
"""Lower-cased dimension names considered as longitude / x axes."""


# ===================================================================
# Dataset coordinate introspection
# ===================================================================


def _get_latlon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Return the names of the latitude and longitude variables in *ds*.

    The function searches coordinate and variables names
    using the priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor._geometry.coords._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor._geometry.coords._LON_NAMES].

    Args:
        ds: Source dataset.

    Returns:
        ``(lat_name, lon_name)`` as strings.
    Raises:
        ValueError: When no latitude/longitude coordinates are present.
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
        raise ValueError(
            "Could not locate latitude/longitude coordinates. "
            f"Available names are: {available}."
        )

    return lat_name, lon_name


def _get_latlon_arrays(ds: xr.Dataset) -> tuple[FloatArray, FloatArray]:
    """Extract latitude and longitude arrays from *ds*.

    The function searches coordinates and data variables using the
    priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor._geometry.coords._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor._geometry.coords._LON_NAMES].

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
    [`_UNSTRUCTURED_DIMS`][grid_doctor._geometry.coords._UNSTRUCTURED_DIMS]
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


def _get_unstructured_vertices(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """Return longitude and latitude vertex variable names for *ds*.

    Recognises the ICON-style names used by existing inputs, common CF-style
    names, and arbitrary CF bounds variables referenced from the cell-centre
    longitude and latitude coordinates.
    """
    for lon_name, lat_name in (
        ("clon_vertices", "clat_vertices"),
        ("lon_vertices", "lat_vertices"),
        ("lon_bnds", "lat_bnds"),
    ):
        if lon_name in ds and lat_name in ds:
            return lon_name, lat_name

    for lon_center, lat_center in (("lon", "lat"), ("clon", "clat")):
        if lon_center not in ds or lat_center not in ds:
            continue
        lon_bounds = ds[lon_center].attrs.get("bounds")
        lat_bounds = ds[lat_center].attrs.get("bounds")
        if (
            isinstance(lon_bounds, str)
            and isinstance(lat_bounds, str)
            and lon_bounds in ds
            and lat_bounds in ds
        ):
            return lon_bounds, lat_bounds

    return None, None


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
    """Return ``(lat_name, lon_name)`` of the per-cell vertex coordinates.

    Recognises ICON-style names, ``lat_bnds``/``lon_bnds`` and CF bounds
    variables referenced from the cell-centre coordinates. Names are
    always returned as a matching pair.

    Raises:
        ValueError: When no vertex coordinates are present.
    """
    for lat_name, lon_name in (
        ("clat_vertices", "clon_vertices"),
        ("lat_vertices", "lon_vertices"),
        ("lat_bnds", "lon_bnds"),
    ):
        if lat_name in ds and lon_name in ds:
            return lat_name, lon_name

    for lat_c, lon_c in (("lat", "lon"), ("clat", "clon")):
        if lat_c not in ds or lon_c not in ds:
            continue
        lat_b = ds[lat_c].attrs.get("bounds")
        lon_b = ds[lon_c].attrs.get("bounds")
        if isinstance(lat_b, str) and isinstance(lon_b, str) and lat_b in ds and lon_b in ds:
            return lat_b, lon_b

    raise ValueError(
        "Could not find per-cell vertex coordinates "
        "(e.g. 'clat_vertices'/'clon_vertices' or CF 'bounds')."
    )


# ===================================================================
# Low-level coordinate helpers (fully vectorised)
# ===================================================================


def _to_float64(values: npt.ArrayLike) -> FloatArray:
    """Cast *values* to a contiguous float64 array.

    Args:
        values: Anything accepted by :func:`numpy.asarray`.

    Returns:
        Float64 NumPy array.
    """
    return np.asarray(values, dtype=np.float64)


# ===================================================================
# Resolution estimation
# ===================================================================


def get_latlon_resolution(ds: xr.Dataset) -> float:
    """Estimate the horizontal resolution of *ds* in degrees.

    Parameters
    ----------
    ds:
        Dataset on a regular lon/lat grid, a curvilinear grid, or an
        unstructured grid.

    Returns
    -------
    float
        Approximate grid spacing in degrees.

    Examples
    --------
    ```python
    resolution = get_latlon_resolution(ds)
    level = resolution_to_healpix_level(resolution)
    ```
    """
    if _is_unstructured(ds):
        cell_dim = _get_unstructured_dim(ds)
        n_cells = ds.sizes[cell_dim]
        return float(np.degrees(np.sqrt(4.0 * np.pi / float(n_cells))))

    lat, lon = _get_latlon_arrays(ds)
    if lat.ndim == 1:
        lat_res = float(np.nanmin(np.abs(np.diff(lat))))
        lon_res = float(np.nanmin(np.abs(np.diff(lon))))
    else:
        lat_res = float(np.nanmean(np.abs(np.diff(lat, axis=0))))
        lon_res = float(np.nanmean(np.abs(np.diff(lon, axis=1))))
    return min(lat_res, lon_res)
