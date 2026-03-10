"""Regridding and HEALPix pyramid construction.

This module provides the core functionality for converting datasets on
regular, curvilinear, or unstructured grids (e.g. ICON) to multi-resolution
HEALPix pyramids stored as Zarr on S3.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union, cast

import healpy as hp
import numpy as np
import numpy.typing as npt
import s3fs
import xarray as xr
from easygems import remap as egr
from scipy.interpolate import griddata

from .types import RegridFunc, ZarrOptions

logger = logging.getLogger(__name__)

#: Known dimension names for unstructured grids (e.g. ICON).
_UNSTRUCTURED_DIMS: frozenset[str] = frozenset(
    {"cell", "ncells", "ncell", "nCells"}
)

#: Common latitude coordinate names (order matters — first match wins).
_LAT_NAMES: List[str] = [
    "clat",  # ICON
    "lat",
    "latitude",
    "LAT",
    "LATITUDE",
    "Latitude",
    "XLAT",
    "XLAT_M",
    "XLAT_U",
    "XLAT_V",  # WRF
    "nav_lat",
    "nav_lat_rho",  # NEMO
    "lat_rho",
    "lat_u",
    "lat_v",
    "lat_psi",  # ROMS
    "gridlat_0",
    "g0_lat_0",  # GRIB
    "yt_ocean",
    "yu_ocean",  # MOM
    "geolat",
    "geolat_t",
    "geolat_c",  # GFDL
]

#: Common longitude coordinate names (order matters — first match wins).
_LON_NAMES: List[str] = [
    "clon",  # ICON
    "lon",
    "longitude",
    "LON",
    "LONGITUDE",
    "Longitude",
    "XLONG",
    "XLONG_M",
    "XLONG_U",
    "XLONG_V",  # WRF
    "nav_lon",
    "nav_lon_rho",  # NEMO
    "lon_rho",
    "lon_u",
    "lon_v",
    "lon_psi",  # ROMS
    "gridlon_0",
    "g0_lon_0",  # GRIB
    "xt_ocean",
    "xu_ocean",  # MOM
    "geolon",
    "geolon_t",
    "geolon_c",  # GFDL
]


# ── coordinate helpers ─────────────────────────────────────────────
def _get_latlon_arrays(
    ds: xr.Dataset,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Extract latitude and longitude arrays from a dataset.

    Searches coordinates and data variables for well-known lat/lon names
    used by ICON, WRF, NEMO, ROMS, MOM, GFDL and standard conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with lat/lon coordinates.

    Returns
    -------
    lat : numpy.ndarray
        Latitude values (1-D for regular or unstructured, 2-D for curvilinear).
    lon : numpy.ndarray
        Longitude values with the same shape as *lat*.

    Raises
    ------
    ValueError
        If no recognised latitude or longitude coordinate is found.
    """
    lat: Optional[npt.NDArray[np.floating[Any]]] = None
    lon: Optional[npt.NDArray[np.floating[Any]]] = None

    for name in _LAT_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lat = np.asarray(ds[name].values, dtype=np.float64)
            break

    for name in _LON_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lon = np.asarray(ds[name].values, dtype=np.float64)
            break

    if lat is None or lon is None:
        available = sorted(set(map(str, ds.coords)) | set(map(str, ds.data_vars)))
        raise ValueError(
            f"Could not find lat/lon coordinates in dataset. "
            f"Searched for latitude names: {_LAT_NAMES[:5]}... "
            f"and longitude names: {_LON_NAMES[:5]}... "
            f"Available coords/vars: {available}"
        )

    return lat, lon


def _get_spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    """Identify the two spatial dimension names of a structured grid.

    Uses a three-pass search: exact match against known names, then
    substring match, then inference from 2-D coordinate shapes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect.

    Returns
    -------
    y_dim : str
        Name of the meridional (latitude-like) dimension.
    x_dim : str
        Name of the zonal (longitude-like) dimension.

    Raises
    ------
    ValueError
        If the spatial dimensions cannot be determined.
    """
    y_candidates = [
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
    ]
    x_candidates = [
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
    ]

    y_dim: Optional[str] = None
    x_dim: Optional[str] = None

    # Pass 1: exact match
    for dim in ds.dims:
        dl = str(dim).lower()
        if y_dim is None and dl in [c.lower() for c in y_candidates]:
            y_dim = str(dim)
        elif x_dim is None and dl in [c.lower() for c in x_candidates]:
            x_dim = str(dim)

    # Pass 2: substring match
    if y_dim is None or x_dim is None:
        for dim in ds.dims:
            dl = str(dim).lower()
            if y_dim is None and any(
                c in dl for c in ["lat", "south", "eta", "nj", "yh"]
            ):
                y_dim = str(dim)
            elif x_dim is None and any(
                c in dl for c in ["lon", "west", "east", "xi", "ni", "xh"]
            ):
                x_dim = str(dim)

    # Pass 3: infer from 2-D coordinates
    if y_dim is None or x_dim is None:
        lat, _ = _get_latlon_arrays(ds)
        if lat.ndim == 2:
            for coord_name in ds.coords:
                coord = ds[coord_name]
                if coord.ndim == 2 and coord.shape == lat.shape:
                    dims = coord.dims
                    if len(dims) == 2:
                        y_dim, x_dim = str(dims[0]), str(dims[1])
                        break

    if y_dim is None or x_dim is None:
        raise ValueError(
            f"Could not identify spatial dimensions. "
            f"Found dims: {list(ds.dims)}."
        )

    return y_dim, x_dim


# ── unstructured grid helpers ──────────────────────────────────────
def _is_unstructured(ds: xr.Dataset) -> bool:
    """Check whether *ds* lives on an unstructured grid.

    Detection is based on well-known dimension names (``cell``,
    ``ncells``, …) and the ``CDI_grid_type`` variable attribute that
    ICON / CDO set on their output.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect.

    Returns
    -------
    bool
        ``True`` if the dataset appears to be on an unstructured grid.
    """
    if _UNSTRUCTURED_DIMS & set(str(d) for d in ds.dims):
        return True

    for var in ds.data_vars.values():
        if var.attrs.get("CDI_grid_type") == "unstructured":
            return True

    return False


def _get_unstructured_dim(ds: xr.Dataset) -> str:
    """Return the single spatial dimension name for an unstructured grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on an unstructured grid.

    Returns
    -------
    str
        Dimension name (e.g. ``"cell"`` or ``"ncells"``).

    Raises
    ------
    ValueError
        If the unstructured spatial dimension cannot be determined.
    """
    for dim in _UNSTRUCTURED_DIMS:
        if dim in ds.dims:
            return dim

    for name in ["clat", "lat", "latitude"]:
        if name in ds.coords and ds[name].ndim == 1:
            return str(ds[name].dims[0])

    raise ValueError(
        f"Cannot determine unstructured spatial dimension. "
        f"Available dims: {list(ds.dims)}"
    )


def _get_unstructured_resolution(ds: xr.Dataset) -> float:
    """Estimate the spatial resolution of an unstructured grid.

    Uses the cell count to derive an equivalent grid spacing:
    ``degrees(sqrt(4π / n))``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on an unstructured grid.

    Returns
    -------
    float
        Approximate resolution in degrees.
    """
    dim = _get_unstructured_dim(ds)
    n = ds.sizes[dim]
    return float(np.degrees(np.sqrt(4 * np.pi / n)))


# ── resolution / level helpers ─────────────────────────────────────
def get_latlon_resolution(ds: xr.Dataset) -> float:
    """Calculate the spatial resolution of a dataset in degrees.

    Supports regular grids (1-D lat/lon), curvilinear grids (2-D lat/lon)
    and unstructured grids (e.g. ICON triangular mesh).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with spatial coordinates.

    Returns
    -------
    float
        Approximate grid spacing in degrees.

    Examples
    --------
    >>> res = get_latlon_resolution(ds)
    >>> level = resolution_to_healpix_level(res)
    """
    if _is_unstructured(ds):
        return _get_unstructured_resolution(ds)

    lat, lon = _get_latlon_arrays(ds)

    if lat.ndim == 1:
        lat_res = float(np.abs(np.diff(lat)).min())
        lon_res = float(np.abs(np.diff(lon)).min())
    else:
        lat_res = float(np.abs(np.diff(lat, axis=0)).mean())
        lon_res = float(np.abs(np.diff(lon, axis=1)).mean())

    return min(lat_res, lon_res)


def resolution_to_healpix_level(resolution_deg: float) -> int:
    """Find the HEALPix level whose pixel spacing is just coarser than *resolution_deg*.

    HEALPix pixel spacing ≈ 58.6° / 2^level.

    Parameters
    ----------
    resolution_deg : float
        Input resolution in degrees.

    Returns
    -------
    int
        HEALPix level (``nside = 2**level``).

    Examples
    --------
    >>> resolution_to_healpix_level(0.1)
    9
    """
    level = int(np.floor(np.log2(58.6 / resolution_deg)))
    return max(0, level)


# ── weight computation ─────────────────────────────────────────────
def compute_weights_delaunay(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
) -> xr.Dataset:
    """Compute Delaunay interpolation weights to a HEALPix grid.

    Builds a Delaunay triangulation of the source grid and derives
    barycentric interpolation weights for every target HEALPix pixel
    using :func:`easygems.remap.compute_weights_delaunay`.

    The returned dataset can be persisted with
    ``weights.to_netcdf("weights.nc")`` and reloaded to skip
    recomputation on subsequent runs.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (any grid type).
    level : int
        Target HEALPix level (``nside = 2**level``).
    nest : bool, optional
        If ``True`` (default) use NESTED pixel ordering, else RING.

    Returns
    -------
    xr.Dataset
        Weight dataset compatible with :func:`easygems.remap.apply_weights`.

    See Also
    --------
    cached_weights : Caching wrapper around this function.
    regrid_unstructured_to_healpix : Apply computed weights.
    """
    nside = 2**level
    npix = hp.nside2npix(nside)

    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat: npt.NDArray[np.float64] = 90.0 - np.degrees(hp_theta)
    hp_lon: npt.NDArray[np.float64] = np.degrees(hp_phi)

    src_lat, src_lon = _get_latlon_arrays(ds)
    src_lat_flat = np.asarray(src_lat).ravel()
    src_lon_flat = np.asarray(src_lon).ravel()

    src_lon_flat = src_lon_flat % 360
    hp_lon = hp_lon % 360

    return egr.compute_weights_delaunay(  # type: ignore[no-any-return]
        points=(src_lon_flat, src_lat_flat),
        xi=(hp_lon, hp_lat),
    )


# ── regridding: unstructured ──────────────────────────────────────
def regrid_unstructured_to_healpix(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
    weights: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """Regrid an unstructured-grid dataset to HEALPix.

    Uses Delaunay-interpolation weights from
    :mod:`easygems.remap`.  This is the recommended path for ICON and
    other triangular-mesh model output.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset on an unstructured grid.
    level : int
        Target HEALPix level (``nside = 2**level``).
    nest : bool, optional
        If ``True`` (default) use NESTED pixel ordering.
    weights : xr.Dataset or None, optional
        Pre-computed weights from :func:`compute_weights_delaunay`.
        When ``None`` weights are computed on the fly.

    Returns
    -------
    xr.Dataset
        Dataset on the HEALPix grid with a ``cell`` dimension.

    Raises
    ------
    ValueError
        If no data variables contain the unstructured spatial dimension.

    See Also
    --------
    regrid_to_healpix : Regridding path for structured grids.
    """
    nside = 2**level

    if weights is None:
        weights = compute_weights_delaunay(ds, level, nest=nest)

    spatial_dim = _get_unstructured_dim(ds)

    vars_to_regrid = [
        var for var, da in ds.data_vars.items() if spatial_dim in da.dims
    ]
    if not vars_to_regrid:
        raise ValueError(
            f"No data variables with dimension '{spatial_dim}' found."
        )

    ds_spatial = ds[vars_to_regrid]

    # The input spatial dimension is replaced by the output "cell" dimension
    # which has a different size.  Exclude it so xarray doesn't try to unify
    # them (same pattern as coarsen_healpix).
    ds_hp: xr.Dataset = xr.apply_ufunc(
        egr.apply_weights,
        ds_spatial,
        kwargs=weights,
        keep_attrs=True,
        input_core_dims=[[spatial_dim]],
        output_core_dims=[["cell"]],
        exclude_dims={spatial_dim},
        output_dtypes=["f4"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"cell": weights.sizes["tgt_idx"]},
        },
    )

    ds_hp = _attach_healpix_coords(ds_hp, ds, nside, level, nest)
    return ds_hp


# ── regridding: structured ─────────────────────────────────────────
def regrid_to_healpix(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
    method: Literal["nearest", "linear", "conservative"] = "nearest",
) -> xr.Dataset:
    """Regrid a structured lat/lon dataset to HEALPix.

    Uses :func:`scipy.interpolate.griddata` for nearest-neighbour or
    linear interpolation.  For unstructured grids use
    :func:`regrid_unstructured_to_healpix` instead.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with lat/lon coordinates (regular or curvilinear).
    level : int
        HEALPix level (``nside = 2**level``).
    nest : bool, optional
        If ``True`` (default), use NESTED pixel ordering.
    method : ``{"nearest", "linear"}``, optional
        Interpolation method passed to :func:`scipy.interpolate.griddata`.

    Returns
    -------
    xr.Dataset
        Dataset on the HEALPix grid with a ``cell`` dimension.

    See Also
    --------
    regrid_unstructured_to_healpix : For ICON and other unstructured meshes.
    """
    nside = 2**level
    npix = hp.nside2npix(nside)

    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    src_lat, src_lon = _get_latlon_arrays(ds)
    y_dim, x_dim = _get_spatial_dims(ds)

    if src_lat.ndim == 1:
        src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)
    else:
        src_lat_2d = src_lat
        src_lon_2d = src_lon

    src_lon_2d = src_lon_2d % 360
    hp_lon = hp_lon % 360

    if method == "conservative":
        lat_bnds = ds["lat_bnds"].values
        lon_bnds = ds["lon_bnds"].values % 360
        if lat_bnds.ndim == 2 and lon_bnds.ndim == 2:
            lat1 = np.deg2rad(lat_bnds[:, 0])
            lat2 = np.deg2rad(lat_bnds[:, 1])
            lon1 = np.deg2rad(lon_bnds[:, 0])
            lon2 = np.deg2rad(lon_bnds[:, 1])
            cell_area = np.abs(
                (np.sin(lat2) - np.sin(lat1))[:, None] * (lon2 - lon1)[None, :]
            )
        else:
            lat1 = np.deg2rad(lat_bnds[..., 0])
            lat2 = np.deg2rad(lat_bnds[..., 1])
            lon1 = np.deg2rad(lon_bnds[..., 0])
            lon2 = np.deg2rad(lon_bnds[..., 1])
            cell_area = np.abs((lon2 - lon1) * (np.sin(lat2) - np.sin(lat1)))

        theta = np.deg2rad(90.0 - src_lat_2d.ravel())
        phi = np.deg2rad(src_lon_2d.ravel())
        pix_index = hp.ang2pix(nside, theta, phi, nest=nest)
        flat_area = cell_area.ravel()
        regrid_core: RegridFunc

        def regrid_conservative_func(
            da: npt.NDArray[np.floating[Any]],
        ) -> npt.NDArray[np.floating[Any]]:
            values = da.ravel()
            valid = ~np.isnan(values)
            if valid.sum() == 0:
                return np.full(npix, np.nan, dtype=da.dtype)
            weighted_sum = np.bincount(
                pix_index[valid],
                weights=values[valid] * flat_area[valid],
                minlength=npix,
            )
            area_sum = np.bincount(
                pix_index[valid],
                weights=flat_area[valid],
                minlength=npix,
            )
            out = weighted_sum / area_sum
            out[area_sum == 0] = np.nan
            return out

        regrid_core = regrid_conservative_func
    else:
        src_points = np.column_stack([src_lat_2d.ravel(), src_lon_2d.ravel()])
        target_points = np.column_stack([hp_lat, hp_lon])

        def regrid_interpolate_func(
            src_2d: npt.NDArray[np.floating[Any]],
        ) -> npt.NDArray[np.floating[Any]]:
            src_values = src_2d.ravel()
            valid_mask = ~np.isnan(src_values)
            if valid_mask.sum() == 0:
                return np.full(npix, np.nan, dtype=src_2d.dtype)
            data: npt.NDArray[np.floating[Any]] = griddata(
                src_points[valid_mask],
                src_values[valid_mask],
                target_points,
                method=method,
                fill_value=np.nan,
            )
            return data

        regrid_core = regrid_interpolate_func

    regridded_vars: dict[str, xr.DataArray] = {}
    for var, da in ds.data_vars.items():
        if not {y_dim, x_dim}.issubset(da.dims):
            logger.info(
                "Skipping variable %s — missing dims (%s, %s)",
                var,
                y_dim,
                x_dim,
            )
            continue

        regridded_vars[str(var)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                regrid_core,
                ds[var],
                input_core_dims=[[y_dim, x_dim]],
                output_core_dims=[["cell"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={"output_sizes": {"cell": npix}},
            ),
        )

    ds_hp = xr.Dataset(regridded_vars)
    ds_hp = _attach_healpix_coords(ds_hp, ds, nside, level, nest)
    return ds_hp


# ── shared helpers ─────────────────────────────────────────────────
def _attach_healpix_coords(
    ds_hp: xr.Dataset,
    ds_src: xr.Dataset,
    nside: int,
    level: int,
    nest: bool,
) -> xr.Dataset:
    """Attach HEALPix coordinates, CRS and metadata to a regridded dataset.

    Parameters
    ----------
    ds_hp : xr.Dataset
        Regridded dataset with a ``cell`` dimension but no coordinates yet.
    ds_src : xr.Dataset
        Original source dataset (attributes are copied from here).
    nside : int
        HEALPix ``nside`` parameter.
    level : int
        HEALPix refinement level.
    nest : bool
        Whether NESTED pixel ordering is used.

    Returns
    -------
    xr.Dataset
        *ds_hp* enriched with coordinates and attributes.
    """
    npix = hp.nside2npix(nside)
    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    ds_hp.attrs |= ds_src.attrs
    ds_hp = ds_hp.assign_coords(
        cell=np.arange(npix),
        latitude=("cell", hp_lat),
        longitude=("cell", hp_lon),
    )
    ds_hp.attrs["healpix_nside"] = nside
    ds_hp.attrs["healpix_level"] = level
    ds_hp.attrs["healpix_order"] = "nested" if nest else "ring"

    crs = xr.DataArray(
        name="crs",
        data=np.nan,
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": nside,
            "healpix_order": "nest" if nest else "ring",
        },
    )
    ds_hp = ds_hp.assign_coords(crs=crs)
    return ds_hp


# ── coarsening & pyramid ──────────────────────────────────────────
def coarsen_healpix(ds: xr.Dataset, target_level: int) -> xr.Dataset:
    """Coarsen a HEALPix dataset to a lower resolution.

    Uses :func:`healpy.ud_grade` to average pixels down from the
    current level to *target_level*.

    Parameters
    ----------
    ds : xr.Dataset
        HEALPix dataset with a ``cell`` dimension and ``healpix_nside``
        in its attributes.
    target_level : int
        Target HEALPix level (must be lower than the current level).

    Returns
    -------
    xr.Dataset
        Coarsened dataset.

    Raises
    ------
    ValueError
        If *target_level* does not result in a lower resolution.
    """
    current_nside: int = ds.attrs["healpix_nside"]
    target_nside = 2**target_level

    if target_nside >= current_nside:
        raise ValueError(
            f"target_level ({target_level}) must result in lower resolution "
            f"than current (nside={current_nside})"
        )

    npix_target = hp.nside2npix(target_nside)
    order: str = ds.attrs.get("healpix_order", "nested")
    is_nested = order in ("nested", "nest")

    hp_theta, hp_phi = hp.pix2ang(
        target_nside, np.arange(npix_target), nest=is_nested
    )
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    coarsened_vars: dict[str, xr.DataArray] = {}
    for var in map(str, ds.data_vars):
        da = ds[str(var)]
        if "cell" not in da.dims:
            coarsened_vars[str(var)] = da
            continue

        da = da.chunk({"cell": -1})

        def _ud_grade(
            x: npt.NDArray[np.floating[Any]],
        ) -> npt.NDArray[np.floating[Any]]:
            return hp.ud_grade(  # type: ignore[no-any-return]
                x,
                target_nside,
                order_in="NESTED" if is_nested else "RING",
                order_out="NESTED" if is_nested else "RING",
            )

        coarsened_vars[var] = xr.apply_ufunc(
            _ud_grade,
            da,
            input_core_dims=[["cell"]],
            output_core_dims=[["cell"]],
            exclude_dims={"cell"},
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"output_sizes": {"cell": npix_target}},
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )

    ds_coarse = xr.Dataset(coarsened_vars, attrs=ds.attrs.copy())
    ds_coarse = ds_coarse.assign_coords(
        cell=np.arange(npix_target),
        latitude=("cell", hp_lat),
        longitude=("cell", hp_lon),
    )
    ds_coarse.attrs["healpix_nside"] = target_nside
    ds_coarse.attrs["healpix_level"] = target_level

    crs = xr.DataArray(
        name="crs",
        data=np.nan,
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": target_nside,
            "healpix_order": order,
        },
    )
    ds_coarse = ds_coarse.assign_coords(crs=crs)
    return ds_coarse


def create_healpix_pyramid(
    ds: xr.Dataset,
    max_level: int,
    min_level: int = 0,
    **kwargs: Any,
) -> Dict[int, xr.Dataset]:
    """Create a multi-resolution HEALPix pyramid.

    Regrids *ds* to the finest level and then repeatedly coarsens down
    to *min_level*.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (regular, curvilinear, or unstructured).
    max_level : int
        Highest resolution HEALPix level.
    min_level : int, optional
        Lowest resolution level (default ``0``).
    **kwargs : Any
        Forwarded to :func:`regrid_to_healpix` or
        :func:`regrid_unstructured_to_healpix`.

    Returns
    -------
    dict[int, xr.Dataset]
        Mapping of level → dataset.
    """
    pyramid: dict[int, xr.Dataset] = {}

    logger.info(
        "Regridding to HEALPix level %d (NSIDE=%d, npix=%d)",
        max_level,
        2**max_level,
        12 * 4**max_level,
    )
    if _is_unstructured(ds):
        ds_hp = regrid_unstructured_to_healpix(ds, max_level, **kwargs)
    else:
        ds_hp = regrid_to_healpix(ds, max_level, **kwargs)
    pyramid[max_level] = ds_hp

    current_ds = ds_hp
    for level in range(max_level - 1, min_level - 1, -1):
        logger.info(
            "Coarsening to level %d (NSIDE=%d, npix=%d)",
            level,
            2**level,
            12 * 4**level,
        )
        current_ds = coarsen_healpix(current_ds, level)
        pyramid[level] = current_ds

    return pyramid


# ── S3 I/O ─────────────────────────────────────────────────────────
def save_pyramid_to_s3(
    pyramid: Dict[int, xr.Dataset],
    s3_path: str,
    s3_options: Dict[str, str],
    mode: Literal["a", "w", "r+"] = "a",
    compute: bool = True,
    region: Union[Literal["auto"], Dict[str, slice]] = "auto",
    zarr_format: Literal[2, 3] = 2,
) -> None:
    """Save a HEALPix pyramid to S3 as Zarr stores.

    Each level is written to ``<s3_path>/level_<N>.zarr``.

    Parameters
    ----------
    pyramid : dict[int, xr.Dataset]
        Mapping of level → dataset.
    s3_path : str
        S3 prefix, e.g. ``"s3://bucket/path/to/pyramid"``.
    s3_options : dict[str, str]
        Options for :class:`s3fs.S3FileSystem` (``key``, ``secret``,
        ``endpoint_url``, …).
    mode : ``{"a", "w", "r+"}``, optional
        Write mode — ``"w"`` to overwrite, ``"a"`` (default) to append.
    compute : bool, optional
        Whether to trigger Dask computation immediately (default ``True``).
    region : ``"auto"`` or dict[str, slice], optional
        Write region for partial updates.
    zarr_format : ``{2, 3}``, optional
        Zarr format version (default ``2``).
    """
    s3_options = s3_options or {}
    fs = s3fs.S3FileSystem(**s3_options)

    for level, ds in pyramid.items():
        level_path = f"{s3_path}/level_{level}.zarr"
        logger.info("Saving level %d to %s", level, level_path)

        store = s3fs.S3Map(root=level_path, s3=fs)
        zarr_options = ZarrOptions(
            compute=compute, mode=mode, zarr_format=zarr_format
        )

        if zarr_format == 2:
            zarr_options["consolidated"] = True

        if region == "auto":
            ds.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
        else:
            to_drop = (
                set(
                    n
                    for n, v in ds.data_vars.items()
                    if set(region.keys()).isdisjoint(str(d) for d in v.dims)
                )
                | set(str(d) for d in ds.dims)
                | set(str(c) for c in ds.coords)
            )
            ds.drop_vars(to_drop, errors="ignore").isel(region).to_zarr(
                store,
                region=region,
                **zarr_options,
            )  # type: ignore[call-overload]

        if mode == "w" and not compute:
            zarr_options["mode"] = "w"
            ds[list(ds.coords)].to_zarr(
                store, **zarr_options
            )  # type: ignore[call-overload]

    logger.info("Pyramid saved to %s", s3_path)


def latlon_to_healpix_pyramid(
    ds: xr.Dataset,
    min_level: int = 0,
    max_level: Optional[int] = None,
    method: Literal["nearest", "linear", "conservative"] = "nearest",
    weights: Optional[xr.Dataset] = None,
) -> Dict[int, xr.Dataset]:
    """Full pipeline: dataset on any supported grid → HEALPix pyramid.

    Automatically detects whether the input is on a regular, curvilinear,
    or unstructured grid and chooses the appropriate regridding strategy.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with spatial coordinates.
    min_level : int, optional
        Minimum HEALPix level (default ``0``).
    max_level : int or None, optional
        Maximum HEALPix level.  When ``None`` it is derived from the
        dataset resolution.
    method : ``{"nearest", "linear", "conservative"}``, optional
        Interpolation method for **structured** grids.  Ignored for
        unstructured grids where Delaunay interpolation is always used.
    weights : xr.Dataset or None, optional
        Pre-computed Delaunay weights for unstructured grids.
        Call :func:`compute_weights_delaunay` once, save to NetCDF, and
        pass here to avoid recomputation.

    Returns
    -------
    dict[int, xr.Dataset]
        Mapping of level → dataset.

    Examples
    --------
    Structured grid (ERA5, CMIP, …):

    >>> pyramid = latlon_to_healpix_pyramid(ds_era5)

    Unstructured grid (ICON) with cached weights:

    >>> weights = compute_weights_delaunay(ds_icon, level=8)
    >>> pyramid = latlon_to_healpix_pyramid(ds_icon, weights=weights)
    """
    resolution = get_latlon_resolution(ds)
    logger.info("Dataset resolution: %.4f°", resolution)

    if max_level is None:
        max_level = resolution_to_healpix_level(resolution)
    logger.info(
        "Selected max HEALPix level: %d (NSIDE=%d)", max_level, 2**max_level
    )

    if _is_unstructured(ds):
        logger.info(
            "Detected unstructured grid, using easygems Delaunay remapping"
        )
        return create_healpix_pyramid(ds, max_level, min_level, weights=weights)
    else:
        return create_healpix_pyramid(ds, max_level, min_level, method=method)
