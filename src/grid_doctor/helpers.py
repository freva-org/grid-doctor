"""
grid-doctor: Convert lat/lon xarray datasets to HEALPix pyramids and save to S3.
"""

import logging
from typing import Any, Literal, Optional, Tuple, Union

import healpy as hp
import numpy as np
import s3fs
import xarray as xr
from easygems import remap as egr
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def get_latlon_resolution(ds: xr.Dataset) -> float:
    """Calculate spatial resolution of a lat/lon dataset in degrees.

    Parameters
    ----------
    ds:
        Dataset with lat/lon coordinates (regular, curvilinear, or unstructured).

    Returns
    -------
    float:
        Approximate resolution (grid spacing) in degrees.
    """
    if _is_unstructured(ds):
        return _get_unstructured_resolution(ds)

    # Try to get lat/lon - could be 1D coordinates or 2D arrays
    lat, lon = _get_latlon_arrays(ds)

    if lat.ndim == 1:
        lat_res = np.abs(np.diff(lat)).min()
        lon_res = np.abs(np.diff(lon)).min()
    else:
        # For 2D grids, estimate resolution from neighboring cells
        lat_res = np.abs(np.diff(lat, axis=0)).mean()
        lon_res = np.abs(np.diff(lon, axis=1)).mean()

    return float(min(lat_res, lon_res))


def _get_latlon_arrays(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Extract latitude and longitude arrays from dataset.

    Handles both regular grids (1D lat/lon) and curvilinear grids (2D lat/lon).
    Supports various naming conventions (lat/lon, latitude/longitude, XLAT/XLONG,
    nav_lat/nav_lon, lat_rho/lon_rho, etc.)

    Parameters
    ----------
    ds:
        Dataset with lat/lon coordinates.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        (latitude, longitude) arrays.
    """
    # Common latitude coordinate names
    lat_names = [
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

    # Common longitude coordinate names
    lon_names = [
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

    lat = None
    lon = None

    # Search in coordinates
    for name in lat_names:
        if name in ds.coords:
            lat = ds[name].values
            break

    for name in lon_names:
        if name in ds.coords:
            lon = ds[name].values
            break

    # If not found in coords, search in data variables (some models store lat/lon as variables)
    if lat is None:
        for name in lat_names:
            if name in ds.data_vars:
                lat = ds[name].values
                break

    if lon is None:
        for name in lon_names:
            if name in ds.data_vars:
                lon = ds[name].values
                break

    if lat is None or lon is None:
        available = list(ds.coords) + list(ds.data_vars)
        raise ValueError(
            f"Could not find lat/lon coordinates in dataset. "
            f"Searched for latitude names: {lat_names[:5]}... "
            f"and longitude names: {lon_names[:5]}... "
            f"Available coords/vars: {available}"
        )

    return lat, lon


def _get_spatial_dims(ds: xr.Dataset) -> Tuple[str, str]:
    """Identify the spatial dimension names in the dataset.

    Parameters
    ----------
    ds:
        Dataset to inspect.

    Returns
    -------
    Tuple[str, str]:
        (y_dim, x_dim) dimension names.
    """
    # Common y dimension names (in priority order)
    y_candidates = [
        "rlat",
        "lat",
        "latitude",
        "y",
        "j",
        "nj",
        "south_north",
        "south_north_stag",  # WRF
        "eta_rho",
        "eta_u",
        "eta_v",
        "eta_psi",  # ROMS
        "yh",
        "yq",  # MOM
        "nj",
        "njp1",
    ]

    # Common x dimension names (in priority order)
    x_candidates = [
        "rlon",
        "lon",
        "longitude",
        "x",
        "i",
        "ni",
        "west_east",
        "west_east_stag",  # WRF
        "xi_rho",
        "xi_u",
        "xi_v",
        "xi_psi",  # ROMS
        "xh",
        "xq",  # MOM
        "ni",
        "nip1",
    ]

    y_dim = None
    x_dim = None

    # First pass: exact match
    for dim in ds.dims:
        dim_lower = dim.lower()
        if y_dim is None and dim_lower in [c.lower() for c in y_candidates]:
            y_dim = dim
        elif x_dim is None and dim_lower in [c.lower() for c in x_candidates]:
            x_dim = dim

    # Second pass: partial match (e.g., "lat" in "gridlat_0")
    if y_dim is None or x_dim is None:
        for dim in ds.dims:
            dim_lower = dim.lower()
            if y_dim is None and any(
                c in dim_lower for c in ["lat", "south", "eta", "nj", "yh"]
            ):
                y_dim = dim
            elif x_dim is None and any(
                c in dim_lower for c in ["lon", "west", "east", "xi", "ni", "xh"]
            ):
                x_dim = dim

    # Third pass: if we have 2D lat/lon coords, infer dims from them
    if y_dim is None or x_dim is None:
        lat, lon = _get_latlon_arrays(ds)
        if lat.ndim == 2:
            # Get dimension names from a 2D coordinate
            for coord_name in ds.coords:
                coord = ds[coord_name]
                if coord.ndim == 2 and coord.shape == lat.shape:
                    dims = coord.dims
                    if len(dims) == 2:
                        y_dim, x_dim = dims[0], dims[1]
                        break

    if y_dim is None or x_dim is None:
        raise ValueError(
            f"Could not identify spatial dimensions. Found dims: {list(ds.dims)}. "
            f"Expected y-dimensions like: {y_candidates[:5]}... "
            f"and x-dimensions like: {x_candidates[:5]}..."
        )

    return y_dim, x_dim


# Known dimension names for unstructured grids (e.g. ICON).
_UNSTRUCTURED_DIMS = {"cell", "ncells", "ncell", "nCells"}


def _is_unstructured(ds: xr.Dataset) -> bool:
    """Check whether *ds* lives on an unstructured grid.

    Detection is based on well-known dimension names and CDI metadata
    attributes that ICON / CDO set on their output.
    """
    if _UNSTRUCTURED_DIMS & set(ds.dims):
        return True

    for var in ds.data_vars.values():
        if var.attrs.get("CDI_grid_type") == "unstructured":
            return True

    return False


def _get_unstructured_dim(ds: xr.Dataset) -> str:
    """Return the single spatial dimension name for an unstructured grid."""
    for dim in _UNSTRUCTURED_DIMS:
        if dim in ds.dims:
            return dim

    # Fall back: look for the dimension that lat is indexed on.
    lat_names = ["clat", "lat", "latitude"]
    for name in lat_names:
        if name in ds.coords and ds[name].ndim == 1:
            return ds[name].dims[0]

    raise ValueError(
        f"Cannot determine unstructured spatial dimension. "
        f"Available dims: {list(ds.dims)}"
    )


def _get_unstructured_resolution(ds: xr.Dataset) -> float:
    """Estimate the spatial resolution (in degrees) of an unstructured grid
    from the number of cells.

    On the unit sphere the mean cell area is ``4π / n`` steradians, so the
    equivalent grid spacing in degrees is ``degrees(sqrt(4π / n))``.
    """
    dim = _get_unstructured_dim(ds)
    n = ds.sizes[dim]
    return float(np.degrees(np.sqrt(4 * np.pi / n)))


def compute_weights_delaunay(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
) -> xr.Dataset:
    """Compute Delaunay interpolation weights from the source grid of *ds*
    to a HEALPix grid at the given *level*.

    The returned weight dataset can be saved with ``weights.to_netcdf(...)``
    and reloaded later to avoid recomputation.

    Parameters
    ----------
    ds:
        Input dataset (any grid type).
    level:
        Target HEALPix level (``nside = 2**level``).
    nest:
        If ``True`` use NESTED ordering, else RING.

    Returns
    -------
    xr.Dataset:
        Weight dataset compatible with :func:`easygems.remap.apply_weights`.
    """
    nside = 2**level
    npix = hp.nside2npix(nside)

    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    src_lat, src_lon = _get_latlon_arrays(ds)
    src_lat = np.asarray(src_lat).ravel()
    src_lon = np.asarray(src_lon).ravel()

    # Normalise longitudes to [0, 360)
    src_lon = src_lon % 360
    hp_lon = hp_lon % 360

    return egr.compute_weights_delaunay(
        points=(src_lon, src_lat),
        xi=(hp_lon, hp_lat),
    )


def regrid_unstructured_to_healpix(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
    weights: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    """Regrid an unstructured-grid dataset (e.g. ICON) to HEALPix using
    Delaunay-interpolation weights from :mod:`easygems.remap`.

    Parameters
    ----------
    ds:
        Input dataset on an unstructured grid.
    level:
        Target HEALPix level (``nside = 2**level``).
    nest:
        If ``True`` use NESTED ordering.
    weights:
        Pre-computed weights (from :func:`compute_weights_delaunay`).
        When ``None`` they are computed on the fly.

    Returns
    -------
    xr.Dataset:
        Dataset on the HEALPix grid with a ``cell`` dimension.
    """
    nside = 2**level
    npix = hp.nside2npix(nside)

    if weights is None:
        weights = compute_weights_delaunay(ds, level, nest=nest)

    spatial_dim = _get_unstructured_dim(ds)

    # Select only variables that have the spatial dimension.
    vars_to_regrid = [var for var, da in ds.data_vars.items() if spatial_dim in da.dims]
    if not vars_to_regrid:
        raise ValueError(f"No data variables with dimension '{spatial_dim}' found.")
    ds_spatial = ds[vars_to_regrid]

    ds_hp = xr.apply_ufunc(
        egr.apply_weights,
        ds_spatial,
        kwargs=weights,
        keep_attrs=True,
        input_core_dims=[[spatial_dim]],
        output_core_dims=[["cell"]],
        output_dtypes=["f4"],
        vectorize=True,
        exclude_dims={spatial_dim},
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"cell": weights.sizes["tgt_idx"]},
        },
    )

    # Attach HEALPix coordinates and metadata
    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    ds_hp = ds_hp.assign_coords(
        cell=np.arange(npix),
        latitude=("cell", hp_lat),
        longitude=("cell", hp_lon),
    )

    ds_hp.attrs |= ds.attrs
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


def resolution_to_healpix_level(resolution_deg: float) -> int:
    """Find HEALPix level with resolution just coarser than input.

    HEALPix resolution ≈ 58.6° / 2^level

    Parameters
    ----------
    resolution_deg:
        Input resolution in degrees.

    Returns
    -------
    int:
        HEALPix level (nside = 2^level).
    """
    level = int(np.floor(np.log2(58.6 / resolution_deg)))
    return max(0, level)


def regrid_to_healpix(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
    method: Literal["nearest", "linear"] = "nearest",
) -> xr.Dataset:
    """Regrid lat/lon dataset to HEALPix using healpy and scipy.

    Parameters
    ----------
    ds:
        Input dataset with lat/lon coordinates (regular or curvilinear).
    level:
        HEALPix level (nside = 2^level).
    nest:
        If True, use NESTED ordering. If False, use RING ordering.
    method:
        Interpolation method: 'nearest' or 'linear'.

    Returns
    -------
    xr.Dataset:
        Dataset on HEALPix grid with 'cell' dimension.
    """
    nside = 2**level
    npix = hp.nside2npix(nside)

    # Get HEALPix pixel center coordinates
    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)  # Convert colatitude to latitude
    hp_lon = np.degrees(hp_phi)  # Convert to degrees [0, 360)

    # Get source lat/lon
    src_lat, src_lon = _get_latlon_arrays(ds)

    # Get spatial dimension names
    y_dim, x_dim = _get_spatial_dims(ds)

    # Ensure lat/lon are 2D for interpolation
    if src_lat.ndim == 1:
        src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)
    else:
        src_lat_2d = src_lat
        src_lon_2d = src_lon

    # Normalize longitudes to [0, 360) for consistent interpolation
    src_lon_2d = src_lon_2d % 360
    hp_lon = hp_lon % 360

    if method == "conservative":
        # Compute cell areas
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

        def regrid_conservative_func(da):
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
        # Flatten source coordinates for scipy.griddata
        src_points = np.column_stack([src_lat_2d.ravel(), src_lon_2d.ravel()])
        target_points = np.column_stack([hp_lat, hp_lon])

        def regrid_interpolate_func(src_2d):
            src_values = src_2d.ravel()
            valid_mask = ~np.isnan(src_values)

            if valid_mask.sum() == 0:
                return np.full(npix, np.nan, dtype=src_2d.dtype)

            return griddata(
                src_points[valid_mask],
                src_values[valid_mask],
                target_points,
                method=method,
                fill_value=np.nan,
            )

        regrid_core = regrid_interpolate_func

    regridded_vars = {}
    # Regrid each variable individually
    # since they might have different dtypes
    for var, da in ds.data_vars.items():
        if not {y_dim, x_dim}.issubset(da.dims):
            logger.info(
                "Skipping regridding for %s (%s, %s) not in its dimensions",
                var,
                y_dim,
                x_dim,
            )
            continue

        regridded_vars[var] = xr.apply_ufunc(
            regrid_core,
            ds[var],
            input_core_dims=[[y_dim, x_dim]],
            output_core_dims=[["cell"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=ds[var].dtype,
            dask_gufunc_kwargs={
                "output_sizes": {"cell": npix},
            },
        )

    # Build output dataset (only regridded variables)
    ds_hp = xr.Dataset(regridded_vars)

    # Copy relevant global attributes
    ds_hp.attrs |= ds.attrs

    # Add HEALPix coordinates
    ds_hp = ds_hp.assign_coords(
        cell=np.arange(npix),
        latitude=("cell", hp_lat),
        longitude=("cell", hp_lon),
    )

    # Add HEALPix metadata
    ds_hp.attrs["healpix_nside"] = nside
    ds_hp.attrs["healpix_level"] = level
    ds_hp.attrs["healpix_order"] = "nested" if nest else "ring"

    # Add CRS coordinate (CF-convention style)
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


def coarsen_healpix(ds: xr.Dataset, target_level: int) -> xr.Dataset:
    """Coarsen HEALPix dataset to lower resolution using healpy.ud_grade.

    Parameters
    ----------
    ds:
        HEALPix dataset with 'cell' dimension.
    target_level:
        Target HEALPix level (must be lower than current).

    Returns
    -------
    xr.Dataset:
        Coarsened dataset.
    """
    current_nside = ds.attrs["healpix_nside"]
    target_nside = 2**target_level

    if target_nside >= current_nside:
        raise ValueError(
            f"target_level ({target_level}) must result in lower resolution "
            f"than current (nside={current_nside})"
        )

    npix_target = hp.nside2npix(target_nside)
    order = ds.attrs.get("healpix_order", "nested")
    is_nested = order in ("nested", "nest")

    # Get new pixel coordinates
    hp_theta, hp_phi = hp.pix2ang(target_nside, np.arange(npix_target), nest=is_nested)
    hp_lat = 90 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    # Coarsen each variable
    coarsened_vars = {}
    for var in ds.data_vars:
        da = ds[var]

        if "cell" not in da.dims:
            coarsened_vars[var] = da
            continue

        da = da.chunk({"cell": -1})

        def _ud_grade(x):
            return hp.ud_grade(
                x,
                target_nside,
                order_in="NESTED" if is_nested else "RING",
                order_out="NESTED" if is_nested else "RING",
            )

        coarse_da = xr.apply_ufunc(
            _ud_grade,
            da,
            input_core_dims=[["cell"]],
            output_core_dims=[["cell"]],
            exclude_dims={"cell"},
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {"cell": npix_target},
            },
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )

        coarsened_vars[var] = coarse_da

    # Build output dataset
    ds_coarse = xr.Dataset(coarsened_vars, attrs=ds.attrs.copy())
    ds_coarse = ds_coarse.assign_coords(
        cell=np.arange(npix_target),
        latitude=("cell", hp_lat),
        longitude=("cell", hp_lon),
    )

    # Update metadata
    ds_coarse.attrs["healpix_nside"] = target_nside
    ds_coarse.attrs["healpix_level"] = target_level

    # Update CRS coordinate
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
    **kwargs,
) -> dict[int, xr.Dataset]:
    """Create HEALPix pyramid from max_level down to min_level.

    Parameters
    ----------
    ds:
        Input dataset (regular, curvilinear, or unstructured).
    max_level:
        Highest resolution HEALPix level.
    min_level:
        Lowest resolution level (default 0).

    Returns
    -------
    dict:
        Mapping of level -> xr.Dataset.
    """
    pyramid = {}

    # Convert to highest resolution HEALPix
    logger.info(
        "Regridding to HEALPix level $d (NSIDE=%d, npix=%d)",
        max_level,
        2**max_level,
        12 * 4**max_level,
    )
    if _is_unstructured(ds):
        ds_hp = regrid_unstructured_to_healpix(ds, max_level, **kwargs)
    else:
        ds_hp = regrid_to_healpix(ds, max_level, **kwargs)
    pyramid[max_level] = ds_hp

    # Coarsen down to min_level
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


def save_pyramid_to_s3(
    pyramid: dict[int, xr.Dataset],
    s3_path: str,
    s3_options: dict[str, Any],
    mode: Literal["a", "w", "r+"] = "a",
    compute: bool = True,
    region: Union[Literal["auto"], dict[str, slice]] = "auto",
    zarr_format: Literal[2, 3] = 2,
) -> None:
    """Save HEALPix pyramid to S3 as Zarr stores.

    Parameters
    ----------
    pyramid
        Dictionary mapping level -> dataset.
    s3_path
        S3 path like 's3://bucket/path/to/pyramid'.
    s3_options
        Options for s3fs (key, secret, endpoint_url, etc.).
    mode
        Write mode: 'w' to overwrite, 'a' to append.
    zarr_format
        Zarr format for saving data.
    """
    s3_options = s3_options or {}

    fs = s3fs.S3FileSystem(**s3_options)

    for level, ds in pyramid.items():
        level_path = f"{s3_path}/level_{level}.zarr"
        logger.info("Saving level %d to %s", level, level_path)

        store = s3fs.S3Map(root=level_path, s3=fs)
        kwargs = {"zarr_format": zarr_format}
        if zarr_format == 2:
            kwargs["consolidated"] = True

        if region == "auto":
            ds.to_zarr(store, mode=mode, compute=compute, **kwargs)
        else:
            to_drop = (
                set(
                    n
                    for n, v in ds.data_vars.items()
                    if region.keys().isdisjoint(v.dims)
                )
                | set(ds.dims)
                | set(ds.coords)
            )
            ds.drop_vars(to_drop, errors="warn").isel(region).to_zarr(
                store, mode=mode, compute=compute, region=region, **kwargs
            )

        if mode == "w" and not compute:
            ds[list(ds.coords)].to_zarr(store, mode="r+", **kwargs)

    logger.info("Pyramid saved to %s", s3_path)


def latlon_to_healpix_pyramid(
    ds: xr.Dataset,
    min_level: int = 0,
    max_level: Optional[int] = None,
    method: Literal["nearest", "linear", "conservative"] = "nearest",
    weights: Optional[xr.Dataset] = None,
    grid: Optional[xr.Dataset] = None,
) -> dict[int, xr.Dataset]:
    """Full pipeline: dataset -> HEALPix pyramid.

    Works with regular, curvilinear, and unstructured (e.g. ICON) grids.
    For unstructured grids the *method* parameter is ignored (Delaunay
    interpolation from :mod:`easygems.remap` is used instead).

    Parameters
    ----------
    ds:
        Input dataset with lat/lon coordinates.
    min_level:
        Minimum HEALPix level (default 0).
    max_level:
        Maximum HEALPix level. If None, computed from dataset resolution.
    method:
        Interpolation method for structured grids: 'nearest', 'linear',
        or 'conservative'.  Ignored for unstructured grids.
    weights:
        Pre-computed Delaunay weights for unstructured grids.  When
        ``None`` they are computed on the fly.  Recommended for large
        grids — call :func:`compute_weights_delaunay` once and reuse.

    Returns
    -------
    dict:
        The generated pyramid.
    """
    # Step 1: Calculate resolution
    resolution = get_latlon_resolution(ds)
    logger.info("Dataset resolution: %d°", round(resolution, 4))

    # Step 2: Find appropriate HEALPix level
    if max_level is None:
        max_level = resolution_to_healpix_level(resolution)
    logger.info("Selected max HEALPix level: %d (NSIDE=%d)", max_level, 2**max_level)

    # Step 3: Create pyramid
    if _is_unstructured(ds):
        logger.info("Detected unstructured grid, using easygems Delaunay remapping")
        return create_healpix_pyramid(ds, max_level, min_level, weights=weights)
    else:
        return create_healpix_pyramid(ds, max_level, min_level, method=method)
