"""
grid-doctor: Convert lat/lon xarray datasets to HEALPix pyramids and save to S3.
"""

from typing import Any, Literal, Optional

import healpy as hp
import numpy as np
import s3fs
import xarray as xr


def get_latlon_resolution(ds: xr.Dataset) -> float:
    """Calculate spatial resolution of a lat/lon dataset in degrees.

    Parameters
    ----------
    ds:
        Dataset with 'lat' and 'lon' coordinates.

    Returns
    -------
    float:
        Minimum resolution (finest grid spacing) in degrees.
    """
    lat = ds["lat"].values
    lon = ds["lon"].values

    lat_res = np.abs(np.diff(lat)).min()
    lon_res = np.abs(np.diff(lon)).min()

    return float(min(lat_res, lon_res))


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
    """Regrid lat/lon dataset to HEALPix using healpy.

    Parameters
    ----------
    ds:
        Input dataset with lat/lon coordinates.
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
    # pix2ang returns (theta, phi) where theta is colatitude [0, pi] and phi is longitude [0, 2pi]
    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
    hp_lat = 90.0 - np.degrees(hp_theta)  # Convert colatitude to latitude
    hp_lon = np.degrees(hp_phi)  # Convert to degrees

    # Adjust longitude to match input dataset convention
    # If input uses [-180, 180], adjust HEALPix longitudes accordingly
    lon_vals = ds["lon"].values
    if lon_vals.min() < 0:
        hp_lon = np.where(hp_lon > 180, hp_lon - 360, hp_lon)

    # Create DataArrays for interpolation targets
    target_lat = xr.DataArray(hp_lat, dims=["cell"])
    target_lon = xr.DataArray(hp_lon, dims=["cell"])

    # Regrid each variable using xarray's interp
    regridded_vars = {}
    for var in ds.data_vars:
        da = ds[var]

        # Check if variable has lat/lon dimensions
        if "lat" not in da.dims or "lon" not in da.dims:
            regridded_vars[var] = da
            continue

        # Interpolate to HEALPix pixel centers
        regridded = da.interp(
            lat=target_lat,
            lon=target_lon,
            method=method,
        )
        regridded_vars[var] = regridded

    # Build output dataset
    ds_hp = xr.Dataset(regridded_vars, attrs=ds.attrs.copy())

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
    hp_theta, hp_phi = hp.pix2ang(
        target_nside, np.arange(npix_target), nest=is_nested
    )
    hp_lat = 90 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    # Coarsen each variable
    coarsened_vars = {}
    for var in ds.data_vars:
        da = ds[var]

        if "cell" not in da.dims:
            coarsened_vars[var] = da
            continue

        other_dims = [d for d in da.dims if d != "cell"]

        if not other_dims:
            # Simple 1D case
            coarse_data = hp.ud_grade(
                da.values,
                target_nside,
                order_in="NESTED" if is_nested else "RING",
                order_out="NESTED" if is_nested else "RING",
            )
            coarsened_vars[var] = xr.DataArray(
                coarse_data, dims=["cell"], attrs=da.attrs
            )
        else:
            # Multi-dimensional: iterate over other dimensions
            da_stacked = da.stack(other=other_dims)
            coarse_maps = []

            for i in range(da_stacked.sizes["other"]):
                coarse_map = hp.ud_grade(
                    da_stacked.isel(other=i).values,
                    target_nside,
                    order_in="NESTED" if is_nested else "RING",
                    order_out="NESTED" if is_nested else "RING",
                )
                coarse_maps.append(coarse_map)

            coarse_stacked = xr.DataArray(
                np.stack(coarse_maps, axis=-1),
                dims=["cell", "other"],
                coords={"other": da_stacked.coords["other"]},
            ).unstack("other")

            coarsened_vars[var] = coarse_stacked.transpose(*da.dims).assign_attrs(
                da.attrs
            )

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
) -> dict[int, xr.Dataset]:
    """Create HEALPix pyramid from max_level down to min_level.

    Parameters
    ----------
    ds:
        Input lat/lon dataset.
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
    print(
        f"Regridding to HEALPix level {max_level} "
        f"(NSIDE={2**max_level}, npix={12 * 4**max_level})"
    )
    ds_hp = regrid_to_healpix(ds, max_level)
    pyramid[max_level] = ds_hp

    # Coarsen down to min_level
    current_ds = ds_hp
    for level in range(max_level - 1, min_level - 1, -1):
        print(
            f"Coarsening to level {level} "
            f"(NSIDE={2**level}, npix={12 * 4**level})"
        )
        current_ds = coarsen_healpix(current_ds, level)
        pyramid[level] = current_ds

    return pyramid


def save_pyramid_to_s3(
    pyramid: dict[int, xr.Dataset],
    s3_path: str,
    s3_options: dict[str, Any],
    mode: Literal["a", "w"] = "a",
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
    """
    s3_options = s3_options or {}

    fs = s3fs.S3FileSystem(**s3_options)

    for level, ds in pyramid.items():
        level_path = f"{s3_path}/level_{level}"
        print(f"Saving level {level} to {level_path}")

        store = s3fs.S3Map(root=level_path, s3=fs)
        ds.to_zarr(store, mode=mode)

    print(f"Pyramid saved to {s3_path}")


def latlon_to_healpix_pyramid(
    ds: xr.Dataset,
    min_level: int = 0,
    max_level: Optional[int] = None,
) -> dict[int, xr.Dataset]:
    """Full pipeline: lat/lon dataset -> HEALPix pyramid.

    Parameters
    ----------
    ds:
        Input dataset with lat/lon coordinates.
    min_level:
        Minimum HEALPix level (default 0).
    max_level:
        Maximum HEALPix level. If None, computed from dataset resolution.

    Returns
    -------
    dict:
        The generated pyramid.
    """
    # Step 1: Calculate resolution
    resolution = get_latlon_resolution(ds)
    print(f"Dataset resolution: {resolution:.4f}°")

    # Step 2: Find appropriate HEALPix level
    if max_level is None:
        max_level = resolution_to_healpix_level(resolution)
    print(f"Selected max HEALPix level: {max_level} (NSIDE={2**max_level})")

    # Step 3: Create pyramid
    return create_healpix_pyramid(ds, max_level, min_level)
