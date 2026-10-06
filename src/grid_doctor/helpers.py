"""High-level helpers for HEALPix pyramids.

The functions in this module cover three tasks:

- estimating source-grid resolution,
- building a HEALPix pyramid with
  [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix],
- and writing the resulting pyramid to Zarr stores.

Remapping itself lives in [`grid_doctor.remap`][grid_doctor.remap].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import s3fs
import xarray as xr

from .io import DeferredWrite, deferred_writes, store_all
from .pyramid import (
    coarse_levels,
    coarsen_dataset,
    coarsen_mean,
    resolve_coarsen_mode,
    resolve_valid_fraction,
    with_finest_fractions,
)
from .remap import (
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
)
from .remap_backend import (
    _get_latlon_arrays,
    _get_unstructured_dim,
    _is_unstructured,
)
from .types import CoarsenMode, FloatArray, ValidFraction, ZarrOptions

logger = logging.getLogger(__name__)

WRITE_COORDS_MAX_LEVEL = 10
"""Highest level for which ``save_pyramid(write_coords="auto")``
materialises coordinate arrays.  At level 10 the two float64 coordinate
arrays cost ~200 MB per store; one level up they double, and by level 16
they would reach hundreds of GB while carrying no information that is not
already implied by the cell index."""

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


# ===================================================================
# Coarsening
# ===================================================================


def _coarsen_array(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Coarsen a HEALPix array by grouping contiguous nested cells.

    The last dimension is treated as the cell dimension.  All leading
    dimensions are batch dimensions that are preserved.

    Thin wrapper around
    [`coarsen_mean`][grid_doctor.pyramid.coarsen_mean], which
    is exact only when ``values`` is the finest level.

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of child cells per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid (non-NaN)
            children required.  Parent cells with fewer valid
            children are set to NaN.  Default ``0.5`` (at least
            half of the children must be valid).

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    return coarsen_mean(values, factor=factor, min_valid_fraction=min_valid_fraction)


def _coarsen_array_mode(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Coarsen a HEALPix array by taking the mode of grouped children.

    Intended for categorical / discrete data (land cover, soil type,
    …) where averaging class labels would be meaningless.

    For each parent cell the most frequent value among its children is
    selected.  Ties are broken by choosing the value that appears
    first (lowest child index).  When fewer than
    ``min_valid_fraction * factor`` children are valid, the parent is
    set to NaN.

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of child cells per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid children.

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    batch_shape = arr.shape[:-1]
    n_cells = arr.shape[-1]
    n_target = n_cells // factor
    grouped = arr.reshape(*batch_shape, n_target, factor)

    valid = np.isfinite(grouped)
    valid_count = valid.sum(axis=-1)

    # For each child position, count how many of the other children
    # share the same value.  This vectorises the mode calculation
    # without any Python-level loops over cells.
    counts = np.zeros_like(grouped)
    for i in range(factor):
        for j in range(factor):
            counts[..., i] += (grouped[..., i] == grouped[..., j]) & valid[..., j]
    counts[~valid] = -1  # invalid positions cannot win

    best = np.argmax(counts, axis=-1)
    result = np.take_along_axis(
        grouped,
        best[..., np.newaxis],
        axis=-1,
    ).squeeze(-1)

    min_count = max(1, int(np.ceil(min_valid_fraction * factor)))
    result[valid_count < min_count] = np.nan
    return result


def coarsen_healpix(
    ds: xr.Dataset,
    target_level: int,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
    valid_fraction: ValidFraction = False,
) -> xr.Dataset:
    """Coarsen a HEALPix dataset to a lower-resolution level.

    The coarsening is performed as a single reshape + reduction over
    all batch dimensions simultaneously — no per-slice Python loops.

    Parameters
    ----------
    ds:
        HEALPix dataset containing a ``cell`` dimension and the
        attributes ``healpix_nside`` and ``healpix_order``.
    target_level:
        Target HEALPix level (must be lower than the current level).
    coarsen_mode:
        ``"mean"`` — NaN-aware averaging for continuous fields.
        ``"mode"`` — most-frequent-value for categorical fields.
        ``"auto"`` — infer from ``grid_doctor_method`` in dataset
        attributes: ``"nearest"`` → mode, everything else → mean.
    min_valid_fraction:
        Minimum fraction of valid (non-NaN) children required to
        produce a valid parent cell.  Parents with fewer valid
        children are set to NaN.  Default ``0.5`` (at least half
        of the children must be valid).
    valid_fraction:
        Add ``<name>_valid_fraction`` variables: the fraction of valid
        cells of *ds* under each coarse cell (see
        [`create_healpix_pyramid`][grid_doctor.helpers.create_healpix_pyramid]).
        Fractions are relative to the level of *ds*, so coarsen from the
        finest level to get weights for exact global means.

    Returns
    -------
    xarray.Dataset
        Coarsened dataset.

    Notes
    -----
    Nested HEALPix indices have a direct parent-child relationship:
    pixel *i* at level *L* contains children ``4*i`` to ``4*i+3`` at
    level *L+1*.  Coarsening therefore reduces to grouping contiguous
    blocks of ``4**delta_level`` child cells and averaging (or taking
    the mode for categorical data).

    Ring-ordered datasets do not have contiguous parent-child layout
    and must be remapped directly at each target level.

    With ``coarsen_mode="mean"``, always coarsen from the finest level
    rather than chaining level by level.  A chained mean weights every
    valid parent equally, regardless of how many valid finest-level
    cells it was built from, so data with NaNs (land, sea ice,
    observation gaps) drifts between levels.

    Raises
    ------
    ValueError
        When the ordering is not nested, or *target_level* is not
        lower than the current level.
    """
    current_nside = int(ds.attrs["healpix_nside"])
    target_nside = 2**target_level
    if target_nside >= current_nside:
        raise ValueError("target_level must be lower than the current HEALPix level.")

    is_nested = str(ds.attrs.get("healpix_order", "nested")) in {"nested", "nest"}
    if not is_nested:
        raise ValueError(
            "coarsen_healpix only supports nested HEALPix ordering. "
            "Use create_healpix_pyramid(..., nest=False) to "
            "regenerate ring levels directly."
        )

    current_level = int(ds.attrs.get("healpix_level", int(np.log2(current_nside))))
    delta_level = current_level - target_level
    if delta_level <= 0:
        raise ValueError("target_level must be lower than the current HEALPix level.")

    return coarsen_dataset(
        ds,
        source_level=current_level,
        target_level=target_level,
        coarsen_mode=resolve_coarsen_mode(ds, coarsen_mode),
        min_valid_fraction=min_valid_fraction,
        mode_kernel=_coarsen_array_mode,
        valid_fraction=valid_fraction,
    )


# ===================================================================
# Pyramid construction
# ===================================================================


def create_healpix_pyramid(
    ds: xr.Dataset,
    max_level: int | None = None,
    min_level: int = 0,
    *,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
    valid_fraction: ValidFraction = False,
    **kwargs: Any,
) -> dict[int, xr.Dataset]:
    """Create a multi-resolution HEALPix pyramid.

    The input dataset is first remapped to *max_level* with
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].
    For nested output ordering, lower levels are derived efficiently
    with
    [`coarsen_healpix`][grid_doctor.helpers.coarsen_healpix].
    For ring ordering, each lower level is regenerated directly from
    the source dataset.

    For dask-backed input the result is lazy and chunked along ``cell``
    (``cell_chunks``, forwarded to
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]); coarser
    levels keep that chunk size.  Every level builds on the finest one,
    so write the pyramid with
    [`save_pyramid`][grid_doctor.helpers.save_pyramid], which computes
    all levels in one pass.  Calling ``.compute()`` per level instead
    regrids the source once per level.

    Parameters
    ----------
    ds:
        Source dataset.
    max_level:
        Finest HEALPix level.
    min_level:
        Coarsest HEALPix level to keep.
    coarsen_mode:
        Coarsening strategy for building lower pyramid levels.
        ``"mean"`` for continuous data, ``"mode"`` for categorical,
        ``"auto"`` to infer from the remapping method.
    min_valid_fraction:
        Minimum fraction of valid children for a parent cell to be
        valid.  Default ``0.5``.
    valid_fraction:
        Store ``<name>_valid_fraction`` next to the selected variables on
        every level: the fraction of valid finest-level cells under each
        cell, as float32, linked via the CF ``ancillary_variables``
        attribute.  Cell values are means over their valid area, so
        global or regional means over a coarse level need these weights,
        e.g. ``ds.sst.weighted(ds.sst_valid_fraction.fillna(0)).mean("cell")``.

        ``False`` (default) stores nothing.  ``True`` gives every cell
        variable a fraction of its full shape (correct for masks that
        change over time or height).  ``"static"`` stores one ``cell``-only
        fraction from the first slice along all other dimensions -- small,
        but only correct when the mask never changes (e.g. land/sea).
        A list of names selects variables (full shape); a mapping such as
        ``{"sst": "static", "ice": True}`` sets the shape per variable.
        Requires nested ordering.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    Returns
    -------
    dict[int, xarray.Dataset]
        Pyramid keyed by level.
    """
    if max_level is None:
        max_level = resolution_to_healpix_level(get_latlon_resolution(ds))

    is_nested = bool(kwargs.get("nest", True))
    if not is_nested and valid_fraction is not False:
        raise ValueError("valid_fraction requires nested ordering (nest=True).")

    pyramid: dict[int, xr.Dataset] = {}
    finest = regrid_to_healpix(ds, max_level, **kwargs)
    pyramid[max_level] = finest

    if is_nested:
        fractions = resolve_valid_fraction(valid_fraction, finest)
        pyramid[max_level] = with_finest_fractions(finest, fractions, max_level)
        pyramid.update(
            coarse_levels(
                finest,
                max_level=max_level,
                min_level=min_level,
                coarsen_mode=resolve_coarsen_mode(finest, coarsen_mode),
                min_valid_fraction=min_valid_fraction,
                mode_kernel=_coarsen_array_mode,
                valid_fraction=fractions,
            )
        )
        return pyramid

    for level in range(max_level - 1, min_level - 1, -1):
        pyramid[level] = regrid_to_healpix(ds, level, **kwargs)
    return pyramid


# ===================================================================
# S3 / Zarr output
# ===================================================================


def save_pyramid(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: dict[str, Any] | None = None,
    *,
    mode: Literal["a", "w", "r+"] = "a",
    region: Literal["auto"] | Literal["init"] | dict[str, slice] = "auto",
    zarr_format: Literal[2, 3] = 2,
    encoding: dict[int, dict[str, dict[str, Any]]] | None = None,
    write_coords: bool | Literal["auto"] = "auto",
) -> None:
    """Write a HEALPix pyramid to Zarr stores on S3 or local disk.

    Each level is stored below ``"<path>/level_<level>.zarr"``.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset.
    path:
        Target prefix.  An ``"s3://bucket/pyramid"`` URL writes to S3; any
        other value is treated as a local directory path.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.  Only used for S3
        targets; ignored (and optional) for local paths.
    mode:
        Zarr write mode.
    region:
        Region writes for partial updates; "init" for initialising the store.
    zarr_format:
        Zarr format version.
    encoding:
        Per-level encoding dictionaries.
    write_coords:
        Whether to materialise the ``cell``/``latitude``/``longitude``
        coordinate arrays in the store.  HEALPix coordinates are a pure
        function of the cell index, and above roughly level 10 the
        arrays dwarf regional payloads (hundreds of GB at level 16 —
        they are never fill values, so chunk elision cannot help).
        ``"auto"`` (default) writes coordinates for levels up to
        :data:`WRITE_COORDS_MAX_LEVEL` and omits them above; ``True`` /
        ``False`` force either behaviour for all levels.  Coordinate-
        less stores keep the ``crs`` variable and all ``healpix_*``
        attributes, carry ``grid_doctor_implicit_coords = 1``, and are
        meant to be accessed through the region selectors
        ([`select_bbox`][grid_doctor.select_bbox],
        [`select_cells`][grid_doctor.select_cells]), which reconstruct
        coordinates for exactly the cells they return.
    """
    is_s3 = path.startswith("s3://")
    fs = s3fs.S3FileSystem(**(s3_options or {})) if is_s3 else None
    # Stores are initialised per level; the data of all levels is written
    # in one pass (see grid_doctor.io._zarr for why).
    deferred: list[DeferredWrite] = []
    for level, dataset in pyramid.items():
        include_coords = (
            write_coords
            if isinstance(write_coords, bool)
            else level <= WRITE_COORDS_MAX_LEVEL
        )
        if not include_coords:
            dataset = dataset.drop_vars(
                ["latitude", "longitude", "cell"], errors="ignore"
            )
            dataset.attrs["grid_doctor_implicit_coords"] = 1
        level_path = f"{path}/level_{level}.zarr"
        logger.info("Writing HEALPix level %s to %s", level, level_path)
        store: Any
        if is_s3:
            store = s3fs.S3Map(root=level_path, s3=fs)
        else:
            Path(level_path).parent.mkdir(parents=True, exist_ok=True)
            store = level_path
        zarr_options = ZarrOptions(compute=False, mode=mode, zarr_format=zarr_format)
        if zarr_format == 2:
            zarr_options["consolidated"] = True
        if encoding is not None:
            zarr_options["encoding"] = encoding[level]

        if region == "init":
            zarr_options["mode"] = "w"
            # Initialise the store with template.
            template = dataset.chunk().pipe(xr.zeros_like)
            template.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
            continue
        if region == "auto":
            dataset.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
            written, write_region = dataset, None
        else:
            # Fill in the region of data.
            region_keys = set(region)
            to_drop = (
                {
                    name
                    for name, var in dataset.data_vars.items()
                    if region_keys.isdisjoint(set(var.dims))
                }
                | {str(dim) for dim in dataset.dims if str(dim) not in region_keys}
                | {str(coord) for coord in dataset.coords if str(coord) not in region_keys}
            )
            written = dataset.drop_vars(to_drop, errors="ignore").isel(region)
            written.to_zarr(  # type: ignore[call-overload]
                store,
                region=region,
                **zarr_options,
            )
            write_region = region
        deferred.extend(
            deferred_writes(
                written,
                store,
                encoding=zarr_options.get("encoding"),
                region=write_region,
                zarr_format=zarr_format,
            )
        )
    store_all(deferred)

# ===================================================================
# Convenience aliases
# ===================================================================


def latlon_to_healpix_pyramid(
    ds: xr.Dataset,
    *,
    min_level: int = 0,
    max_level: int | None = None,
    **kwargs: Any,
) -> dict[int, xr.Dataset]:
    """Convert a source dataset into a HEALPix pyramid.

    Parameters
    ----------
    ds:
        Source dataset.
    min_level:
        Coarsest level to keep.
    max_level:
        Finest level to generate.  When omitted, the level is
        estimated from the source-grid resolution.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    Returns
    -------
    dict[int, xarray.Dataset]
        HEALPix pyramid keyed by level.

    Examples
    --------
    ```python
    pyramid = latlon_to_healpix_pyramid(ds, method="nearest")
    ```
    """
    return create_healpix_pyramid(
        ds,
        max_level=max_level,
        min_level=min_level,
        **kwargs,
    )


__all__ = [
    "coarsen_healpix",
    "create_healpix_pyramid",
    "get_latlon_resolution",
    "latlon_to_healpix_pyramid",
    "regrid_to_healpix",
    "regrid_unstructured_to_healpix",
    "resolution_to_healpix_level",
    "save_pyramid",
]
