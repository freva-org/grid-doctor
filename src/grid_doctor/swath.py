"""Point-binning helpers for swath and other point-sampled data.

This module maps *point-sampled* data — satellite Level-2 swaths, station
records, trajectories, or any dataset where every sample carries its own
latitude/longitude — onto the standard grid-doctor HEALPix representation.

Point data cannot go through the ESMF weight path in
[`grid_doctor.remap`][grid_doctor.remap]:

- every granule (orbit frame) has unique geometry, so weight files can
  never be reused, and
- ESMF's nearest source-to-destination method would assign a value to
  *every* global HEALPix cell, smearing a narrow swath over the whole
  sphere.

Instead, [`bin_to_healpix`][grid_doctor.swath.bin_to_healpix] assigns each
sample to the HEALPix cell containing it and reduces all samples per cell.
In the oversampled limit — source pixel spacing much finer than the target
cell spacing — the per-cell sample **mean** converges to the area-weighted
mean and is therefore the binning analogue of conservative remapping,
while the per-cell **mode** is the analogue of nearest-neighbour remapping
for categorical fields.  Choose the target level accordingly: the HEALPix
cell spacing (``58.6° / 2**level``) should be coarser than the sample
spacing, exactly as
[`resolution_to_healpix_level`][grid_doctor.helpers.resolution_to_healpix_level]
would suggest.

All cell geometry uses a perfect sphere, consistent with the rest of
grid-doctor and with the Waterpark technical decisions.  Geodetic (WGS84)
latitudes are deliberately interpreted as spherical: the maximum
discrepancy (~0.19° at 45° latitude) is accepted so that all datasets in a
hub share one indexing geometry and overlay without systematic offsets.

The output of the dense path carries the full grid-doctor metadata
(``crs`` variable, ``healpix_*`` and ``grid_doctor_*`` attributes) so that
[`coarsen_healpix`][grid_doctor.helpers.coarsen_healpix],
[`save_pyramid`][grid_doctor.helpers.save_pyramid], and CF-aware viewers
work unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy_groupies as npg
import xarray as xr

from .remap import _attach_healpix_coords
from .remap_backend import (
    _LAT_NAMES,
    _LON_NAMES,
    _canonical_lon,
    _normalise_angle_units,
    _require_healpix_geo_module,
)
from .types import BinAgg, FloatArray, Int64Array, SourceUnits

logger = logging.getLogger(__name__)

_MAX_NESTED_LEVEL = 29
"""Maximum depth supported by 64-bit nested HEALPix indices."""

_AGG_TO_METHOD: dict[str, str] = {
    "mean": "binned-mean",
    "mode": "binned-mode",
    "min": "binned-min",
    "max": "binned-max",
    "count": "binned-count",
}
"""Mapping from aggregation name to the ``grid_doctor_method`` attribute."""

_NPG_FUNC: dict[str, str] = {
    "mean": "nanmean",
    "min": "nanmin",
    "max": "nanmax",
}
"""NaN-aware numpy-groupies reductions for the simple aggregations."""

_FILL_ATTR_NAMES: tuple[str, ...] = ("_FillValue", "missing_value")
"""Variable attributes inspected for fill values, in priority order."""


# ===================================================================
# Coordinate resolution
# ===================================================================


def _find_coord_name(
    ds: xr.Dataset,
    candidates: tuple[str, ...],
    explicit: str | None,
    kind: str,
) -> str:
    """Return the name of a coordinate variable in *ds*.

    Args:
        ds: Source dataset.
        candidates: Priority-ordered candidate names.
        explicit: Explicit name that overrides auto-detection.
        kind: ``"latitude"`` or ``"longitude"`` (for error messages).

    Returns:
        The resolved variable name.

    Raises:
        KeyError: When an explicit name is not present in *ds*.
        ValueError: When no candidate name is found.
    """
    if explicit is not None:
        if explicit not in ds.coords and explicit not in ds.data_vars:
            raise KeyError(f"{kind} variable {explicit!r} not found in dataset.")
        return explicit
    for name in candidates:
        if name in ds.coords or name in ds.data_vars:
            return name
    available = sorted({*map(str, ds.coords), *map(str, ds.data_vars)})
    raise ValueError(
        f"Could not locate a {kind} coordinate. Pass "
        f"``{kind[:3]}_name=...`` explicitly. Available names: {available}."
    )


def _resolve_point_coords(
    ds: xr.Dataset,
    *,
    lat_name: str | None,
    lon_name: str | None,
    source_units: SourceUnits,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Extract per-sample coordinates and the sample dimensions.

    Args:
        ds: Source dataset.
        lat_name: Explicit latitude variable name or *None*.
        lon_name: Explicit longitude variable name or *None*.
        source_units: Angular unit convention.

    Returns:
        ``(lat, lon, sample_dims)`` where *lat* and *lon* are flattened
        float64 arrays and *sample_dims* are the dimensions of the
        coordinate variables (e.g. ``("along_track", "across_track")``).

    Raises:
        ValueError: When latitude and longitude disagree in dims/shape.
    """
    lat_var = ds[_find_coord_name(ds, _LAT_NAMES, lat_name, "latitude")]
    lon_var = ds[_find_coord_name(ds, _LON_NAMES, lon_name, "longitude")]

    if tuple(lat_var.dims) != tuple(lon_var.dims):
        raise ValueError(
            "Latitude and longitude must share the same dimensions, got "
            f"{tuple(lat_var.dims)} vs {tuple(lon_var.dims)}."
        )
    if lat_var.shape != lon_var.shape:
        raise ValueError(
            "Latitude and longitude must share the same shape, got "
            f"{lat_var.shape} vs {lon_var.shape}."
        )

    lat = _normalise_angle_units(
        np.asarray(lat_var.values, dtype=np.float64).ravel(), source_units
    )
    lon = _canonical_lon(
        _normalise_angle_units(
            np.asarray(lon_var.values, dtype=np.float64).ravel(), source_units
        )
    )
    return lat, lon, tuple(map(str, lat_var.dims))


# ===================================================================
# Fill-value handling
# ===================================================================


def _declared_fill_value(da: xr.DataArray) -> float | None:
    """Return the fill value declared on *da*, if any.

    Both ``attrs`` and ``encoding`` are inspected for ``_FillValue`` and
    ``missing_value``.  No dtype-based guessing is performed: a fill
    value is only honoured when the producer declared one.

    Args:
        da: Data variable.

    Returns:
        The declared fill value, or *None*.
    """
    for source in (da.attrs, da.encoding):
        for attr in _FILL_ATTR_NAMES:
            value = source.get(attr)
            if value is not None:
                return float(np.asarray(value).ravel()[0])
    return None


def _masked_float64(
    da: xr.DataArray,
    *,
    fill_value: float | None,
) -> FloatArray:
    """Return *da* as float64 with invalid samples set to NaN.

    Invalid samples are those equal to the fill value (explicit
    *fill_value* argument, falling back to a declared ``_FillValue`` /
    ``missing_value``) and, for floating-point input, non-finite values.

    Args:
        da: Data variable.
        fill_value: Explicit fill value overriding declared attributes.
            Note that ``0`` is a perfectly valid fill value here — the
            check is ``is None``, never truthiness.

    Returns:
        Float64 array of the same shape with NaN at invalid samples.
    """
    fill = fill_value if fill_value is not None else _declared_fill_value(da)
    raw = np.asarray(da.values)
    values = raw.astype(np.float64)
    invalid = ~np.isfinite(values)
    if fill is not None and np.isfinite(fill):
        # Compare in float64 so integer fills (255, -32768, 0, ...) and
        # large CF float fills (9.9692e36) are both handled without
        # dtype-cast surprises.
        invalid |= values == float(fill)
    values[invalid] = np.nan
    return values


# ===================================================================
# Per-cell reductions
# ===================================================================


def _bin_simple(
    group_idx: Int64Array,
    values: FloatArray,
    *,
    n_cells: int,
    func: str,
) -> FloatArray:
    """Apply a NaN-aware numpy-groupies reduction along the last axis."""
    result = npg.aggregate(
        group_idx,
        values,
        axis=-1,
        func=func,
        size=n_cells,
        fill_value=np.nan,
    )
    return cast(FloatArray, np.asarray(result, dtype=np.float64))


def _bin_mode(
    group_idx: Int64Array,
    values: FloatArray,
    *,
    n_cells: int,
    max_classes: int = 256,
) -> FloatArray:
    """Per-cell mode for categorical data.

    Ties are broken deterministically by the *lowest class value* (the
    class loop runs over sorted unique values and a later class only wins
    with a strictly greater count).

    Args:
        group_idx: Compact cell index per sample (last axis).
        values: Sample values with NaN marking invalid samples.
        n_cells: Number of cells.
        max_classes: Guard against accidentally binning continuous data.

    Returns:
        Mode per cell; NaN where no valid sample exists.

    Raises:
        ValueError: When the number of distinct classes exceeds
            *max_classes*.
    """
    classes = np.unique(values[np.isfinite(values)])
    batch_shape = values.shape[:-1]
    best_value = np.full((*batch_shape, n_cells), np.nan)
    if classes.size == 0:
        return best_value
    if classes.size > max_classes:
        raise ValueError(
            f"Found {classes.size} distinct classes (> {max_classes}). "
            "Mode binning is meant for categorical data; use agg='mean' "
            "for continuous fields."
        )
    best_count = np.zeros((*batch_shape, n_cells), dtype=np.int64)
    for cls in classes:
        count = npg.aggregate(
            group_idx,
            (values == cls).astype(np.int64),
            axis=-1,
            func="sum",
            size=n_cells,
            fill_value=0,
        ).astype(np.int64)
        wins = count > best_count
        best_value[wins] = cls
        best_count[wins] = count[wins]
    return best_value


def _bin_count(
    group_idx: Int64Array,
    values: FloatArray,
    *,
    n_cells: int,
) -> FloatArray:
    """Calculate the number of valid samples per cell."""
    result = npg.aggregate(
        group_idx,
        # numpy-groupies treats 1-D boolean input to ``sum`` as a
        # logical ``any`` -- cast explicitly to count samples.
        np.isfinite(values).astype(np.int64),
        axis=-1,
        func="sum",
        size=n_cells,
        fill_value=0,
    )
    return cast(FloatArray, np.asarray(result, dtype=np.float64))


# ===================================================================
# Public API
# ===================================================================


def bin_to_healpix(
    ds: xr.Dataset,
    level: int,
    *,
    agg: BinAgg | Mapping[str, BinAgg] = "mean",
    nest: bool = True,
    lat_name: str | None = None,
    lon_name: str | None = None,
    source_units: SourceUnits = "auto",
    fill_values: Mapping[str, float] | None = None,
    min_count: int = 1,
    with_counts: bool = False,
    dense: bool = True,
) -> xr.Dataset:
    """Bin point-sampled data into HEALPix cells.

    Every sample is assigned to the HEALPix cell containing its
    coordinates (perfect-sphere geometry, consistent with all other
    grid-doctor output) and all samples per cell are reduced with the
    requested aggregation.

    Parameters
    ----------
    ds:
        Dataset with per-sample latitude/longitude variables (swath
        geolocation, station coordinates, …).  Variables that span the
        sample dimensions are binned; variables that share none of the
        sample dimensions are passed through unchanged; variables that
        overlap them only partially are skipped with a warning.
        Non-sample dimensions (``time``, ``channel``, …) are preserved
        as batch dimensions.
    level:
        Target HEALPix level.  Choose it so the cell spacing
        (``58.6° / 2**level``) is coarser than the sample spacing —
        only then is the per-cell mean a faithful stand-in for
        conservative remapping.
    agg:
        Aggregation per variable: a single method applied to all
        variables or a mapping ``{variable: method}``.  Supported
        methods: ``"mean"`` (continuous fields; binning analogue of
        conservative remapping), ``"mode"`` (categorical fields;
        analogue of nearest-neighbour), ``"min"``, ``"max"``, and
        ``"count"``.  A per-sample *sum* is deliberately not offered:
        it scales with sample density (orbit overlap, across-track
        pixel count) rather than any physical integral.
    nest:
        Use nested HEALPix ordering when *True*.  Nested ordering is
        required for pyramid coarsening.
    lat_name, lon_name:
        Explicit coordinate variable names.  When omitted, the standard
        grid-doctor name lists are searched (``latitude``, ``lat``, …).
    source_units:
        Angular unit convention of the coordinates.
    fill_values:
        Explicit per-variable fill values, e.g. ``{"cloud_type": 255}``.
        When omitted, declared ``_FillValue`` / ``missing_value``
        attributes are honoured; floating-point non-finite values are
        always treated as invalid.  ``0`` is accepted as a fill value.
    min_count:
        Minimum number of valid samples required for a cell to be
        valid.  Cells with fewer samples are set to NaN (does not apply
        to ``"count"``).
    with_counts:
        Add an ``<name>_count`` companion variable per binned variable
        holding the number of valid samples per cell.  Recommended for
        published datasets: it makes coverage auditable and enables
        unbiased merging of overlapping granules downstream.
    dense:
        When *True* (default), return the full ``12 * 4**level`` cell
        array with standard grid-doctor coordinates, CRS variable, and
        attributes — directly consumable by
        [`coarsen_healpix`][grid_doctor.helpers.coarsen_healpix] and
        [`save_pyramid`][grid_doctor.helpers.save_pyramid].  When
        *False*, return a compact dataset containing only the touched
        cells (the ``cell`` coordinate holds the actual HEALPix
        indices) — useful as a per-granule intermediate at high levels;
        convert with
        [`sparse_to_dense`][grid_doctor.swath.sparse_to_dense] before
        coarsening or publishing.

    Returns
    -------
    xarray.Dataset
        Binned dataset on the HEALPix grid.

    Raises
    ------
    ValueError
        On unknown aggregation methods, invalid levels, or when no
        valid sample coordinates exist.

    Examples
    --------
    ```python
    hpx = gd.bin_to_healpix(
        swath,
        level=11,
        agg={"radiance": "mean", "cloud_type": "mode"},
        fill_values={"cloud_type": 255},
        with_counts=True,
    )
    pyramid = {11: hpx}
    for lvl in range(10, -1, -1):
        pyramid[lvl] = gd.coarsen_healpix(pyramid[lvl + 1], lvl)
    ```
    """
    if not 0 <= level <= _MAX_NESTED_LEVEL:
        raise ValueError(f"level must be within [0, {_MAX_NESTED_LEVEL}].")
    if min_count < 1:
        raise ValueError("min_count must be at least 1.")

    lat, lon, sample_dims = _resolve_point_coords(
        ds, lat_name=lat_name, lon_name=lon_name, source_units=source_units
    )

    valid_coords = np.isfinite(lat) & np.isfinite(lon)
    if not valid_coords.any():
        raise ValueError("No samples with valid (finite) coordinates found.")

    module, kwargs = _require_healpix_geo_module(nest)
    cell_ids = np.asarray(
        module.lonlat_to_healpix(
            lon[valid_coords], lat[valid_coords], depth=level, **kwargs
        ),
        dtype=np.int64,
    )

    # Compact indexing: bin into the touched cells only, scatter later.
    unique_cells, group_idx = np.unique(cell_ids, return_inverse=True)
    group_idx = group_idx.astype(np.int64)
    n_cells = int(unique_cells.size)

    binned: dict[str, xr.DataArray] = {}
    counts: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        var_name = str(name)
        overlap = set(sample_dims) & set(map(str, da.dims))
        if not overlap:
            binned[var_name] = da
            continue
        if overlap != set(sample_dims):
            logger.warning(
                "Skipping %r: covers only part of the sample dimensions %s.",
                var_name,
                sample_dims,
            )
            continue

        method = str(agg[var_name] if isinstance(agg, Mapping) else agg)
        if method not in _AGG_TO_METHOD:
            raise ValueError(
                f"Unknown aggregation {method!r} for {var_name!r}. "
                f"Supported: {sorted(_AGG_TO_METHOD)}."
            )

        batch_dims = tuple(d for d in map(str, da.dims) if d not in sample_dims)
        arranged = da.transpose(*batch_dims, *sample_dims)
        fill = (fill_values or {}).get(var_name)
        values = _masked_float64(arranged, fill_value=fill)
        values = values.reshape(*arranged.shape[: len(batch_dims)], -1)
        values = values[..., valid_coords]

        valid_count = _bin_count(group_idx, values, n_cells=n_cells)

        if method == "count":
            result = valid_count
        elif method == "mode":
            result = _bin_mode(group_idx, values, n_cells=n_cells)
            result[valid_count < min_count] = np.nan
        else:
            result = _bin_simple(
                group_idx, values, n_cells=n_cells, func=_NPG_FUNC[method]
            )
            result[valid_count < min_count] = np.nan

        attrs = {
            key: value for key, value in da.attrs.items() if key not in _FILL_ATTR_NAMES
        }
        attrs["grid_doctor_method"] = _AGG_TO_METHOD[method]
        binned[var_name] = xr.DataArray(result, dims=(*batch_dims, "cell"), attrs=attrs)
        if with_counts and method != "count":
            counts[f"{var_name}_count"] = xr.DataArray(
                valid_count.astype(np.int32),
                dims=(*batch_dims, "cell"),
                attrs={"long_name": f"number of valid samples binned into {var_name}"},
            )

    binned.update(counts)
    result_ds = xr.Dataset(binned, attrs=ds.attrs.copy())

    # Preserve batch coordinates (time, channel, ...).
    keep_coords = {
        str(coord): ds.coords[coord]
        for coord in ds.coords
        if not set(sample_dims) & set(map(str, ds.coords[coord].dims))
    }
    result_ds = result_ds.assign_coords(keep_coords)
    result_ds.attrs["grid_doctor_method"] = _dominant_method(result_ds)

    if dense:
        return _scatter_to_dense(result_ds, unique_cells, level=level, nest=nest)
    return _attach_sparse_coords(result_ds, unique_cells, level=level, nest=nest)


def sparse_to_dense(ds: xr.Dataset) -> xr.Dataset:
    """Scatter a compact (sparse) binned dataset onto the full grid.

    Parameters
    ----------
    ds:
        Output of ``bin_to_healpix(..., dense=False)``.

    Returns
    -------
    xarray.Dataset
        Dense dataset with the standard grid-doctor HEALPix coordinates
        and metadata, ready for
        [`coarsen_healpix`][grid_doctor.helpers.coarsen_healpix].

    Raises
    ------
    ValueError
        When *ds* does not look like a sparse binned dataset.
    """
    if int(ds.attrs.get("grid_doctor_sparse", 0)) != 1:
        raise ValueError("Dataset is not a sparse binned dataset.")
    level = int(ds.attrs["healpix_level"])
    nest = str(ds.attrs.get("healpix_order", "nested")) in {"nested", "nest"}
    cell_ids = np.asarray(ds["cell"].values, dtype=np.int64)
    stripped = ds.drop_vars(
        [name for name in ("latitude", "longitude", "crs", "cell") if name in ds]
    )
    return _scatter_to_dense(stripped, cell_ids, level=level, nest=nest)


# ===================================================================
# Output assembly
# ===================================================================


def _dominant_method(ds: xr.Dataset) -> str:
    """Return the dataset-level ``grid_doctor_method`` attribute.

    ``"binned-mode"`` wins when *all* binned variables are categorical,
    so that ``coarsen_healpix(..., coarsen_mode="auto")`` selects mode
    coarsening; otherwise ``"binned-mean"`` is used and categorical
    variables should be coarsened separately or with an explicit
    ``coarsen_mode``.
    """
    methods = {
        str(da.attrs.get("grid_doctor_method"))
        for da in ds.data_vars.values()
        if "grid_doctor_method" in da.attrs
    }
    if methods == {"binned-mode"}:
        return "binned-mode"
    return "binned-mean"


def _scatter_to_dense(
    ds: xr.Dataset,
    cell_ids: Int64Array,
    *,
    level: int,
    nest: bool,
) -> xr.Dataset:
    """Scatter compact per-cell arrays onto the full HEALPix grid."""
    npix = 12 * 4**level
    scattered: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if "cell" not in da.dims:
            scattered[str(name)] = da
            continue
        arranged = da.transpose(..., "cell")
        if np.issubdtype(arranged.dtype, np.integer):
            full = np.zeros((*arranged.shape[:-1], npix), dtype=arranged.dtype)
        else:
            full = np.full((*arranged.shape[:-1], npix), np.nan)
        full[..., cell_ids] = arranged.values
        scattered[str(name)] = xr.DataArray(full, dims=arranged.dims, attrs=da.attrs)
    dense = xr.Dataset(scattered, attrs=ds.attrs.copy())
    dense = dense.assign_coords(
        {
            str(coord): ds.coords[coord]
            for coord in ds.coords
            if "cell" not in ds.coords[coord].dims and str(coord) != "cell"
        }
    )
    dense.attrs.pop("grid_doctor_sparse", None)
    method = str(dense.attrs.pop("grid_doctor_method", "binned-mean"))
    return _attach_healpix_coords(dense, level=level, nest=nest, method=method)


def _attach_sparse_coords(
    ds: xr.Dataset,
    cell_ids: Int64Array,
    *,
    level: int,
    nest: bool,
) -> xr.Dataset:
    """Attach coordinates and metadata to a compact binned dataset.

    The ``cell`` coordinate holds the *actual* HEALPix indices of the
    touched cells (unlike the dense representation, where ``cell`` is a
    positional ``arange``).  The dataset is marked with
    ``grid_doctor_sparse = 1``.
    """
    module, kwargs = _require_healpix_geo_module(nest)
    lon_deg, lat_deg = module.healpix_to_lonlat(cell_ids, level, **kwargs)

    from .remap import _make_crs_variable

    result = ds.assign_coords(
        cell=cell_ids,
        latitude=("cell", np.asarray(lat_deg, dtype=np.float64)),
        longitude=("cell", _canonical_lon(np.asarray(lon_deg, dtype=np.float64))),
        crs=_make_crs_variable(
            level=level, nside=2**level, order="nested" if nest else "ring"
        ),
    )
    for name in result.data_vars:
        if "cell" in result[name].dims:
            result[name].attrs["grid_mapping"] = "crs"

    import grid_doctor as _gd

    result.attrs["healpix_level"] = level
    result.attrs["healpix_nside"] = 2**level
    result.attrs["healpix_order"] = "nested" if nest else "ring"
    result.attrs["grid_doctor_version"] = _gd.__version__
    result.attrs["grid_doctor_sparse"] = 1
    return result


__all__ = [
    "bin_to_healpix",
    "sparse_to_dense",
]
