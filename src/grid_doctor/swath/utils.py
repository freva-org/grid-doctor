"""Helpers for swath and other point-sampled data."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy_groupies as npg
import xarray as xr

from ..remap_backend import (
    _LAT_NAMES,
    _LON_NAMES,
    _canonical_lon,
    _normalise_angle_units,
)
from ..types import BinAgg, FloatArray, Int64Array, SourceUnits

logger = logging.getLogger(__name__)

MAX_NESTED_LEVEL = 29
"""Maximum depth supported by 64-bit nested HEALPix indices."""

AGG_TO_METHOD: dict[str, str] = {
    "mean": "binned-mean",
    "mode": "binned-mode",
    "min": "binned-min",
    "max": "binned-max",
    "count": "binned-count",
}
"""Mapping from aggregation name to the ``grid_doctor_method`` attribute."""

NPG_FUNC: dict[str, str] = {
    "mean": "nanmean",
    "min": "nanmin",
    "max": "nanmax",
}
"""NaN-aware numpy-groupies reductions for the simple aggregations."""

FILL_ATTR_NAMES: tuple[str, ...] = ("_FillValue", "missing_value")
"""Variable attributes inspected for fill values, in priority order."""


# ===================================================================
# Helpers
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
        for attr in FILL_ATTR_NAMES:
            value = source.get(attr)
            if value is not None:
                return float(np.asarray(value).ravel()[0])
    return None


# ===================================================================
# Public utils
# ===================================================================


def bin_simple(
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


def bin_mode(
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


def bin_count(
    group_idx: Int64Array,
    values: FloatArray,
    *,
    n_cells: int,
) -> FloatArray:
    """Count the number of valid samples per cell."""
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


def masked_float64(
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


def resolve_methods(
    ds: xr.Dataset,
    sample_dims: tuple[str, ...],
    agg: "BinAgg | Mapping[str, BinAgg]",
) -> dict[str, str]:
    """Validate and resolve the aggregation method per binnable variable.

    Runs *before* any data is loaded or transformed, so invalid input
    fails in microseconds instead of after gigabytes of masking and
    reshaping (fail-fast).  Catches three error classes the previous
    in-loop lookup surfaced late or not at all:

    - an unknown aggregation method,
    - a mapping that omits a variable that will be binned
      (previously a bare ``KeyError`` mid-loop), and
    - a mapping key that matches no binnable variable (a typo like
      ``"radaince"`` was previously ignored silently).
    """
    binnable = [
        str(name)
        for name, da in ds.data_vars.items()
        if set(sample_dims) <= set(map(str, da.dims))
    ]
    if isinstance(agg, Mapping):
        unknown_keys = sorted(set(map(str, agg)) - set(binnable))
        if unknown_keys:
            raise ValueError(
                f"agg contains entries for {unknown_keys}, which match no "
                f"binnable variable. Binnable variables: {sorted(binnable)}."
            )
        missing = sorted(set(binnable) - set(map(str, agg)))
        if missing:
            raise ValueError(
                f"agg is missing entries for the binnable variables "
                f"{missing}. Provide a method for every variable or pass "
                f"a single method for all."
            )
        resolved = {name: str(agg[name]) for name in binnable}
    else:
        resolved = {name: str(agg) for name in binnable}

    bad = {n: m for n, m in resolved.items() if m not in AGG_TO_METHOD}
    if bad:
        raise ValueError(
            f"Unknown aggregation(s) {bad}. Supported: {sorted(AGG_TO_METHOD)}."
        )
    return resolved


def resolve_point_coords(
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
