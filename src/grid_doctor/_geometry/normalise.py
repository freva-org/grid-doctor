"""Normalise longitudes, latitudes and angle units.

Source coordinates arrive in degrees or radians, with longitudes in
``0..360`` or ``-180..180``.  The functions here bring them to one
canonical form (degrees, longitudes in ``[-180, 180)``, rounded to
micro-degrees) so that geometry can be hashed and meshed reliably.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import numpy as np
import xarray as xr

from ..metadata import LAT_UNITS, LON_UNITS
from ..types import FloatArray, SourceUnits
from .coords import _get_latlon_names, _get_vertex_names, _to_float64

logger = logging.getLogger(__name__)


_MICRO_SCALE = 1_000_000


def _canonical_lon(lon_deg: FloatArray, *, scale: int | None = None) -> FloatArray:
    """Map longitudes into the range ``[-180, 180)``.

    Without *scale* the values are wrapped with float arithmetic. With
    *scale* they are first rounded to multiples of ``1 / scale`` and
    wrapped exactly on integers, so equivalent longitudes (``-10`` and
    ``350``, ``-180`` and ``180``) give bit-identical results. NaN is
    kept in both cases; with *scale* Inf raises.

    Args:
        lon_deg: Longitude values in degrees.
        scale: Rounding resolution, e.g. ``1_000_000`` for micro-degrees.

    Returns:
        Canonicalised longitude array (same shape as input).
    """
    if scale is None:
        return ((lon_deg + 180.0) % 360.0) - 180.0
    q, nan = _to_micro(lon_deg, scale)
    half = 180 * scale
    q = (q + half) % (2 * half) - half
    return np.where(nan, np.nan, q / scale)


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
    [`_looks_like_radians`][grid_doctor._geometry.normalise._looks_like_radians]
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
# Dataset normalisation
# ===================================================================


def _replace_values(ds: xr.Dataset, mapping: Mapping[str, xr.DataArray]) -> xr.Dataset:
    """Return *ds* with each variable in *mapping* replaced, as coordinate or data variable."""
    for name, new in mapping.items():
        ds = ds.assign_coords({name: new}) if name in ds.coords else ds.assign({name: new})
    return ds


def _to_micro(a: FloatArray, scale: int) -> tuple[FloatArray, FloatArray]:
    """Return *a* as integer multiples of ``1 / scale`` plus its NaN mask.

    NaN (e.g. padding vertices) cannot be cast to an integer, so it is
    returned as 0 and must be restored by the caller via the mask.
    """
    a = _to_float64(a)
    if np.isinf(a).any():
        raise ValueError("Cannot scale Inf values.")
    nan = np.isnan(a)
    return np.rint(np.where(nan, 0.0, a) * scale).astype(np.int64), nan


def _check_coordinate(da: xr.DataArray, *, allow_nan: bool) -> None:
    """Raise when coordinate *da* is empty or holds non-finite values.

    Inf is never valid. NaN is accepted only when *allow_nan* is set,
    e.g. for padding vertices of cells with fewer corners.
    """
    if da.size == 0:
        raise ValueError(f"Located coordinate `{da.name}` is empty.")
    values = _to_float64(da.values)
    invalid = np.isinf(values) if allow_nan else ~np.isfinite(values)
    if invalid.any():
        what = "Inf" if allow_nan else "Inf or NaN"
        raise ValueError(f"Located coordinate `{da.name}` contains {what} values.")


def normalise_lon(lon: xr.DataArray, *, allow_nan: bool = False) -> xr.DataArray:
    """Convert longitudes to the expected [-180, 180) range.

    Values are expected in degrees (see ``normalise_degrees``).

    Args:
        lon: xr.DataArray - longitude coordinate in degree
        allow_nan: Accept NaN values (e.g. padding vertices). Inf is always
            rejected.
    Raises:
        ValueError: When *lon* is empty or holds invalid values.
    Return:
        xr.DataArray with values and `unit` attribute adjusted (`degrees_east`)
    """
    _check_coordinate(lon, allow_nan=allow_nan)

    return xr.DataArray(
        _canonical_lon(lon.values, scale=_MICRO_SCALE), dims=lon.dims, attrs=lon.attrs | {"units": LON_UNITS}
    )


def _norm_lat(lat: FloatArray, *, scale: int = _MICRO_SCALE) -> FloatArray:
    """Convention lat: [-90, 90]°."""
    q, nan = _to_micro(lat, scale)
    if np.any(np.abs(q) > 90 * scale):
        raise ValueError("outside [-90, 90]")
    return np.where(nan, np.nan, q / scale)


def normalise_lat(lat: xr.DataArray, *, allow_nan: bool = False) -> xr.DataArray:
    """Convert latitudes to expected [-90, 90] range.

    Values are expected in degrees (see ``normalise_degrees``).

    Args:
        lat: xr.DataArray - latitude coordinate in degree
        allow_nan: Accept NaN values (e.g. padding vertices). Inf is always
            rejected.
    Raises:
        ValueError: When *lat* is empty, holds invalid values or lies
            outside ``[-90, 90]``.
    Return:
        xr.DataArray with values and `unit` attribute adjusted (`degrees_north`)
    """
    _check_coordinate(lat, allow_nan=allow_nan)

    try:
        values = _norm_lat(lat.values)
    except ValueError as exc:
        raise ValueError(f"Located coordinate `{lat.name}` has values {exc}.") from exc
    return xr.DataArray(values, dims=lat.dims, attrs=lat.attrs | {"units": LAT_UNITS})


def normalise_degrees(array: xr.DataArray, source_units: SourceUnits = "auto") -> xr.DataArray:
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


def _attr_units(da: xr.DataArray) -> Literal["deg", "rad"] | None:
    """Angular unit stated by the ``units`` attribute of *da*, if any."""
    units = str(da.attrs.get("units", "")).lower()
    if units.startswith("rad"):
        return "rad"
    if units.startswith("deg"):
        return "deg"
    return None


def _resolve_pair_units(
    lat: xr.DataArray,
    lon: xr.DataArray,
    source_units: SourceUnits,
    default: Literal["deg", "rad"] | None = None,
) -> Literal["deg", "rad"]:
    """Decide the angular unit shared by a latitude/longitude pair.

    An explicit *source_units* wins. Otherwise the ``units`` attributes
    decide; they must not contradict each other. Without attributes the
    pair falls back to *default* (e.g. the unit of the cell centres for
    their vertices) and, failing that, to a value check: the pair is taken
    as radians only when both lie within ``[-2*pi, 2*pi]``.

    Raises:
        ValueError: When the ``units`` attributes of *lat* and *lon* disagree.
    """
    if source_units != "auto":
        return source_units
    stated = {u for u in (_attr_units(lat), _attr_units(lon)) if u is not None}
    if len(stated) > 1:
        raise ValueError(
            f"`{lat.name}` and `{lon.name}` state different angular units "
            f"({lat.attrs.get('units')!r} vs {lon.attrs.get('units')!r}); pass `source_units`."
        )
    if stated:
        return stated.pop()
    if default is not None:
        return default
    guess: Literal["deg", "rad"] = (
        "rad"
        if _looks_like_radians(_to_float64(lat.values)) and _looks_like_radians(_to_float64(lon.values))
        else "deg"
    )
    logger.warning(
        "No angular units found for `%s`/`%s`; values suggest %s.",
        lat.name,
        lon.name,
        "radians" if guess == "rad" else "degrees",
    )
    return guess


def normalise_dataset(ds: xr.Dataset, source_units: SourceUnits = "auto") -> xr.Dataset:
    """Return a normalised copy of the input dataset.

    Latitude/longitude coordinates and their units are inferred and
    normalised. On unstructured grids, the cell vertices are normalised
    as well when present.

    Args:
        ds: Source geometry dataset.
        source_units: Angular unit convention of the coordinates. With
            ``"auto"`` the ``units`` attributes decide per latitude/longitude
            pair; without them, vertices follow their cell centres and the
            centres are taken as radians only when both lie within
            ``[-2*pi, 2*pi]`` (with a warning).

    Returns:
        Normalised copy of *ds*.

    Raises:
        ValueError: When no latitude/longitude coordinates are present,
            a coordinate is empty, the cell centres contain NaN or Inf
            (masked grids are not supported), the vertices contain Inf,
            a latitude lies outside ``[-90, 90]``, or the ``units``
            attributes of a latitude/longitude pair disagree.
    """
    # Cell centres are required and must be finite; vertices are optional
    # and may be NaN-padded.
    names = [(_get_latlon_names(ds), False)]
    try:
        names.append((_get_vertex_names(ds), True))
    except ValueError:
        pass  # vertices are optional

    mapping: dict[str, xr.DataArray] = {}
    units: Literal["deg", "rad"] | None = None  # vertices follow their centres
    for (lat_name, lon_name), allow_nan in names:
        lat, lon = ds[lat_name], ds[lon_name]
        units = _resolve_pair_units(lat, lon, source_units, default=units)
        mapping[lat_name] = normalise_lat(normalise_degrees(lat, units), allow_nan=allow_nan)
        mapping[lon_name] = normalise_lon(normalise_degrees(lon, units), allow_nan=allow_nan)
    return _replace_values(ds, mapping)
