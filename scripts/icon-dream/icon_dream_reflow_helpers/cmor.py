#!/usr/bin/env python
"""CMOR / CF standardisation for ICON-DREAM variables.

Each ICON-DREAM source file contains exactly one data variable (the files
live in per-variable directories on the DWD open data server).  The name
cfgrib assigns to that variable is unreliable (``t2m``, ``tp``, ``unknown``,
...), so the mapping below is keyed on the *directory* variable name and the
single data variable of a file is renamed unconditionally.

The table is intentionally declarative: adding or correcting a mapping means
editing one dictionary entry, nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

LOGGER = logging.getLogger(__name__)

#: Nominal output interval per ICON-DREAM frequency in seconds.  Used as a
#: fallback for rate conversions when the actual interval cannot be derived
#: from the time axis (e.g. single-step files).
FREQ_SECONDS: dict[str, float] = {
    "hourly": 3600.0,
    "daily": 86400.0,
    "monthly": 30.4375 * 86400.0,
}

Conversion = Literal["none", "interval_rate", "deaccumulate_rate"]


@dataclass(frozen=True)
class CmorEntry:
    """One source-variable -> CMOR mapping.

    Parameters
    ----------
    cmor_name:
        Target (CMOR/CF) variable name.
    units:
        Target units string written to the variable attributes.
    standard_name:
        CF standard name, if one exists.
    long_name:
        Human readable description.
    expected_units:
        Source units for which ``scale``/``offset``/``conversion`` are
        valid.  When the source units attribute is present but matches
        none of these, the numeric conversion is skipped (attributes are
        still updated) and a warning is logged.  An empty tuple disables
        the check.
    scale, offset:
        Linear transform ``new = old * scale + offset``.
    conversion:
        ``"interval_rate"`` divides by the per-step interval in seconds
        (amount per interval -> flux), ``"deaccumulate_rate"``
        additionally differences along time first (accumulated since
        model start -> flux).  ``"none"`` applies only ``scale``/``offset``.
    """

    cmor_name: str
    units: str
    standard_name: str | None = None
    long_name: str | None = None
    expected_units: tuple[str, ...] = ()
    scale: float = 1.0
    offset: float = 0.0
    conversion: Conversion = "none"
    extra_attrs: dict[str, str] = field(default_factory=dict)


def _u(*variants: str) -> tuple[str, ...]:
    """Expand common spelling variants of a units string."""
    out: set[str] = set()
    for value in variants:
        out.add(value)
        out.add(value.replace("**", ""))
        out.add(value.replace("**", "^"))
        out.add(value.replace(" ", ""))
    return tuple(sorted(out))


#: Source (directory) variable name -> CMOR mapping.
#:
#: 2D near-surface and surface fields first, then 3D model-level fields,
#: then invariant fields found in the ``fx`` constant-fields file.
CMOR_TABLE: dict[str, CmorEntry] = {
    # --- 2D (near-)surface fields -------------------------------------
    "t_2m": CmorEntry(
        "tas", "K", "air_temperature", "Near-Surface Air Temperature",
        expected_units=_u("K"),
    ),
    "td_2m": CmorEntry(
        "tdps", "K", "dew_point_temperature",
        "2m Dew Point Temperature", expected_units=_u("K"),
    ),
    "tmax_2m": CmorEntry(
        "tasmax", "K", "air_temperature",
        "Maximum Near-Surface Air Temperature", expected_units=_u("K"),
    ),
    "tmin_2m": CmorEntry(
        "tasmin", "K", "air_temperature",
        "Minimum Near-Surface Air Temperature", expected_units=_u("K"),
    ),
    # NOTE: verify whether ICON-DREAM tot_prec is the amount accumulated
    # over each output interval (-> "interval_rate", current setting) or
    # accumulated since model start (-> switch to "deaccumulate_rate").
    "tot_prec": CmorEntry(
        "pr", "kg m-2 s-1", "precipitation_flux", "Precipitation",
        expected_units=_u("kg m**-2", "mm"),
        conversion="interval_rate",
    ),
    "pmsl": CmorEntry(
        "psl", "Pa", "air_pressure_at_mean_sea_level",
        "Sea Level Pressure", expected_units=_u("Pa"),
    ),
    "ps": CmorEntry(
        "ps", "Pa", "surface_air_pressure", "Surface Air Pressure",
        expected_units=_u("Pa"),
    ),
    "clct": CmorEntry(
        "clt", "%", "cloud_area_fraction", "Total Cloud Cover Percentage",
        expected_units=_u("%"),
    ),
    "aswdir_s": CmorEntry(
        "rsdsdir", "W m-2",
        "surface_direct_downwelling_shortwave_flux_in_air",
        "Surface Direct Downwelling Shortwave Radiation",
        expected_units=_u("W m**-2"),
    ),
    "aswdifd_s": CmorEntry(
        "rsdsdiff", "W m-2",
        "surface_diffuse_downwelling_shortwave_flux_in_air",
        "Surface Diffuse Downwelling Shortwave Radiation",
        expected_units=_u("W m**-2"),
    ),
    "u_10m": CmorEntry(
        "uas", "m s-1", "eastward_wind", "Eastward Near-Surface Wind",
        expected_units=_u("m s**-1"),
    ),
    "v_10m": CmorEntry(
        "vas", "m s-1", "northward_wind", "Northward Near-Surface Wind",
        expected_units=_u("m s**-1"),
    ),
    "ws_10m": CmorEntry(
        "sfcWind", "m s-1", "wind_speed", "Near-Surface Wind Speed",
        expected_units=_u("m s**-1"),
    ),
    "vmax_10m": CmorEntry(
        "wsgsmax", "m s-1", "wind_speed_of_gust",
        "Maximum Near-Surface Wind Speed of Gust",
        expected_units=_u("m s**-1"),
    ),
    # ICON qv_s is the specific humidity at the *surface* (skin), not at
    # 2 m; huss is used as the closest CMOR analogue -- see the long_name.
    "qv_s": CmorEntry(
        "huss", "1", "specific_humidity",
        "Surface (Skin) Specific Humidity",
        expected_units=_u("kg kg**-1", "1"),
    ),
    "z0": CmorEntry(
        "z0", "m", "surface_roughness_length", "Surface Roughness Length",
        expected_units=_u("m"),
    ),
    # --- 3D model-level fields ----------------------------------------
    "t": CmorEntry(
        "ta", "K", "air_temperature", "Air Temperature",
        expected_units=_u("K"),
    ),
    "u": CmorEntry(
        "ua", "m s-1", "eastward_wind", "Eastward Wind",
        expected_units=_u("m s**-1"),
    ),
    "v": CmorEntry(
        "va", "m s-1", "northward_wind", "Northward Wind",
        expected_units=_u("m s**-1"),
    ),
    "qv": CmorEntry(
        "hus", "1", "specific_humidity", "Specific Humidity",
        expected_units=_u("kg kg**-1", "1"),
    ),
    "p": CmorEntry(
        "pfull", "Pa", "air_pressure",
        "Pressure at Model Full-Levels", expected_units=_u("Pa"),
    ),
    "den": CmorEntry(
        "den", "kg m-3", "air_density", "Air Density",
        expected_units=_u("kg m**-3"),
    ),
    "tke": CmorEntry(
        "tke", "m2 s-2",
        None, "Turbulent Kinetic Energy",
        expected_units=_u("m**2 s**-2", "J kg**-1"),
    ),
    "ws": CmorEntry(
        "ws", "m s-1", "wind_speed", "Wind Speed",
        expected_units=_u("m s**-1"),
    ),
    # --- invariant (fx) fields ----------------------------------------
    "hsurf": CmorEntry(
        "orog", "m", "surface_altitude", "Surface Altitude",
        expected_units=_u("m"),
    ),
    "fr_land": CmorEntry(
        "sftlf", "%", "land_area_fraction", "Percentage of the Grid Cell "
        "Occupied by Land", expected_units=_u("1", "%"), scale=100.0,
    ),
}


def cmor_name(source_variable: str) -> str:
    """Return the CMOR name for a source variable (identity if unmapped)."""
    entry = CMOR_TABLE.get(source_variable.lower())
    return source_variable if entry is None else entry.cmor_name


def target_variable_name(source_variable: str, cmor: bool) -> str:
    """Return the name a source variable has in the target store."""
    return cmor_name(source_variable) if cmor else source_variable


def _interval_seconds(ds: "xr.Dataset", frequency: str) -> "xr.DataArray | float":
    """Per-step interval length in seconds derived from the time axis.

    The interval of the first step is copied from the second; single-step
    datasets fall back to the nominal frequency interval.
    """
    if "time" not in ds.dims or ds.sizes["time"] < 2:
        return FREQ_SECONDS.get(frequency, 3600.0)
    import xarray as xr

    deltas = ds["time"].diff("time") / np.timedelta64(1, "s")
    first = deltas.isel(time=0)
    padded = xr.concat([first, deltas], dim="time")
    padded = padded.assign_coords(time=ds["time"])
    return padded.astype("float64")


def _units_match(entry: CmorEntry, current: str | None) -> bool:
    """Return True when the numeric conversion may be applied."""
    if not entry.expected_units:
        return True
    if current is None:
        # Missing units attribute: trust the table (GRIB decoding
        # frequently loses attrs for "unknown" parameters).
        return True
    return current.strip() in entry.expected_units


def _convert_values(
    da: "xr.DataArray",
    entry: CmorEntry,
    *,
    frequency: str,
) -> "xr.DataArray":
    """Apply the numeric part of a CMOR conversion to one array."""
    out = da
    if entry.conversion == "deaccumulate_rate":
        if "time" in out.dims and out.sizes["time"] > 1:
            first = out.isel(time=slice(0, 1))
            out = out.diff("time", label="upper")
            import xarray as xr

            out = xr.concat([first, out], dim="time")
        out = out / _interval_seconds(out.to_dataset(name="_x"), frequency)
    elif entry.conversion == "interval_rate":
        out = out / _interval_seconds(out.to_dataset(name="_x"), frequency)
    if entry.scale != 1.0:
        out = out * entry.scale
    if entry.offset != 0.0:
        out = out + entry.offset
    return out


def cmorize_dataset(
    ds: "xr.Dataset",
    source_variable: str,
    *,
    frequency: str,
) -> "xr.Dataset":
    """Rename and unit-convert the variables of one source dataset.

    For the regular per-variable files the single data variable is
    renamed to the CMOR name of *source_variable*.  Multi-variable
    datasets (the ``fx`` constant-fields file) are mapped variable by
    variable; unmapped names pass through unchanged.
    """
    data_vars = [str(name) for name in ds.data_vars]
    if len(data_vars) == 1 and source_variable.lower() in CMOR_TABLE:
        mapping = {data_vars[0]: source_variable.lower()}
    else:
        mapping = {
            name: name.lower()
            for name in data_vars
            if name.lower() in CMOR_TABLE
        }
        skipped = sorted(set(data_vars) - set(mapping))
        if skipped:
            LOGGER.info("No CMOR mapping for %s; keeping as-is.", skipped)

    for current_name, table_key in mapping.items():
        entry = CMOR_TABLE[table_key]
        da = ds[current_name]
        current_units = da.attrs.get("units")
        if _units_match(entry, current_units):
            da = _convert_values(da, entry, frequency=frequency)
        else:
            LOGGER.warning(
                "Units %r of %s do not match expected %s; "
                "renaming without numeric conversion.",
                current_units, current_name, entry.expected_units,
            )
        attrs = dict(da.attrs)
        attrs["units"] = entry.units
        if entry.standard_name:
            attrs["standard_name"] = entry.standard_name
        elif "standard_name" in attrs:
            del attrs["standard_name"]
        if entry.long_name:
            attrs["long_name"] = entry.long_name
        attrs["original_name"] = table_key
        attrs.update(entry.extra_attrs)
        da.attrs = attrs
        ds[current_name] = da
        if current_name != entry.cmor_name:
            ds = ds.rename({current_name: entry.cmor_name})
    return ds


__all__ = [
    "CMOR_TABLE",
    "CmorEntry",
    "FREQ_SECONDS",
    "cmor_name",
    "cmorize_dataset",
    "target_variable_name",
]
