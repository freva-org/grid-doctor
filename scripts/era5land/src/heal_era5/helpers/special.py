"""Special-case helpers for ERA5/ERA5-Land publication."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grid_doctor as gd
import numpy as np
import xarray as xr
from grid_doctor.remap import _healpix_centres, _make_crs_variable

from .file_fetcher import (
    extract_output_attrs,
    find_variable_entry,
    load_cmor_variable_entries,
    load_json,
)
from .formatter import destination_for_level
from .metadata import clean_output_attrs, global_attrs_for_dataset_frequency
from .zarr_publisher import update_zarr_store

EARTH_RADIUS_M = 6_371_007.181
AREACELLA = "areacella"
AREACELLA_METADATA_FREQUENCY = "fx"
DEFAULT_SPECIAL_MAX_LEVEL = 9


def split_special_variables(
    variable_names: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split requested variables into source-backed and special-case groups."""

    source_backed: list[str] = []
    special: list[str] = []
    for name in variable_names:
        if name == AREACELLA:
            special.append(name)
        else:
            source_backed.append(name)
    return tuple(source_backed), tuple(special)


def _areacella_data_array(
    *,
    zoom_number: int,
    attrs: dict[str, Any],
    radius_m: float = EARTH_RADIUS_M,
) -> xr.DataArray:
    """Build the analytical HEALPix `areacella` field for one zoom level."""

    nside = 2 ** int(zoom_number)
    ncell = 12 * nside**2
    cell_area_m2 = (4.0 * np.pi * float(radius_m) ** 2) / float(ncell)
    values = np.full(ncell, cell_area_m2, dtype=np.float64)
    return xr.DataArray(
        values,
        dims=("cell",),
        coords={"cell": np.arange(ncell, dtype=np.int64)},
        attrs=attrs,
        name="areacella",
    )


def _attach_healpix_metadata(ds: xr.Dataset, *, zoom_number: int) -> xr.Dataset:
    """Attach the HEALPix coordinates and attrs expected by downstream code."""

    nside = 2 ** int(zoom_number)
    order = "nested"
    lat_deg, lon_deg = _healpix_centres(int(zoom_number), nest=True)
    ds = ds.assign_coords(
        cell=np.arange(lat_deg.size, dtype=np.int64),
        latitude=("cell", lat_deg),
        longitude=("cell", lon_deg),
        crs=_make_crs_variable(
            level=int(zoom_number),
            nside=nside,
            order=order,
        ),
    )

    for name in ds.data_vars:
        if "cell" in ds[name].dims:
            ds[name].attrs["grid_mapping"] = "crs"

    ds.attrs["healpix_level"] = int(zoom_number)
    ds.attrs["healpix_nside"] = nside
    ds.attrs["healpix_order"] = order
    ds.attrs["grid_doctor_version"] = gd.__version__
    return ds


def _special_variable_attrs(
    *,
    dataset: str,
    variable_name: str,
    target_frequency: str,
    cmor_tables_dir: str | Path,
    mapper_path: str | Path,
) -> dict[str, Any]:
    """Load published attrs for one special-case variable from the CMOR tables."""

    mapper = load_json(mapper_path)
    dataset_cfg = mapper["datasets"][dataset]
    table_prefix = str(dataset_cfg["table_prefix"])
    entries = load_cmor_variable_entries(
        cmor_tables_dir,
        table_prefix=table_prefix,
        frequency=AREACELLA_METADATA_FREQUENCY,
    )
    match = find_variable_entry(entries, variable_name)
    if match is None:
        raise KeyError(
            f"Could not find CMOR metadata for special variable {variable_name!r} "
            f"in {table_prefix}_{AREACELLA_METADATA_FREQUENCY}.json"
        )

    _, entry = match
    attrs = extract_output_attrs(entry)
    attrs["frequency"] = target_frequency
    return clean_output_attrs(attrs)


def build_special_variable_dataset(
    *,
    dataset: str,
    variable_name: str,
    target_frequency: str,
    zoom_number: int,
    cmor_tables_dir: str | Path,
    mapper_path: str | Path,
) -> xr.Dataset:
    """Build one special-case dataset for a target frequency and zoom level."""

    attrs = _special_variable_attrs(
        dataset=dataset,
        variable_name=variable_name,
        target_frequency=target_frequency,
        cmor_tables_dir=cmor_tables_dir,
        mapper_path=mapper_path,
    )

    if variable_name == "areacella":
        array = _areacella_data_array(
            zoom_number=zoom_number,
            attrs=attrs,
        )
    else:
        raise KeyError(f"Unsupported special variable {variable_name!r}")

    ds = xr.Dataset({variable_name: array})
    ds = _attach_healpix_metadata(ds, zoom_number=zoom_number)
    ds.attrs.update(global_attrs_for_dataset_frequency(dataset, target_frequency))
    return ds


def special_variable_attrs_by_name(
    *,
    dataset: str,
    frequency: str,
    variable_names: Iterable[str],
    cmor_tables_dir: str | Path,
    mapper_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Return published attrs for special-case variables at one frequency."""

    return {
        variable_name: build_special_variable_dataset(
            dataset=dataset,
            variable_name=variable_name,
            target_frequency=frequency,
            zoom_number=0,
            cmor_tables_dir=cmor_tables_dir,
            mapper_path=mapper_path,
        )[variable_name].attrs
        for variable_name in variable_names
    }


def special_zoom_numbers(
    *,
    max_level: int = DEFAULT_SPECIAL_MAX_LEVEL,
    highest_level_only: bool,
    coarsen_only: bool,
) -> tuple[int, ...]:
    """Return the zoom levels to publish for special-case variables."""

    if coarsen_only:
        return tuple(range(int(max_level) - 1, -1, -1))
    if highest_level_only:
        return (int(max_level),)
    return tuple(range(int(max_level), -1, -1))


def write_special_variables(
    *,
    dataset: str,
    frequency: str,
    variable_names: Iterable[str],
    zoom_numbers: Iterable[int],
    zarr_format: int,
    clean: bool,
    target_chunk_mb: int,
    cmor_tables_dir: str | Path,
    mapper_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Publish special-case variables into the target HEALPix stores."""

    unique_zoom_numbers = tuple(dict.fromkeys(int(zoom) for zoom in zoom_numbers))
    for zoom_number in unique_zoom_numbers:
        destination = destination_for_level(
            dataset,
            frequency,
            zoom_number,
            output_path=output_path,
        )
        clean_store = bool(clean)
        for variable_name in variable_names:
            special_ds = build_special_variable_dataset(
                dataset=dataset,
                variable_name=variable_name,
                target_frequency=frequency,
                zoom_number=zoom_number,
                cmor_tables_dir=cmor_tables_dir,
                mapper_path=mapper_path,
            )
            update_zarr_store(
                special_ds,
                destination,
                clean=clean_store,
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            )
            clean_store = False
