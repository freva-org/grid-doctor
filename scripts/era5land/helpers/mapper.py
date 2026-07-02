"""Orchestration helpers for ERA5/ERA5-Land HEALPix conversion."""

from datetime import date
import logging
from pathlib import Path
import re
from typing import Iterable, Optional

import grid_doctor as gd
import xarray as xr

from .special import (
    special_zoom_numbers,
    special_variable_attrs_by_name,
    split_special_variables,
    write_special_variables,
)
from .datasets import (
    merge_frequency_dataset,
    normalise_reduced_gaussian_dataset,
)
from .file_fetcher import SourceRecord
from .formatter import (
    destination_for_level,
    existing_destinations_for_frequency,
    group_records_by_frequency,
)
from .logging_utils import log_stage
from .metadata import (
    attrs_for_record,
    global_attrs_for_dataset_frequency,
    global_attrs_for_records,
)
from .zarr_publisher import (
    sync_global_attrs,
    sync_named_variable_attrs,
    update_zarr_store,
)

LOGGER = logging.getLogger(__name__)
LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")
HEALPIX_DIR_MODE = 0o2775
PROTECTED_GRID_ATTRS = {
    "healpix_level",
    "healpix_nside",
    "healpix_order",
    "grid_doctor_version",
    "grid_doctor_method",
    "grid_doctor_coarsened_from_level",
}


def _ensure_output_directory(destination: str) -> None:
    """Create the output directory tree and enforce HEALPix group permissions.

    The directory containing the target Zarr store is created when missing. The
    mode is then normalised to ``2775`` from the ``healpix`` path component
    downward so the setgid bit and group-writable permissions are preserved even
    when the process umask is more restrictive.
    """

    target_dir = Path(destination).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    parts = target_dir.parts
    try:
        healpix_index = parts.index("healpix")
    except ValueError:
        chmod_start = len(parts) - 1
    else:
        chmod_start = healpix_index

    current = Path(parts[0]) if target_dir.is_absolute() else Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current /= part
        if index >= chmod_start and current.is_dir():
            current.chmod(HEALPIX_DIR_MODE)

    if chmod_start == 0 and current.is_dir():
        current.chmod(HEALPIX_DIR_MODE)


def _variable_names(
    records: Iterable[SourceRecord],
    extra: Iterable[str] = (),
) -> str:
    """Return a stable comma-separated list of variable names for logging."""

    names = {record.variable for record in records}
    names.update(str(name) for name in extra)
    return ",".join(sorted(names))


def _write_zoom_level(
    dataset,
    *,
    source_dataset: str,
    frequency: str,
    variables: str,
    zoom_number: int,
    global_attrs: dict[str, object],
    clean: bool,
    zarr_format: int,
) -> None:
    """Write one zoom level with consistent progress logging."""

    merged_attrs = dict(global_attrs)
    for key in PROTECTED_GRID_ATTRS:
        if key in dataset.attrs:
            merged_attrs[key] = dataset.attrs[key]
    dataset.attrs = merged_attrs
    destination = destination_for_level(source_dataset, frequency, zoom_number)
    log_stage(
        LOGGER,
        "zarr_write_start",
        frequency=frequency,
        variables=variables,
        zoom=zoom_number,
        destination=destination,
    )
    _ensure_output_directory(destination)
    update_zarr_store(
        dataset,
        destination,
        clean=clean,
        zarr_format=zarr_format,
    )


def _prepare_dataset_for_coarsen(ds: xr.Dataset) -> xr.Dataset:
    """Drop level-specific HEALPix geometry that must be regenerated after coarsening."""

    return ds.drop_vars(
        [
            "latitude",
            "longitude",
            "lat_vertices",
            "lon_vertices",
        ],
        errors="ignore",
    )


def _existing_level_destinations(
    source_dataset: str,
    frequency: str,
) -> list[tuple[int, str]]:
    """Return existing destinations for one frequency with parsed zoom levels."""

    destinations: list[tuple[int, str]] = []
    for destination in existing_destinations_for_frequency(source_dataset, frequency):
        match = LEVEL_RE.search(destination)
        if match is None:
            continue
        destinations.append((int(match.group("level")), destination))
    return sorted(destinations, reverse=True)


def _coarsen_existing_frequency(
    *,
    source_dataset: str,
    frequency: str,
    variables: str,
    zarr_format: int,
    clean: bool,
) -> None:
    """Build lower zoom levels from the highest existing Zarr store."""

    existing = _existing_level_destinations(source_dataset, frequency)
    if not existing:
        raise ValueError(
            f"No existing HEALPix Zarr stores found for frequency {frequency!r}."
        )

    highest_level, highest_destination = existing[0]
    log_stage(
        LOGGER,
        "coarsen_source_open",
        frequency=frequency,
        variables=variables,
        zoom=highest_level,
        source=highest_destination,
    )
    current = xr.open_zarr(highest_destination, consolidated=(zarr_format == 2)).load()
    global_attrs = dict(current.attrs)
    for zoom_number in range(highest_level - 1, -1, -1):
        current = gd.coarsen_healpix(_prepare_dataset_for_coarsen(current), zoom_number)
        _write_zoom_level(
            current,
            source_dataset=source_dataset,
            frequency=frequency,
            variables=variables,
            zoom_number=zoom_number,
            global_attrs=global_attrs,
            clean=clean,
            zarr_format=zarr_format,
        )


def _existing_zoom_numbers(source_dataset: str, frequency: str) -> tuple[int, ...]:
    """Return the zoom levels already present for one output frequency."""

    return tuple(
        zoom_number
        for zoom_number, _ in _existing_level_destinations(source_dataset, frequency)
    )


def _special_zoom_numbers_for_frequency(
    *,
    dataset: str,
    frequency: str,
    written_zoom_numbers: tuple[int, ...],
    highest_level_only: bool,
    coarsen_only: bool,
) -> tuple[int, ...]:
    """Resolve the zoom levels to use for special-case variables."""

    if written_zoom_numbers:
        return written_zoom_numbers

    existing_zoom_numbers = _existing_zoom_numbers(dataset, frequency)
    if existing_zoom_numbers:
        return existing_zoom_numbers

    return special_zoom_numbers(
        highest_level_only=highest_level_only,
        coarsen_only=coarsen_only,
    )


def _write_special_frequency(
    *,
    dataset: str,
    frequency: str,
    variable_names: tuple[str, ...],
    written_zoom_numbers: tuple[int, ...],
    highest_level_only: bool,
    coarsen_only: bool,
    zarr_format: int,
    clean: bool,
) -> None:
    """Write special-case variables for one output frequency."""

    zoom_numbers = _special_zoom_numbers_for_frequency(
        dataset=dataset,
        frequency=frequency,
        written_zoom_numbers=written_zoom_numbers,
        highest_level_only=highest_level_only,
        coarsen_only=coarsen_only,
    )
    write_special_variables(
        dataset=dataset,
        frequency=frequency,
        variable_names=variable_names,
        zoom_numbers=zoom_numbers,
        zarr_format=zarr_format,
        clean=clean,
        cmor_tables_dir=CMOR_TABLES_DIR,
        mapper_path=SOURCE_MAPPER_PATH,
    )


def map_grib_to_healpix(
    records: list[SourceRecord],
    *,
    dataset: str,
    frequencies: tuple[str, ...],
    requested_variables: tuple[str, ...],
    interval: tuple[Optional[date], Optional[date]] = (None, None),
    zarr_format: int = 2,
    use_inventory_cache: bool = True,
    use_input_cache: bool = False,
    use_record_threads: bool = False,
    weights_dir: Optional[str] = None,
    clean: bool = False,
    pyramid_strategy: str = "lazy",
    highest_level_only: bool = False,
    coarsen_only: bool = False,
) -> None:
    """Convert resolved GRIB records to per-frequency HEALPix Zarr pyramids."""

    _, special_requested = split_special_variables(requested_variables)
    grouped_records = group_records_by_frequency(records)
    if not grouped_records and not special_requested and not coarsen_only:
        raise ValueError("No matching source files were found for conversion.")
    records_by_frequency = {
        frequency: [record for record in records if record.frequency == frequency]
        for frequency in frequencies
    }

    log_stage(LOGGER, "convert_start", frequencies=",".join(frequencies), records=len(records))

    for frequency in frequencies:
        freq_records = grouped_records.get(frequency, [])
        frequency_records = records_by_frequency.get(frequency, [])
        special_requested_for_frequency = special_requested if frequency == "fx" else ()
        variable_names = _variable_names(frequency_records, special_requested_for_frequency)
        written_zoom_numbers: tuple[int, ...] = ()
        if coarsen_only:
            if not variable_names:
                variable_names = "unknown"
            log_stage(
                LOGGER,
                "frequency_start",
                frequency=frequency,
                variables=variable_names,
                mode="coarsen_only",
            )
            _coarsen_existing_frequency(
                source_dataset=dataset,
                frequency=frequency,
                variables=variable_names,
                zarr_format=zarr_format,
                clean=clean,
            )
            if special_requested_for_frequency:
                _write_special_frequency(
                    dataset=dataset,
                    frequency=frequency,
                    variable_names=special_requested_for_frequency,
                    written_zoom_numbers=_existing_zoom_numbers(dataset, frequency),
                    highest_level_only=highest_level_only,
                    coarsen_only=coarsen_only,
                    zarr_format=zarr_format,
                    clean=clean,
            )
            log_stage(LOGGER, "frequency_done", frequency=frequency, variables=variable_names)
            continue
        if not freq_records and not special_requested_for_frequency:
            continue
        if not variable_names:
            variable_names = _variable_names(freq_records)

        log_stage(
            LOGGER,
            "frequency_start",
            frequency=frequency,
            variables=variable_names,
            records=len(freq_records),
        )

        if freq_records:
            global_attrs = global_attrs_for_records(freq_records)
            ds = merge_frequency_dataset(
                freq_records,
                use_inventory_cache=use_inventory_cache,
                use_input_cache=use_input_cache,
                use_record_threads=use_record_threads,
                interval=interval,
            )
            ds.attrs.update(global_attrs)
            if "time" in ds.dims and ds.sizes.get("time", 0) == 0:
                log_stage(
                    LOGGER,
                    "frequency_skip_empty",
                    frequency=frequency,
                    variables=variable_names,
                )
                continue

            log_stage(
                LOGGER,
                "grib_merge_done",
                frequency=frequency,
                variables=variable_names,
                dims=dict(ds.sizes),
            )
            ds = normalise_reduced_gaussian_dataset(ds)
            if "cell" in ds.dims:
                ds = ds.chunk({"cell": -1})

            log_stage(LOGGER, "weight_calculation", frequency=frequency, variables=variable_names)
            max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
            weight_file = gd.cached_weights(
                ds,
                level=max_level,
                cache_path=weights_dir,
            )
            log_stage(
                LOGGER,
                "remap_start",
                frequency=frequency,
                variables=variable_names,
                max_level=max_level,
                weights=weight_file,
                strategy=pyramid_strategy,
            )
            if pyramid_strategy == "stepwise":
                finest = gd.regrid_to_healpix(
                    ds,
                    max_level,
                    weights_path=weight_file,
                )
                current = finest.load()
                written_zoom_numbers = (max_level,)
                log_stage(
                    LOGGER,
                    "remap_materialize_done",
                    frequency=frequency,
                    variables=variable_names,
                    zoom=max_level,
                )
                _write_zoom_level(
                    current,
                    source_dataset=dataset,
                    frequency=frequency,
                    variables=variable_names,
                    zoom_number=max_level,
                    global_attrs=global_attrs,
                    clean=clean,
                    zarr_format=zarr_format,
                )
                if not highest_level_only:
                    remaining_zoom_numbers = tuple(range(max_level - 1, -1, -1))
                    for zoom_number in remaining_zoom_numbers:
                        current = gd.coarsen_healpix(
                            _prepare_dataset_for_coarsen(current),
                            zoom_number,
                        )
                        _write_zoom_level(
                            current,
                            source_dataset=dataset,
                            frequency=frequency,
                            variables=variable_names,
                            zoom_number=zoom_number,
                            global_attrs=global_attrs,
                            clean=clean,
                            zarr_format=zarr_format,
                        )
                    written_zoom_numbers += remaining_zoom_numbers
            else:
                pyramid = gd.latlon_to_healpix_pyramid(
                    ds,
                    max_level=max_level,
                    weights_path=weight_file,
                )
                written_levels: list[int] = []
                for zoom_number, ds_level in pyramid.items():
                    written_levels.append(int(zoom_number))
                    _write_zoom_level(
                        ds_level,
                        source_dataset=dataset,
                        frequency=frequency,
                        variables=variable_names,
                        zoom_number=zoom_number,
                        global_attrs=global_attrs,
                        clean=clean,
                        zarr_format=zarr_format,
                    )
                    if highest_level_only:
                        break
                written_zoom_numbers = tuple(written_levels)

        if special_requested_for_frequency:
            _write_special_frequency(
                dataset=dataset,
                frequency=frequency,
                variable_names=special_requested_for_frequency,
                written_zoom_numbers=written_zoom_numbers,
                highest_level_only=highest_level_only,
                coarsen_only=coarsen_only,
                zarr_format=zarr_format,
                clean=clean,
            )

        log_stage(LOGGER, "frequency_done", frequency=frequency, variables=variable_names)


SOURCE_MAPPER_PATH = Path(__file__).resolve().parent.parent / "assets" / "source_mapper.json"
CMOR_TABLES_DIR = Path(__file__).resolve().parent.parent / "tables" / "era5-cmor-tables" / "Tables"


def update_healpix_attrs_only(
    records: list[SourceRecord],
    *,
    dataset: str,
    frequencies: tuple[str, ...],
    requested_variables: tuple[str, ...],
) -> None:
    """Refresh published variable attrs on existing Zarr stores without remapping."""

    _, special_requested = split_special_variables(requested_variables)
    records_by_frequency = {
        frequency: [record for record in records if record.frequency == frequency]
        for frequency in frequencies
    }

    for frequency in frequencies:
        freq_records = records_by_frequency.get(frequency, [])
        special_requested_for_frequency = special_requested if frequency == "fx" else ()
        variable_names = _variable_names(freq_records, special_requested_for_frequency)
        if not freq_records and not special_requested_for_frequency:
            continue

        log_stage(LOGGER, "attrs_only", frequency=frequency, variables=variable_names)
        global_attrs = (
            global_attrs_for_records(freq_records)
            if freq_records
            else global_attrs_for_dataset_frequency(dataset, frequency)
        )
        attrs_by_name = {
            record.variable: attrs_for_record(record)
            for record in freq_records
        }
        attrs_by_name.update(
            special_variable_attrs_by_name(
                dataset=dataset,
                frequency=frequency,
                variable_names=special_requested_for_frequency,
                cmor_tables_dir=CMOR_TABLES_DIR,
                mapper_path=SOURCE_MAPPER_PATH,
            )
        )
        for destination in existing_destinations_for_frequency(dataset, frequency):
            sync_global_attrs(global_attrs, destination)
            sync_named_variable_attrs(attrs_by_name, destination)
