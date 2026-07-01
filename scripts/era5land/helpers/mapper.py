"""Orchestration helpers for ERA5/ERA5-Land HEALPix conversion."""

from datetime import date
import logging
from pathlib import Path
import re
from typing import Optional

import grid_doctor as gd
import xarray as xr

from .datasets import (
    merge_frequency_dataset,
    normalise_reduced_gaussian_dataset,
    select_time_interval,
)
from .file_fetcher import SourceRecord
from .formatter import (
    destination_for_level,
    existing_destinations_for_frequency,
    group_records_by_frequency,
)
from .metadata import attrs_for_record, global_attrs_for_records
from .zarr_publisher import (
    sync_global_attrs,
    sync_named_variable_attrs,
    update_zarr_store,
)

LOGGER = logging.getLogger(__name__)
LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")
PROTECTED_GRID_ATTRS = {
    "healpix_level",
    "healpix_nside",
    "healpix_order",
    "grid_doctor_version",
    "grid_doctor_method",
    "grid_doctor_coarsened_from_level",
}


def _variable_names(records: list[SourceRecord]) -> str:
    """Return a stable comma-separated list of variable names for logging."""

    return ",".join(sorted({record.variable for record in records}))


def _write_zoom_level(
    dataset,
    *,
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
    destination = destination_for_level(frequency, zoom_number)
    LOGGER.info(
        (
            "stage=zarr_write_start frequency=%s variables=%s "
            "zoom=%d destination=%s"
        ),
        frequency,
        variables,
        zoom_number,
        destination,
    )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    update_zarr_store(
        dataset,
        destination,
        clean=clean,
        zarr_format=zarr_format,
    )
    LOGGER.info(
        "stage=zarr_write_done frequency=%s variables=%s zoom=%d",
        frequency,
        variables,
        zoom_number,
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


def _existing_level_destinations(frequency: str) -> list[tuple[int, str]]:
    """Return existing destinations for one frequency with parsed zoom levels."""

    destinations: list[tuple[int, str]] = []
    for destination in existing_destinations_for_frequency(frequency):
        match = LEVEL_RE.search(destination)
        if match is None:
            continue
        destinations.append((int(match.group("level")), destination))
    return sorted(destinations, reverse=True)


def _coarsen_existing_frequency(
    *,
    frequency: str,
    variables: str,
    zarr_format: int,
    clean: bool,
) -> None:
    """Build lower zoom levels from the highest existing Zarr store."""

    existing = _existing_level_destinations(frequency)
    if not existing:
        raise ValueError(
            f"No existing HEALPix Zarr stores found for frequency {frequency!r}."
        )

    highest_level, highest_destination = existing[0]
    LOGGER.info(
        "stage=coarsen_source_open frequency=%s variables=%s zoom=%d source=%s",
        frequency,
        variables,
        highest_level,
        highest_destination,
    )
    current = xr.open_zarr(highest_destination, consolidated=(zarr_format == 2)).load()
    global_attrs = dict(current.attrs)
    for zoom_number in range(highest_level - 1, -1, -1):
        LOGGER.info(
            "stage=coarsen_start frequency=%s variables=%s zoom=%d",
            frequency,
            variables,
            zoom_number,
        )
        current = gd.coarsen_healpix(_prepare_dataset_for_coarsen(current), zoom_number)
        LOGGER.info(
            "stage=coarsen_done frequency=%s variables=%s zoom=%d",
            frequency,
            variables,
            zoom_number,
        )
        _write_zoom_level(
            current,
            frequency=frequency,
            variables=variables,
            zoom_number=zoom_number,
            global_attrs=global_attrs,
            clean=clean,
            zarr_format=zarr_format,
        )


def map_grib_to_healpix(
    records: list[SourceRecord],
    *,
    frequencies: tuple[str, ...],
    interval: tuple[Optional[date], Optional[date]] = (None, None),
    zarr_format: int = 2,
    use_cache: bool = False,
    weights_dir: Optional[str] = None,
    clean: bool = False,
    pyramid_strategy: str = "lazy",
    highest_level_only: bool = False,
    coarsen_only: bool = False,
) -> None:
    """Convert resolved GRIB records to per-frequency HEALPix Zarr pyramids."""

    grouped_records = group_records_by_frequency(records)
    if not grouped_records and not coarsen_only:
        raise ValueError("No matching source files were found for conversion.")
    records_by_frequency = {
        frequency: [record for record in records if record.frequency == frequency]
        for frequency in frequencies
    }

    LOGGER.info(
        "stage=convert_start frequencies=%s records=%d",
        ",".join(frequencies),
        len(records),
    )

    for frequency in frequencies:
        freq_records = grouped_records.get(frequency, [])
        variable_names = _variable_names(records_by_frequency.get(frequency, []))
        if coarsen_only:
            if not variable_names:
                variable_names = "unknown"
            LOGGER.info(
                "stage=frequency_start frequency=%s variables=%s mode=coarsen_only",
                frequency,
                variable_names,
            )
            _coarsen_existing_frequency(
                frequency=frequency,
                variables=variable_names,
                zarr_format=zarr_format,
                clean=clean,
            )
            LOGGER.info(
                "stage=frequency_done frequency=%s variables=%s",
                frequency,
                variable_names,
            )
            continue
        if not freq_records:
            continue
        if not variable_names:
            variable_names = _variable_names(freq_records)

        LOGGER.info(
            "stage=frequency_start frequency=%s variables=%s records=%d",
            frequency,
            variable_names,
            len(freq_records),
        )
        global_attrs = global_attrs_for_records(freq_records)
        LOGGER.info(
            "stage=grib_merge_start frequency=%s variables=%s",
            frequency,
            variable_names,
        )
        ds = merge_frequency_dataset(freq_records, use_cache=use_cache)
        ds.attrs.update(global_attrs)
        ds = select_time_interval(ds, interval)
        if "time" in ds.dims and ds.sizes.get("time", 0) == 0:
            LOGGER.info(
                "stage=frequency_skip_empty frequency=%s variables=%s",
                frequency,
                variable_names,
            )
            continue

        LOGGER.info(
            "stage=grib_merge_done frequency=%s variables=%s dims=%s",
            frequency,
            variable_names,
            dict(ds.sizes),
        )
        ds = normalise_reduced_gaussian_dataset(ds)
        if "cell" in ds.dims:
            ds = ds.chunk({"cell": -1})

        LOGGER.info(
            "stage=weight_calculation frequency=%s variables=%s",
            frequency,
            variable_names,
        )
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
        weight_file = gd.cached_weights(
            ds,
            level=max_level,
            cache_path=weights_dir,
        )
        LOGGER.info(
            (
                "stage=remap_start frequency=%s variables=%s max_level=%d "
                "weights=%s strategy=%s"
            ),
            frequency,
            variable_names,
            max_level,
            weight_file,
            pyramid_strategy,
        )
        if pyramid_strategy == "stepwise":
            finest = gd.regrid_to_healpix(
                ds,
                max_level,
                weights_path=weight_file,
            )
            LOGGER.info(
                "stage=remap_graph_ready frequency=%s variables=%s zoom_levels=%d",
                frequency,
                variable_names,
                max_level + 1,
            )
            LOGGER.info(
                "stage=remap_materialize_start frequency=%s variables=%s zoom=%d",
                frequency,
                variable_names,
                max_level,
            )
            current = finest.load()
            LOGGER.info(
                "stage=remap_materialize_done frequency=%s variables=%s zoom=%d",
                frequency,
                variable_names,
                max_level,
            )
            _write_zoom_level(
                current,
                frequency=frequency,
                variables=variable_names,
                zoom_number=max_level,
                global_attrs=global_attrs,
                clean=clean,
                zarr_format=zarr_format,
            )
            if highest_level_only:
                LOGGER.info(
                    "stage=highest_level_only_done frequency=%s variables=%s zoom=%d",
                    frequency,
                    variable_names,
                    max_level,
                )
                LOGGER.info(
                    "stage=frequency_done frequency=%s variables=%s",
                    frequency,
                    variable_names,
                )
                continue
            for zoom_number in range(max_level - 1, -1, -1):
                LOGGER.info(
                    "stage=coarsen_start frequency=%s variables=%s zoom=%d",
                    frequency,
                    variable_names,
                    zoom_number,
                )
                current = gd.coarsen_healpix(
                    _prepare_dataset_for_coarsen(current),
                    zoom_number,
                )
                LOGGER.info(
                    "stage=coarsen_done frequency=%s variables=%s zoom=%d",
                    frequency,
                    variable_names,
                    zoom_number,
                )
                _write_zoom_level(
                    current,
                    frequency=frequency,
                    variables=variable_names,
                    zoom_number=zoom_number,
                    global_attrs=global_attrs,
                    clean=clean,
                    zarr_format=zarr_format,
                )
        else:
            pyramid = gd.latlon_to_healpix_pyramid(
                ds,
                max_level=max_level,
                weights_path=weight_file,
            )
            LOGGER.info(
                "stage=remap_graph_ready frequency=%s variables=%s zoom_levels=%d",
                frequency,
                variable_names,
                len(pyramid),
            )
            for zoom_number, dataset in pyramid.items():
                _write_zoom_level(
                    dataset,
                    frequency=frequency,
                    variables=variable_names,
                    zoom_number=zoom_number,
                    global_attrs=global_attrs,
                    clean=clean,
                    zarr_format=zarr_format,
                )
                if highest_level_only:
                    LOGGER.info(
                        "stage=highest_level_only_done frequency=%s variables=%s zoom=%d",
                        frequency,
                        variable_names,
                        zoom_number,
                    )
                    break
        LOGGER.info(
            "stage=frequency_done frequency=%s variables=%s",
            frequency,
            variable_names,
        )


def update_healpix_attrs_only(
    records: list[SourceRecord],
    *,
    frequencies: tuple[str, ...],
) -> None:
    """Refresh published variable attrs on existing Zarr stores without remapping."""

    records_by_frequency = {
        frequency: [record for record in records if record.frequency == frequency]
        for frequency in frequencies
    }

    for frequency in frequencies:
        freq_records = records_by_frequency.get(frequency, [])
        if not freq_records:
            continue

        LOGGER.info(
            "stage=attrs_only frequency=%s variables=%s",
            frequency,
            _variable_names(freq_records),
        )
        global_attrs = global_attrs_for_records(freq_records)
        attrs_by_name = {
            record.variable: attrs_for_record(record)
            for record in freq_records
        }
        for destination in existing_destinations_for_frequency(frequency):
            sync_global_attrs(global_attrs, destination)
            sync_named_variable_attrs(attrs_by_name, destination)
