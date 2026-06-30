"""Orchestration helpers for ERA5/ERA5-Land HEALPix conversion."""

from datetime import date
from pathlib import Path
from typing import Optional

import grid_doctor as gd

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


def map_grib_to_healpix(
    records: list[SourceRecord],
    *,
    frequencies: tuple[str, ...],
    interval: tuple[Optional[date], Optional[date]] = (None, None),
    time_chunk: int = 48,
    zarr_format: int = 2,
    use_cache: bool = False,
    weights_dir: Optional[str] = None,
    clean: bool = False,
) -> None:
    """Convert resolved GRIB records to per-frequency HEALPix Zarr pyramids."""

    grouped_records = group_records_by_frequency(records)
    if not grouped_records:
        raise ValueError("No matching source files were found for conversion.")

    for frequency in frequencies:
        freq_records = grouped_records.get(frequency, [])
        if not freq_records:
            continue

        global_attrs = global_attrs_for_records(freq_records)
        ds = merge_frequency_dataset(freq_records, use_cache=use_cache)
        ds.attrs.update(global_attrs)
        ds = select_time_interval(ds, interval)
        if "time" in ds.dims and ds.sizes.get("time", 0) == 0:
            continue
        ds = normalise_reduced_gaussian_dataset(ds)
        if "cell" in ds.dims:
            ds = ds.chunk({"cell": -1})
        if time_chunk and "time" in ds.dims:
            ds = ds.chunk({"time": time_chunk})

        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
        weight_file = gd.cached_weights(
            ds,
            level=max_level,
            cache_path=weights_dir,
        )
        pyramid = gd.latlon_to_healpix_pyramid(
            ds,
            max_level=max_level,
            weights_path=weight_file,
        )
        for zoom_number, dataset in pyramid.items():
            dataset.attrs.update(global_attrs)
            destination = destination_for_level(frequency, zoom_number)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            update_zarr_store(
                dataset,
                destination,
                clean=clean,
                zarr_format=zarr_format,
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

        global_attrs = global_attrs_for_records(freq_records)
        attrs_by_name = {
            record.variable: attrs_for_record(record)
            for record in freq_records
        }
        for destination in existing_destinations_for_frequency(frequency):
            sync_global_attrs(global_attrs, destination)
            sync_named_variable_attrs(attrs_by_name, destination)
