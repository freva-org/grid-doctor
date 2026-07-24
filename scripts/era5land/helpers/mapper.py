"""Orchestration helpers for ERA5/ERA5-Land HEALPix conversion."""

import gc
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
    EmptySourceDataError,
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
    rechunk_zarr_store,
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


def _close_dataset_quietly(dataset: xr.Dataset | None) -> None:
    """Best-effort dataset cleanup for long-running batch conversions."""

    if dataset is None:
        return
    try:
        dataset.close()
    except Exception:
        pass


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


def _missing_frequency_variables(
    frequency_records: Iterable[SourceRecord],
    resolved_frequency_records: Iterable[SourceRecord],
) -> tuple[str, ...]:
    """Return requested variable names with no resolved source files."""

    requested = {record.variable for record in frequency_records}
    resolved = {record.variable for record in resolved_frequency_records}
    return tuple(sorted(requested - resolved))


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
    target_chunk_mb: int,
    output_path: str | Path | None = None,
    truncate_after: str | None = None,
) -> None:
    """Write one zoom level with consistent progress logging."""

    merged_attrs = dict(global_attrs)
    for key in PROTECTED_GRID_ATTRS:
        if key in dataset.attrs:
            merged_attrs[key] = dataset.attrs[key]
    dataset.attrs = merged_attrs
    destination = destination_for_level(
        source_dataset,
        frequency,
        zoom_number,
        output_path=output_path,
    )
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
        truncate_after=truncate_after,
        target_chunk_mb=target_chunk_mb,
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
    *,
    output_path: str | Path | None = None,
) -> list[tuple[int, str]]:
    """Return existing destinations for one frequency with parsed zoom levels."""

    destinations: list[tuple[int, str]] = []
    for destination in existing_destinations_for_frequency(
        source_dataset,
        frequency,
        output_path=output_path,
    ):
        match = LEVEL_RE.search(destination)
        if match is None:
            continue
        destinations.append((int(match.group("level")), destination))
    return sorted(destinations, reverse=True)


def rechunk_existing_healpix_stores(
    *,
    dataset: str,
    frequencies: tuple[str, ...],
    zarr_format: int,
    target_chunk_mb: int,
    highest_level_only: bool,
    output_path: str | Path | None = None,
) -> int:
    """Rechunk selected existing HEALPix Zarr stores before a rerun."""

    rewritten_count = 0
    for frequency in frequencies:
        destinations = _existing_level_destinations(
            dataset,
            frequency,
            output_path=output_path,
        )
        if highest_level_only and destinations:
            destinations = destinations[:1]

        for _zoom_number, destination in destinations:
            if rechunk_zarr_store(
                destination,
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            ):
                rewritten_count += 1
    return rewritten_count


def _coarsen_existing_frequency(
    *,
    source_dataset: str,
    frequency: str,
    variables: str,
    zarr_format: int,
    clean: bool,
    target_chunk_mb: int,
    output_path: str | Path | None = None,
    interval: tuple[Optional[date], Optional[date]] = (None, None),
    target_levels: tuple[int, ...] | None = None,
    truncate_after: str | None = None,
) -> tuple[int, ...]:
    """Build lower zoom levels from the highest existing Zarr store."""

    existing = _existing_level_destinations(
        source_dataset,
        frequency,
        output_path=output_path,
    )
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
    start, end = interval
    selected_levels = _resolve_requested_coarsen_levels(
        highest_level=highest_level,
        requested_levels=target_levels,
        available_levels=tuple(level for level, _ in existing),
    )
    written_levels: list[int] = []
    for zoom_number in selected_levels:
        source_level = zoom_number + 1
        source_destination = (
            highest_destination
            if source_level == highest_level
            else destination_for_level(
                source_dataset,
                frequency,
                source_level,
                output_path=output_path,
            )
        )
        current: xr.Dataset | None = xr.open_zarr(
            source_destination,
            consolidated=(zarr_format == 2),
        )
        try:
            if "time" in current.dims and (start is not None or end is not None):
                time_slice = slice(
                    start.isoformat() if start is not None else None,
                    end.isoformat() if end is not None else None,
                )
                current = current.sel(time=time_slice)
            global_attrs = dict(current.attrs)
            coarsened = gd.coarsen_healpix(
                _prepare_dataset_for_coarsen(current),
                zoom_number,
            )
            try:
                _write_zoom_level(
                    coarsened,
                    source_dataset=source_dataset,
                    frequency=frequency,
                    variables=variables,
                    zoom_number=zoom_number,
                    global_attrs=global_attrs,
                    clean=clean,
                    zarr_format=zarr_format,
                    target_chunk_mb=target_chunk_mb,
                    output_path=output_path,
                    truncate_after=truncate_after,
                )
            finally:
                _close_dataset_quietly(coarsened)
            written_levels.append(zoom_number)
        finally:
            _close_dataset_quietly(current)
    return tuple(written_levels)


def _resolve_requested_coarsen_levels(
    *,
    highest_level: int,
    requested_levels: tuple[int, ...] | None,
    available_levels: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve and validate the target levels for one coarsen-only pass."""

    if requested_levels is None:
        return tuple(range(highest_level - 1, -1, -1))

    invalid = [level for level in requested_levels if level >= highest_level]
    if invalid:
        invalid_text = ", ".join(str(level) for level in invalid)
        raise ValueError(
            f"Requested coarsen levels must be lower than the highest existing level "
            f"{highest_level}: {invalid_text}"
        )

    resolved_levels: list[int] = []
    available = set(int(level) for level in available_levels)
    for level in requested_levels:
        parent_level = level + 1
        if parent_level not in available:
            raise ValueError(
                f"Cannot coarsen level {level}: required parent level {parent_level} "
                "does not exist. Requested sparse coarsening assumes the immediate "
                "higher level is already present."
            )
        resolved_levels.append(level)
        available.add(level)

    return tuple(resolved_levels)


def _existing_zoom_numbers(
    source_dataset: str,
    frequency: str,
    *,
    output_path: str | Path | None = None,
) -> tuple[int, ...]:
    """Return the zoom levels already present for one output frequency."""

    return tuple(
        zoom_number
        for zoom_number, _ in _existing_level_destinations(
            source_dataset,
            frequency,
            output_path=output_path,
        )
    )


def _fallback_special_zoom_numbers(
    *,
    dataset: str,
    highest_level_only: bool,
    coarsen_only: bool,
    output_path: str | Path | None = None,
) -> tuple[int, ...]:
    """Infer special-variable zoom levels from existing non-`fx` outputs.

    The static `fx` products should mirror the highest HEALPix level already
    published for the data-bearing frequencies. We prefer monthly outputs
    first, then daily, then hourly, because those stores are typically present
    earlier in a publication workflow while still advertising the intended
    pyramid depth for the dataset.
    """

    for candidate_frequency in ("mon", "day", "1hr"):
        existing_zoom_numbers = _existing_zoom_numbers(
            dataset,
            candidate_frequency,
            output_path=output_path,
        )
        if not existing_zoom_numbers:
            continue
        max_level = existing_zoom_numbers[0]
        return special_zoom_numbers(
            max_level=max_level,
            highest_level_only=highest_level_only,
            coarsen_only=coarsen_only,
        )

    return special_zoom_numbers(
        highest_level_only=highest_level_only,
        coarsen_only=coarsen_only,
    )


def _special_zoom_numbers_for_frequency(
    *,
    dataset: str,
    frequency: str,
    written_zoom_numbers: tuple[int, ...],
    highest_level_only: bool,
    coarsen_only: bool,
    output_path: str | Path | None = None,
) -> tuple[int, ...]:
    """Resolve the zoom levels to use for special-case variables."""

    if written_zoom_numbers:
        return written_zoom_numbers

    existing_zoom_numbers = _existing_zoom_numbers(
        dataset,
        frequency,
        output_path=output_path,
    )
    if existing_zoom_numbers:
        return existing_zoom_numbers

    return _fallback_special_zoom_numbers(
        dataset=dataset,
        highest_level_only=highest_level_only,
        coarsen_only=coarsen_only,
        output_path=output_path,
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
    target_chunk_mb: int,
    output_path: str | Path | None = None,
) -> None:
    """Write special-case variables for one output frequency."""

    zoom_numbers = _special_zoom_numbers_for_frequency(
        dataset=dataset,
        frequency=frequency,
        written_zoom_numbers=written_zoom_numbers,
        highest_level_only=highest_level_only,
        coarsen_only=coarsen_only,
        output_path=output_path,
    )
    write_special_variables(
        dataset=dataset,
        frequency=frequency,
        variable_names=variable_names,
        zoom_numbers=zoom_numbers,
        zarr_format=zarr_format,
        clean=clean,
        target_chunk_mb=target_chunk_mb,
        cmor_tables_dir=CMOR_TABLES_DIR,
        mapper_path=SOURCE_MAPPER_PATH,
        output_path=output_path,
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
    drop_duplicate_time_rows: bool = True,
    weights_dir: Optional[str] = None,
    clean: bool = False,
    target_chunk_mb: int = 100,
    highest_level_only: bool = False,
    coarsen_only: bool = False,
    coarsen_levels: tuple[int, ...] | None = None,
    output_path: str | Path | None = None,
    coarsen_interval: tuple[Optional[date], Optional[date]] = (None, None),
    truncate_after: str | None = None,
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
        ds: xr.Dataset | None = None
        current: xr.Dataset | None = None
        finest: xr.Dataset | None = None
        if coarsen_only:
            if not variable_names:
                variable_names = "unknown"
            highest_existing = _existing_zoom_numbers(
                dataset,
                frequency,
                output_path=output_path,
            )
            if not highest_existing:
                raise ValueError(
                    f"No existing HEALPix Zarr stores found for frequency {frequency!r}."
                )
            selected_coarsen_levels = _resolve_requested_coarsen_levels(
                highest_level=highest_existing[0],
                requested_levels=coarsen_levels,
                available_levels=highest_existing,
            )
            log_stage(
                LOGGER,
                "frequency_start",
                frequency=frequency,
                variables=variable_names,
                mode="coarsen_only",
            )
            written_zoom_numbers = _coarsen_existing_frequency(
                source_dataset=dataset,
                frequency=frequency,
                variables=variable_names,
                zarr_format=zarr_format,
                clean=clean,
                target_chunk_mb=target_chunk_mb,
                output_path=output_path,
                interval=coarsen_interval,
                target_levels=selected_coarsen_levels,
                truncate_after=truncate_after,
            )
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
                    target_chunk_mb=target_chunk_mb,
                    output_path=output_path,
                )
            log_stage(LOGGER, "frequency_done", frequency=frequency, variables=variable_names)
            continue
        if not freq_records and not special_requested_for_frequency:
            continue
        if not variable_names:
            variable_names = _variable_names(freq_records)

        missing_variables = _missing_frequency_variables(
            frequency_records,
            freq_records,
        )
        if missing_variables:
            missing_names = ",".join(missing_variables)
            LOGGER.warning(
                "Skipping %s frequency for %s because source data is missing for %s.",
                frequency,
                dataset,
                missing_names,
            )
            log_stage(
                LOGGER,
                "frequency_skip_missing",
                frequency=frequency,
                variables=variable_names,
                missing=missing_names,
            )
            continue

        log_stage(
            LOGGER,
            "frequency_start",
            frequency=frequency,
            variables=variable_names,
            records=len(freq_records),
        )

        try:
            if freq_records:
                global_attrs = global_attrs_for_records(freq_records)
                
                try:
                    ds = merge_frequency_dataset(
                        freq_records,
                        use_inventory_cache=use_inventory_cache,
                        use_input_cache=use_input_cache,
                        drop_duplicate_time_rows=drop_duplicate_time_rows,
                        interval=interval,
                    )
                except EmptySourceDataError as exc:
                    LOGGER.warning(
                        "Skipping %s frequency for %s because no source data was "
                        "found in the requested interval: %s",
                        frequency,
                        dataset,
                        exc,
                    )
                    log_stage(
                        LOGGER,
                        "frequency_skip_empty",
                        frequency=frequency,
                        variables=variable_names,
                        reason=str(exc),
                    )
                    continue

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
                ds = normalise_reduced_gaussian_dataset(
                    ds,
                    use_cache=use_inventory_cache,
                )
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
                    strategy="stepwise",
                )
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
                    target_chunk_mb=target_chunk_mb,
                    output_path=output_path,
                    truncate_after=truncate_after,
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
                            target_chunk_mb=target_chunk_mb,
                            output_path=output_path,
                            truncate_after=truncate_after,
                        )
                    written_zoom_numbers += remaining_zoom_numbers

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
                    target_chunk_mb=target_chunk_mb,
                    output_path=output_path,
                )

            log_stage(LOGGER, "frequency_done", frequency=frequency, variables=variable_names)
        except xr.AlignmentError as exc:
            LOGGER.warning(
                "Skipping %s frequency for %s because variable time coverage is incomplete: %s",
                frequency,
                dataset,
                exc,
            )
            log_stage(
                LOGGER,
                "frequency_skip_incomplete",
                frequency=frequency,
                variables=variable_names,
            )
        finally:
            _close_dataset_quietly(current)
            if finest is not current:
                _close_dataset_quietly(finest)
            _close_dataset_quietly(ds)
            gc.collect()


SOURCE_MAPPER_PATH = Path(__file__).resolve().parent.parent / "assets" / "source_mapper.json"
CMOR_TABLES_DIR = Path(__file__).resolve().parent.parent / "tables" / "era5-cmor-tables" / "Tables"


def update_healpix_attrs_only(
    records: list[SourceRecord],
    *,
    dataset: str,
    frequencies: tuple[str, ...],
    requested_variables: tuple[str, ...],
    output_path: str | Path | None = None,
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
        for destination in existing_destinations_for_frequency(
            dataset,
            frequency,
            output_path=output_path,
        ):
            sync_global_attrs(global_attrs, destination)
            sync_named_variable_attrs(attrs_by_name, destination)
