"""Zarr publication helpers for ERA5/ERA5-Land outputs."""

import gc
import logging
import re
import shutil
import uuid
from collections.abc import Hashable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from .datasets import normalise_published_dataset
from .file_fetcher import SOURCE_MAPPER
from .formatter import merge_dataset_root
from .metadata import (
    LAST_DATA_UPDATE_ATTR,
    LAST_PERMANENT_UPDATE_ATTR,
    clean_output_attrs,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_CHUNK_MB = 100
LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")


def _string_name(name: Hashable) -> str:
    """Return a supported xarray variable or dimension name."""

    if not isinstance(name, str):
        raise TypeError(f"Expected a string name, got {name!r}")
    return name


def _is_selected_level_store(
    store: Path,
    levels: tuple[int, ...] | None,
) -> bool:
    """Return whether a level store is included by an optional selection."""

    match = LEVEL_RE.search(store.name)
    return match is not None and (levels is None or int(match.group("level")) in levels)


def _select_merge_interval(
    dataset: xr.Dataset,
    interval: tuple[date | None, date | None] | None,
) -> xr.Dataset:
    """Select the requested inclusive date interval from a source dataset.

    Static datasets are returned unchanged. Xarray treats an ISO date string
    as the complete calendar day, so the end date remains inclusive for
    sub-daily data.
    """

    if interval is None or "time" not in dataset.dims:
        return dataset

    start, end = interval
    return dataset.sel(
        time=slice(
            start.isoformat() if start is not None else None,
            end.isoformat() if end is not None else None,
        )
    )


def _stamp_data_update_attrs(dataset: xr.Dataset) -> None:
    """Set a fresh data-update timestamp on every published data variable."""

    if not dataset.data_vars:
        return

    update_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    for data in dataset.data_vars.values():
        data.attrs[LAST_DATA_UPDATE_ATTR] = update_timestamp


def _resolve_target_chunks(
    da: xr.DataArray,
    *,
    target_bytes: int,
    cell_size: int,
) -> tuple[int, ...]:
    """Derive map-style chunks that preserve the full horizontal field.

    For HEALPix outputs we keep ``cell`` as one full chunk. When a vertical
    dimension such as ``plev`` is present, we chunk it as ``1`` so each chunk
    represents one complete map at one level across a bounded number of time
    steps. The remaining byte budget is then used to size the ``time`` chunk.
    """

    full_cell_chunk = int(da.sizes["cell"])
    non_time_extent = 1
    for dim in da.dims:
        if dim == "time":
            continue
        if dim == "plev":
            non_time_extent *= 1
            continue
        non_time_extent *= int(da.sizes[dim])

    if non_time_extent <= 0:
        non_time_extent = 1

    chunks: list[int] = []
    for dim in da.dims:
        if dim == "time":
            chunks.append(int(max(1, target_bytes // (cell_size * non_time_extent))))
        elif dim == "plev":
            chunks.append(1)
        elif dim == "cell":
            chunks.append(full_cell_chunk)
        else:
            chunks.append(int(da.sizes[dim]))

    return tuple(chunks)


def _fill_value_for_dtype(dtype: np.dtype[Any]) -> Any:
    """Return a sensible missing value for a dtype."""

    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    return None


def _make_fill_variable(template: xr.DataArray, name: str, time_size: int) -> xr.DataArray:
    """Create a placeholder variable for appended time slices."""

    if "time" not in template.dims:
        values = np.asarray(template.values)
        coords = template.coords
    else:
        values = np.asarray(template.isel(time=slice(0, time_size)).values)
        coords = template.isel(time=slice(0, time_size)).coords

    fill = _fill_value_for_dtype(template.dtype)
    data = np.full(values.shape, fill, dtype=template.dtype)
    array = xr.DataArray(data, dims=template.dims, coords=coords, attrs=template.attrs)
    array.name = name
    return array


def _pad_missing_existing_vars_for_append(
    candidate: xr.Dataset,
    existing: xr.Dataset,
) -> xr.Dataset:
    """Pad appended time slices with placeholder arrays for existing variables."""

    padded = candidate.copy()
    time_size = padded.sizes.get("time", 0)
    for raw_name, data in existing.data_vars.items():
        name = _string_name(raw_name)
        if name not in padded.data_vars:
            padded[name] = _make_fill_variable(data, name, time_size)

    ordered = [name for name in existing.data_vars if name in padded.data_vars]
    ordered.extend(name for name in padded.data_vars if name not in ordered)
    return padded[ordered]


def _encoding_for_target_chunks(
    dataset: xr.Dataset,
    *,
    target_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return Zarr encoding that keeps chunk sizes near the requested budget."""

    target_bytes = target_mb * 1024**2
    encoding: dict[str, dict[str, tuple[int, ...]]] = {}

    for raw_name, da in dataset.data_vars.items():
        name = _string_name(raw_name)
        if "cell" not in da.dims:
            continue

        encoding[name] = {
            "chunks": _resolve_target_chunks(
                da,
                target_bytes=target_bytes,
                cell_size=da.dtype.itemsize,
            )
        }

    return encoding


def _write_dataset(
    dataset: xr.Dataset,
    destination: str,
    *,
    mode: str,
    zarr_format: int,
    append_dim: str | None = None,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> None:
    """Write one dataset to a Zarr store with consistent options."""

    dataset = normalise_published_dataset(dataset)
    options: dict[str, Any] = {
        "mode": mode,
        "zarr_format": zarr_format,
        "consolidated": (zarr_format == 2),
    }
    if append_dim is not None:
        options["append_dim"] = append_dim
        # Let xarray rechunk the append region when its start is not aligned
        # with the existing Zarr chunk grid.
        options["align_chunks"] = True
        dataset.to_zarr(destination, **options)
        return

    encoding = _encoding_for_target_chunks(dataset, target_mb=target_chunk_mb)

    # Ensure Dask chunks match the explicit Zarr chunks.
    # Otherwise xarray may reject writes because one Zarr chunk overlaps
    # multiple Dask chunks.
    chunk_map: dict[str, int] = {}
    for name, enc in encoding.items():
        if name in dataset:
            dims = dataset[name].dims
            chunks = enc.get("chunks")
            if chunks is not None:
                for dim, chunk in zip(dims, chunks, strict=True):
                    chunk_map[_string_name(dim)] = chunk

    if chunk_map:
        dataset = dataset.chunk(chunk_map)

    dataset.to_zarr(
        destination,
        **options,
        encoding=encoding,
    )


def _rewrite_dataset_via_temp(
    dataset: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> None:
    """Rewrite a store via a temporary path to avoid reading and writing it in place."""

    dataset = normalise_published_dataset(dataset)
    destination_path = Path(destination)
    temp_path = destination_path.parent / (f".{destination_path.name}.tmp-{uuid.uuid4().hex}")

    try:
        _write_dataset(
            dataset,
            str(temp_path),
            mode="w",
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.move(str(temp_path), str(destination_path))
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def rechunk_zarr_store(
    destination: str,
    *,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> bool:
    """Rewrite one existing Zarr store with a new target chunk size.

    Parameters
    ----------
    destination:
        Path to the target Zarr store.
    zarr_format:
        Output Zarr format version used when rewriting the store.
    target_chunk_mb:
        Approximate chunk-size budget in megabytes used to derive the written
        Zarr chunk layout.

    Returns
    -------
    bool
        ``True`` when the store existed and was rewritten, else ``False``.
    """

    path = Path(destination)
    if not path.exists():
        return False

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))
    try:
        LOGGER.info(
            "📦 Rechunking existing Zarr store %s with target chunk size %s MB",
            destination,
            target_chunk_mb,
        )
        _rewrite_dataset_via_temp(
            existing,
            destination,
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )
        return True
    finally:
        existing.close()
        gc.collect()


def _replace_public_attrs(zarr_array, attrs: dict[str, Any]) -> bool:
    keep = {}
    for key in ("_ARRAY_DIMENSIONS", "coordinates", "grid_mapping"):
        if key in zarr_array.attrs:
            keep[key] = zarr_array.attrs[key]

    attrs = dict(attrs)
    previous_data_update = zarr_array.attrs.get(LAST_DATA_UPDATE_ATTR)
    if previous_data_update and not attrs.get(LAST_DATA_UPDATE_ATTR):
        attrs[LAST_DATA_UPDATE_ATTR] = previous_data_update

    previous_permanent_update = zarr_array.attrs.get(LAST_PERMANENT_UPDATE_ATTR)
    requested_permanent_update = attrs.get(LAST_PERMANENT_UPDATE_ATTR)
    if previous_permanent_update and requested_permanent_update:
        attrs[LAST_PERMANENT_UPDATE_ATTR] = max(
            str(previous_permanent_update),
            str(requested_permanent_update),
        )
    elif previous_permanent_update and not requested_permanent_update:
        attrs[LAST_PERMANENT_UPDATE_ATTR] = previous_permanent_update

    new_attrs = {**keep, **attrs}
    old_attrs = dict(zarr_array.attrs)

    if old_attrs == new_attrs:
        return False

    zarr_array.attrs.clear()
    zarr_array.attrs.update(new_attrs)
    return True


def _sync_global_attrs(attrs: dict[str, Any], destination: str) -> None:
    """Update root attrs in one Zarr store without touching data chunks."""

    root = zarr.open_group(destination, mode="a")
    existing_attrs = dict(root.attrs)
    new_attrs = dict(attrs)

    if existing_attrs == {**existing_attrs, **new_attrs}:
        return

    root.attrs.update(new_attrs)
    zarr.consolidate_metadata(destination)


def sync_global_attrs(attrs: dict[str, Any], destination: str) -> None:
    """Update store-level attrs without touching data chunks."""

    _sync_global_attrs(attrs, destination)


def _sync_variable_attrs(dataset: xr.Dataset, destination: str) -> None:
    """Overwrite destination variable attrs with attrs from the latest dataset."""

    root = zarr.open_group(destination, mode="a")
    changed = False

    for raw_name, data in dataset.data_vars.items():
        name = _string_name(raw_name)
        if name not in root:
            continue
        attrs = clean_output_attrs(dict(data.attrs))
        changed |= _replace_public_attrs(root[name], attrs)

    if changed:
        zarr.consolidate_metadata(destination)


def _sync_dataset_metadata(dataset: xr.Dataset, destination: str) -> None:
    """Synchronize root and variable metadata after a publication update."""

    _sync_global_attrs(dict(dataset.attrs), destination)
    _sync_variable_attrs(dataset, destination)


def sync_named_variable_attrs(
    attrs_by_name: dict[str, dict[str, Any]],
    destination: str,
) -> None:
    """Update attrs in one Zarr store without touching data chunks."""

    root = zarr.open_group(destination, mode="a")
    changed = False

    for name, attrs in attrs_by_name.items():
        if name not in root:
            continue
        changed |= _replace_public_attrs(root[name], attrs)

    if changed:
        zarr.consolidate_metadata(destination)


def _preserve_update_attrs(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    merged: xr.Dataset,
) -> xr.Dataset:
    """Preserve monotonic update watermarks when rebuilding a store."""

    for name in merged.data_vars:
        sources = [source[name] for source in (existing, candidate) if name in source]
        values = [
            source.attrs[LAST_PERMANENT_UPDATE_ATTR] for source in sources if LAST_PERMANENT_UPDATE_ATTR in source.attrs
        ]
        if values:
            merged[name].attrs[LAST_PERMANENT_UPDATE_ATTR] = max(map(str, values))
    return merged


def _merge_time_updates(existing: xr.Dataset, candidate: xr.Dataset) -> xr.Dataset:
    """Merge disjoint time slices, preferring candidate values for rewritten times.

    Overlapping timestamps are handled before this helper is called, so the
    remaining merge only needs to combine disjoint slices along the time axis.
    Using ``combine_first`` here can trigger a very large outer alignment across
    multidimensional coordinates, so we rebuild the dataset with a concat-based
    merge instead.
    """

    candidate = _pad_missing_existing_vars_for_append(candidate, existing)
    existing_only = existing.drop_sel(time=candidate.indexes["time"], errors="ignore")
    merged = xr.concat(
        [existing_only, candidate],
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )
    if "time" in merged.coords:
        merged = merged.sortby("time")
    return _preserve_update_attrs(existing, candidate, merged)


def _requires_vertical_rewrite(existing: xr.Dataset, candidate: xr.Dataset) -> bool:
    """Return True when a candidate updates only a subset of vertical levels."""

    if "plev" not in existing.indexes or "plev" not in candidate.indexes:
        return False
    return not candidate.indexes["plev"].equals(existing.indexes["plev"])


def _contiguous_slices(indices: np.ndarray) -> list[slice]:
    """Convert sorted integer indices into contiguous slices."""

    if indices.size == 0:
        return []

    slices: list[slice] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            slices.append(slice(start, previous + 1))
            start = current
        previous = current
    slices.append(slice(start, previous + 1))
    return slices


def _can_append_new_times(existing: xr.Dataset, candidate: xr.Dataset) -> bool:
    """Return True when candidate times can be appended at the end of existing time."""

    if "time" not in existing.dims or "time" not in candidate.dims:
        return False
    if existing.sizes.get("time", 0) == 0 or candidate.sizes.get("time", 0) == 0:
        return False

    existing_times = existing.indexes["time"]
    candidate_times = candidate.indexes["time"]
    overlap = candidate_times.intersection(existing_times)
    if len(overlap) > 0:
        return False

    try:
        return bool(candidate_times.min() > existing_times.max())
    except TypeError:
        return False


def _write_missing_variables(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> xr.Dataset:
    """Add variables missing from an existing store across the current time axis."""

    missing = [_string_name(name) for name in candidate.data_vars if name not in existing.data_vars]
    if not missing:
        return existing

    root = zarr.open_group(destination, mode="a")
    encoding = _encoding_for_target_chunks(
        candidate[missing],
        target_mb=target_chunk_mb,
    )

    for name in missing:
        data = candidate[name]
        shape = tuple(int(existing.sizes.get(dim, data.sizes[dim])) for dim in data.dims)
        array = root.create_dataset(  # type: ignore[attr-defined]
            name,
            shape=shape,
            chunks=encoding.get(name, {}).get("chunks", shape),
            dtype=data.dtype,
            fill_value=_fill_value_for_dtype(data.dtype),
        )
        attrs = clean_output_attrs(dict(data.attrs))
        attrs["_ARRAY_DIMENSIONS"] = list(data.dims)
        for key in ("coordinates", "grid_mapping"):
            value = data.attrs.get(key)
            if value not in ("", None):
                attrs[key] = value
        array.attrs.update(attrs)

    if zarr_format == 2:
        zarr.consolidate_metadata(destination)
    return existing


def _merge_static_updates(existing: xr.Dataset, candidate: xr.Dataset) -> xr.Dataset:
    """Merge non-time variables, preferring values from *candidate*."""

    merged = candidate.combine_first(existing)
    return _preserve_update_attrs(existing, candidate, merged)


def _align_to_existing_chunks(
    dataset: xr.Dataset,
    existing: xr.Dataset,
) -> xr.Dataset:
    """Rechunk a dataset to the Zarr chunks already used by an existing store."""

    aligned = dataset.copy(deep=False)
    chunk_map: dict[str, int] = {}
    for raw_name, data in aligned.data_vars.items():
        name = _string_name(raw_name)
        if name not in existing:
            continue
        chunks = existing[name].encoding.get("chunks")
        if chunks is None:
            continue
        for dim, chunk in zip(data.dims, chunks, strict=True):
            chunk_map[_string_name(dim)] = int(chunk)
        aligned[name].encoding = dict(aligned[name].encoding)
        aligned[name].encoding["chunks"] = tuple(int(size) for size in chunks)

    return aligned.chunk(chunk_map) if chunk_map else aligned


def _append_new_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> bool:
    """Append strictly newer time slices to an existing store."""

    existing_time_strings = set(map(str, existing["time"].values))
    new_time_values = [value for value in candidate["time"].values if str(value) not in existing_time_strings]
    if not new_time_values:
        return False

    append_ds = _pad_missing_existing_vars_for_append(
        candidate.sel(time=new_time_values),
        existing,
    )
    append_ds = _align_to_existing_chunks(append_ds, existing)
    _write_dataset(
        append_ds,
        destination,
        mode="a",
        append_dim="time",
        zarr_format=zarr_format,
    )
    return True


def _rewrite_overlapping_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> bool:
    """Rewrite only the overlapping time regions in an existing store."""

    overlap = candidate.indexes["time"].intersection(existing.indexes["time"])
    if len(overlap) == 0:
        return False

    time_positions = existing.indexes["time"].get_indexer(overlap)
    time_slices = _contiguous_slices(np.asarray(time_positions, dtype=np.int64))
    for time_slice in time_slices:
        region_times = existing["time"].values[time_slice]
        region_ds = normalise_published_dataset(candidate.sel(time=region_times))
        to_drop = {name for name, var in region_ds.variables.items() if "time" not in var.dims}
        region_ds = region_ds.drop_vars(to_drop, errors="ignore")
        region_ds = _align_to_existing_chunks(region_ds, existing)
        region = {"time": time_slice}
        region_ds.to_zarr(
            destination,
            mode="r+",
            region=region,
            zarr_format=zarr_format,
            consolidated=(zarr_format == 2),
            align_chunks=True,
        )
    return True


def update_zarr_store(
    dataset: xr.Dataset,
    destination: str,
    *,
    clean: bool,
    zarr_format: int,
    truncate_after: str | None = None,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> None:
    """Incrementally update or recreate one destination Zarr store."""

    dataset = normalise_published_dataset(dataset)
    path = Path(destination)
    if clean or not path.exists():
        _stamp_data_update_attrs(dataset)
        _write_dataset(
            dataset,
            destination,
            mode="w",
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )
        _sync_dataset_metadata(dataset, destination)
        return

    if truncate_after is not None:
        from .cleanup import truncate_zarr_store_after

        truncate_zarr_store_after(
            destination,
            cutoff=truncate_after,
            zarr_format=zarr_format,
        )

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))
    try:
        if _requires_vertical_rewrite(existing, dataset):
            _stamp_data_update_attrs(dataset)
            merged = dataset.combine_first(existing)
            if "time" in merged.coords:
                merged = merged.sortby("time")
            merged = _preserve_update_attrs(existing, dataset, merged)
            _rewrite_dataset_via_temp(
                merged,
                destination,
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            )
            _sync_dataset_metadata(dataset, destination)
            return

        if "time" not in dataset.dims or "time" not in existing.dims:
            missing = [name for name in dataset.data_vars if name not in existing.data_vars]
            overlapping = [name for name in dataset.data_vars if name in existing.data_vars]
            if overlapping or missing:
                _stamp_data_update_attrs(dataset)
            if overlapping:
                merged = _merge_static_updates(existing, dataset)
                _rewrite_dataset_via_temp(
                    merged,
                    destination,
                    zarr_format=zarr_format,
                    target_chunk_mb=target_chunk_mb,
                )
            elif missing:
                _write_dataset(
                    dataset,
                    destination,
                    mode="a",
                    zarr_format=zarr_format,
                    target_chunk_mb=target_chunk_mb,
                )
            _sync_dataset_metadata(dataset, destination)
            return

        missing_names = [name for name in dataset.data_vars if name not in existing.data_vars]
        overlap_rewritten = False
        existing = _write_missing_variables(
            existing,
            dataset,
            destination,
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )

        overlap_rewritten = _rewrite_overlapping_times(
            existing,
            dataset,
            destination,
            zarr_format=zarr_format,
        )

        candidate_times = dataset.indexes["time"]
        existing_times = existing.indexes["time"]
        new_times = candidate_times.difference(existing_times)
        if len(new_times) == 0:
            if missing_names or overlap_rewritten:
                _stamp_data_update_attrs(dataset)
            _sync_dataset_metadata(dataset, destination)
            return

        appendable_candidate = dataset.sel(time=new_times.values)
        if _can_append_new_times(existing, appendable_candidate):
            _stamp_data_update_attrs(dataset)
            _append_new_times(
                existing,
                appendable_candidate,
                destination,
                zarr_format=zarr_format,
            )
            _sync_dataset_metadata(dataset, destination)
            return

        _stamp_data_update_attrs(dataset)
        merged = _merge_time_updates(existing, dataset)
        _rewrite_dataset_via_temp(
            merged,
            destination,
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )
        _sync_dataset_metadata(dataset, destination)
    finally:
        existing.close()
        gc.collect()


def _merge_source_stores(
    source_destinations: Iterable[tuple[Path, str]],
    *,
    clean: bool,
    zarr_format: int,
    target_chunk_mb: int,
    interval: tuple[date | None, date | None] | None,
) -> list[str]:
    """Merge source stores into destinations, cleaning each destination once."""

    cleaned_destinations: set[str] = set()
    merged_destinations: list[str] = []
    for source_store, destination in source_destinations:
        match = LEVEL_RE.search(source_store.name)
        level = match.group("level") if match is not None else "unknown"
        LOGGER.info(
            "stage=merge_start level=%s source=%s destination=%s",
            level,
            source_store,
            destination,
        )
        source_dataset = xr.open_zarr(
            str(source_store),
            consolidated=(zarr_format == 2),
        )
        try:
            source_dataset = _select_merge_interval(source_dataset, interval)
            if "time" in source_dataset.dims and source_dataset.sizes["time"] == 0:
                LOGGER.info("Skipping %s: no data in requested interval", source_store)
                continue
            update_zarr_store(
                source_dataset,
                destination,
                clean=(clean and destination not in cleaned_destinations),
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            )
        finally:
            source_dataset.close()

        cleaned_destinations.add(destination)
        merged_destinations.append(destination)
        LOGGER.info(
            "stage=merge_done level=%s destination=%s",
            level,
            destination,
        )

    return sorted(set(merged_destinations))


def _worker_output_roots(
    sources: Iterable[str | Path],
    *,
    dataset: str,
    frequencies: tuple[str, ...] | None,
    variables: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve worker-output roots into matching worker item directories."""

    roots: list[str] = []
    for source in sources:
        path = Path(source)
        if path.name == dataset:
            roots.append(str(path))
            continue
        candidates = sorted(path.glob("*")) if path.name == "worker-output" else [path]
        for candidate in candidates:
            name = candidate.name
            if candidate.is_dir():
                has_nested_dataset = (candidate / dataset).is_dir()
                if (
                    not has_nested_dataset
                    and frequencies
                    and not any(f"-{frequency}-" in name for frequency in frequencies)
                ):
                    continue
                if (
                    not has_nested_dataset
                    and variables
                    and not any(name.endswith(f"-{variable}") for variable in variables)
                ):
                    continue
                roots.append(str(candidate))
    return tuple(dict.fromkeys(roots))


def _frequency_names(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize optional frequency selections to canonical CLI names."""

    if value is None:
        return None
    values = (value,) if isinstance(value, str) else tuple(value)
    selected = tuple(item.strip() for value_item in values for item in str(value_item).split(",") if item.strip())
    if selected == ("all",):
        return tuple(SOURCE_MAPPER["output_frequency"])
    unknown = sorted(set(selected) - set(SOURCE_MAPPER["output_frequency"]))
    if unknown:
        raise ValueError(f"Unsupported merge frequencies: {', '.join(unknown)}")
    return tuple(dict.fromkeys(selected))


def _variable_names(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Normalize optional variable selections."""

    if value is None:
        return None
    values = (value,) if isinstance(value, str) else tuple(value)
    selected = tuple(item.strip() for value_item in values for item in str(value_item).split(",") if item.strip())
    return tuple(dict.fromkeys(selected))


def _dataset_root_destinations(
    source_roots: Iterable[str],
    *,
    dataset: str,
    frequencies: tuple[str, ...] | None,
    levels: tuple[int, ...] | None,
    target_root: str | Path,
) -> Iterable[tuple[Path, str]]:
    """Yield store pairs from worker roots organized by dataset/frequency."""

    output_frequency = SOURCE_MAPPER["output_frequency"]
    selected_output_frequencies = {output_frequency[frequency] for frequency in frequencies} if frequencies else None

    for source_root in source_roots:
        source_path = Path(source_root)
        dataset_root = source_path if source_path.name == dataset else source_path / dataset
        frequency_dirs = sorted(dataset_root.iterdir()) if dataset_root.is_dir() else ()
        for source_frequency in frequency_dirs:
            if not source_frequency.is_dir():
                continue
            if selected_output_frequencies and source_frequency.name not in selected_output_frequencies:
                continue
            frequency = next(
                (name for name, output_name in output_frequency.items() if output_name == source_frequency.name),
                None,
            )
            if frequency is None:
                continue
            target_frequency = Path(target_root) / source_frequency.name
            for source_store in sorted(source_frequency.glob("level_*.zarr")):
                if _is_selected_level_store(source_store, levels):
                    yield source_store, str(target_frequency / source_store.name)


def merge_zarr_stores(
    *,
    sources: Iterable[str | Path],
    target_dir: str | Path,
    clean: bool,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
    dataset: str | None = None,
    frequency: str | Iterable[str] | None = None,
    variable: str | Iterable[str] | None = None,
    levels: tuple[int, ...] | None = None,
    interval: tuple[date | None, date | None] | None = None,
) -> list[str]:
    """Merge direct stores or selected Reflow worker roots into one target.

    When ``dataset`` is provided, ``sources`` are treated as output
    roots and nested dataset/frequency stores are resolved automatically.
    ``frequency``, ``variable``, ``levels``, and ``interval`` optionally filter
    those worker directories and stores. ``levels=None`` selects all available
    levels. The interval is inclusive and applies to time-dependent stores.
    Without ``dataset``, sources must directly contain ``level_*.zarr`` stores.
    """

    selected_frequencies = _frequency_names(frequency)
    selected_variables = _variable_names(variable)
    if dataset is not None:
        target_dir = merge_dataset_root(
            dataset,
            output_path=target_dir,
            frequencies=selected_frequencies,
        )
        worker_roots = _worker_output_roots(
            sources,
            dataset=dataset,
            frequencies=selected_frequencies,
            variables=selected_variables,
        )
        source_destinations = _dataset_root_destinations(
            worker_roots,
            dataset=dataset,
            frequencies=selected_frequencies,
            levels=levels,
            target_root=target_dir,
        )
    else:
        target_path = Path(target_dir)
        source_destinations = (
            (source_store, str(target_path / source_store.name))
            for source in sources
            for source_store in sorted(Path(source).glob("level_*.zarr"))
            if _is_selected_level_store(source_store, levels)
        )

    return _merge_source_stores(
        source_destinations,
        clean=clean,
        zarr_format=zarr_format,
        target_chunk_mb=target_chunk_mb,
        interval=interval,
    )
