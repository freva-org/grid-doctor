"""Zarr publication helpers for ERA5/ERA5-Land outputs."""

import gc
import logging
from pathlib import Path
import re
import shutil
import uuid
from typing import Any, Iterable, Optional, Union

import numpy as np
import xarray as xr
import zarr

from .datasets import normalise_published_dataset
from .formatter import destination_for_level, existing_destinations_for_frequency
from .metadata import clean_output_attrs

LOGGER = logging.getLogger(__name__)
DEFAULT_TARGET_CHUNK_MB = 100
LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")


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

    resolved_time_chunk = None
    if "time" in da.dims:
        resolved_time_chunk = max(1, target_bytes // (cell_size * non_time_extent))
        resolved_time_chunk = int(resolved_time_chunk)

    chunks: list[int] = []
    for dim in da.dims:
        if dim == "time":
            chunks.append(resolved_time_chunk)
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
    for name, data in existing.data_vars.items():
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

    for name, da in dataset.data_vars.items():
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
    append_dim: Optional[str] = None,
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
    chunk_map: dict[str, tuple[int, ...]] = {}
    for name, enc in encoding.items():
        if name in dataset:
            dims = dataset[name].dims
            chunks = enc.get("chunks")
            if chunks is not None:
                chunk_map.update(dict(zip(dims, chunks)))

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
    temp_path = destination_path.parent / (
        f".{destination_path.name}.tmp-{uuid.uuid4().hex}"
    )

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

    for name, data in dataset.data_vars.items():
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
    return merged


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

    missing = [name for name in candidate.data_vars if name not in existing.data_vars]
    if not missing:
        return existing

    root = zarr.open_group(destination, mode="a")
    encoding = _encoding_for_target_chunks(
        candidate[missing],
        target_mb=target_chunk_mb,
    )

    for name in missing:
        data = candidate[name]
        shape = tuple(
            int(existing.sizes.get(dim, data.sizes[dim]))
            for dim in data.dims
        )
        array = root.create_dataset(
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

    return candidate.combine_first(existing)


def _align_to_existing_chunks(
    dataset: xr.Dataset,
    existing: xr.Dataset,
) -> xr.Dataset:
    """Rechunk a dataset to the Zarr chunks already used by an existing store."""

    aligned = dataset.copy(deep=False)
    chunk_map: dict[str, int] = {}
    for name, data in aligned.data_vars.items():
        if name not in existing:
            continue
        chunks = existing[name].encoding.get("chunks")
        if chunks is None:
            continue
        chunk_map.update(dict(zip(data.dims, (int(size) for size in chunks))))
        aligned[name].encoding = dict(aligned[name].encoding)
        aligned[name].encoding["chunks"] = tuple(int(size) for size in chunks)

    return aligned.chunk(chunk_map) if chunk_map else aligned


def _append_new_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> None:
    """Append strictly newer time slices to an existing store."""

    existing_time_strings = set(map(str, existing["time"].values))
    new_time_values = [
        value for value in candidate["time"].values if str(value) not in existing_time_strings
    ]
    if not new_time_values:
        return

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


def _rewrite_overlapping_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> None:
    """Rewrite only the overlapping time regions in an existing store."""

    overlap = candidate.indexes["time"].intersection(existing.indexes["time"])
    if len(overlap) == 0:
        return

    time_positions = existing.indexes["time"].get_indexer(overlap)
    time_slices = _contiguous_slices(np.asarray(time_positions, dtype=np.int64))
    for time_slice in time_slices:
        region_times = existing["time"].values[time_slice]
        region_ds = normalise_published_dataset(candidate.sel(time=region_times))
        to_drop = {
            name
            for name, var in region_ds.variables.items()
            if "time" not in var.dims
        }
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
            merged = dataset.combine_first(existing)
            if "time" in merged.coords:
                merged = merged.sortby("time")
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

        existing = _write_missing_variables(
            existing,
            dataset,
            destination,
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )

        _rewrite_overlapping_times(
            existing,
            dataset,
            destination,
            zarr_format=zarr_format,
        )

        candidate_times = dataset.indexes["time"]
        existing_times = existing.indexes["time"]
        new_times = candidate_times.difference(existing_times)
        if len(new_times) == 0:
            _sync_dataset_metadata(dataset, destination)
            return

        appendable_candidate = dataset.sel(time=new_times.values)
        if _can_append_new_times(existing, appendable_candidate):
            _append_new_times(
                existing,
                appendable_candidate,
                destination,
                zarr_format=zarr_format,
            )
            _sync_dataset_metadata(dataset, destination)
            return

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
) -> list[str]:
    """Merge source stores into destinations, cleaning each destination once."""

    cleaned_destinations: set[str] = set()
    merged_destinations: list[str] = []
    for source_store, destination in source_destinations:
        source_dataset = xr.open_zarr(
            str(source_store),
            consolidated=(zarr_format == 2),
        )
        try:
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

    return sorted(set(merged_destinations))


def merge_zarr_store_roots(
    *,
    dataset: str,
    frequency: str,
    source_roots: Iterable[Union[str, Path]],
    target_root: Optional[Union[str, Path]],
    clean: bool,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> list[str]:
    """Merge one or more source output roots into a target publication root.

    Parameters
    ----------
    dataset:
        Dataset identifier such as ``"era5land"`` or ``"era5"``.
    frequency:
        Output frequency whose stores should be merged.
    source_roots:
        Directories that contain dataset/frequency/``level_*.zarr`` stores.
    target_root:
        Destination publication root. When ``None``, the configured default
        publication root is used.
    clean:
        Whether to recreate each touched destination store on its first merge.
    zarr_format:
        Output Zarr format version used to read source stores and write the
        merged destination stores.
    target_chunk_mb:
        Approximate target chunk size in megabytes for rewritten destination
        stores.

    Returns
    -------
    list[str]
        Sorted unique destination store paths that received merged data.
    """

    source_destinations = (
        (
            Path(store_name),
            destination_for_level(
                dataset,
                frequency,
                int(LEVEL_RE.search(Path(store_name).name).group("level")),
                output_path=target_root,
            ),
        )
        for source_root in source_roots
        for store_name in existing_destinations_for_frequency(
            dataset,
            frequency,
            output_path=source_root,
        )
        if LEVEL_RE.search(Path(store_name).name) is not None
    )
    return _merge_source_stores(
        source_destinations,
        clean=clean,
        zarr_format=zarr_format,
        target_chunk_mb=target_chunk_mb,
    )


def merge_zarr_store_directories(
    *,
    source_dirs: Iterable[Union[str, Path]],
    target_dir: Union[str, Path],
    clean: bool,
    zarr_format: int,
    target_chunk_mb: int = DEFAULT_TARGET_CHUNK_MB,
) -> list[str]:
    """Merge one or more directories of ``level_*.zarr`` stores into a target.

    Parameters
    ----------
    source_dirs:
        Directories that directly contain ``level_*.zarr`` stores.
    target_dir:
        Destination directory that should receive merged ``level_*.zarr``
        stores.
    clean:
        Whether to recreate each touched destination store on its first merge.
    zarr_format:
        Output Zarr format version used to read source stores and write merged
        destination stores.
    target_chunk_mb:
        Approximate target chunk size in megabytes for rewritten destination
        stores.

    Returns
    -------
    list[str]
        Sorted unique destination store paths that received merged data.
    """

    target_path = Path(target_dir)
    source_destinations = (
        (source_store, str(target_path / source_store.name))
        for source_dir in source_dirs
        for source_store in sorted(Path(source_dir).glob("level_*.zarr"))
        if LEVEL_RE.search(source_store.name) is not None
    )
    return _merge_source_stores(
        source_destinations,
        clean=clean,
        zarr_format=zarr_format,
        target_chunk_mb=target_chunk_mb,
    )
