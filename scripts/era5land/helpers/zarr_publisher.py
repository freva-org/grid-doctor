"""Zarr publication helpers for ERA5/ERA5-Land outputs."""

import gc
import logging
from pathlib import Path
import shutil
import uuid
from typing import Any, Optional

import numpy as np
import xarray as xr
import zarr

from .datasets import normalise_published_dataset
from .metadata import clean_output_attrs

LOGGER = logging.getLogger(__name__)


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

def _encoding_for_full_horizontal_chunks(
    dataset: xr.Dataset,
    *,
    target_mb: int = 100,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return Zarr encoding with full horizontal chunks and bounded time chunks."""

    target_bytes = target_mb * 1024**2
    encoding: dict[str, dict[str, tuple[int, ...]]] = {}

    for name, da in dataset.data_vars.items():
        if "cell" not in da.dims:
            continue

        dtype_size = da.dtype.itemsize
        ncell = dataset.sizes["cell"]
        bytes_per_time = ncell * dtype_size

        if "time" in da.dims:
            resolved_time_chunk = max(1, target_bytes // bytes_per_time)
        else:
            resolved_time_chunk = None

        chunks = []
        for dim in da.dims:
            if dim == "time":
                chunks.append(resolved_time_chunk)
            elif dim == "cell":
                chunks.append(ncell)
            else:
                chunks.append(dataset.sizes[dim])

        encoding[name] = {"chunks": tuple(chunks)}

    return encoding

def _write_dataset(
    dataset: xr.Dataset,
    destination: str,
    *,
    mode: str,
    zarr_format: int,
    append_dim: Optional[str] = None,
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
        # Do not pass encoding when appending along time.
        # xarray rejects encoding for variables that already exist.        
        dataset.to_zarr(destination, **options)
        return

    encoding = _encoding_for_full_horizontal_chunks(dataset)

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
        )
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.move(str(temp_path), str(destination_path))
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def truncate_zarr_store_after(
    destination: str,
    *,
    cutoff: str,
    zarr_format: int,
) -> bool:
    """Remove timestamps strictly after ``cutoff`` from one existing Zarr store.

    Parameters
    ----------
    destination:
        Path to the target Zarr store.
    cutoff:
        Inclusive ISO date string used as the upper bound of the retained time
        selection.
    zarr_format:
        Output Zarr format version used when rewriting the truncated store.

    Returns
    -------
    bool
        ``True`` when the store was rewritten, else ``False``.
    """

    path = Path(destination)
    if not path.exists():
        return False

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))
    truncated: xr.Dataset | None = None
    try:
        if "time" not in existing.dims:
            return False

        original_size = existing.sizes.get("time", 0)
        truncated = existing.sel(time=slice(None, cutoff))
        truncated_size = truncated.sizes.get("time", 0)
        if truncated_size == original_size:
            return False

        LOGGER.info(
            "Truncating existing Zarr store %s after %s (time: %s -> %s)",
            destination,
            cutoff,
            original_size,
            truncated_size,
        )
        _rewrite_dataset_via_temp(
            truncated,
            destination,
            zarr_format=zarr_format,
        )
        return True
    finally:
        if truncated is not None:
            truncated.close()
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
) -> xr.Dataset:
    """Add variables missing from an existing store across the current time axis."""

    missing = [name for name in candidate.data_vars if name not in existing.data_vars]
    if not missing:
        return existing

    add_ds = candidate[missing].reindex(time=existing["time"].values)
    _write_dataset(
        add_ds,
        destination,
        mode="a",
        zarr_format=zarr_format,
    )
    updated = existing.copy()
    for name in missing:
        updated[name] = add_ds[name]
    return updated


def _merge_static_updates(existing: xr.Dataset, candidate: xr.Dataset) -> xr.Dataset:
    """Merge non-time variables, preferring values from *candidate*."""

    return candidate.combine_first(existing)


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
        region = {"time": time_slice}
        region_ds.to_zarr(
            destination,
            mode="r+",
            region=region,
            zarr_format=zarr_format,
            consolidated=(zarr_format == 2),
        )


def update_zarr_store(
    dataset: xr.Dataset,
    destination: str,
    *,
    clean: bool,
    zarr_format: int,
    truncate_after: str | None = None,
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
        )
        _sync_global_attrs(dict(dataset.attrs), destination)
        _sync_variable_attrs(dataset, destination)
        return

    if truncate_after is not None:
        truncate_zarr_store_after(
            destination,
            cutoff=truncate_after,
            zarr_format=zarr_format,
        )

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))
    try:
        if "time" not in dataset.dims or "time" not in existing.dims:
            missing = [name for name in dataset.data_vars if name not in existing.data_vars]
            overlapping = [name for name in dataset.data_vars if name in existing.data_vars]
            if overlapping:
                merged = _merge_static_updates(existing, dataset)
                _rewrite_dataset_via_temp(
                    merged,
                    destination,
                    zarr_format=zarr_format,
                )
            elif missing:
                _write_dataset(
                    dataset,
                    destination,
                    mode="a",
                    zarr_format=zarr_format,
                )
            _sync_global_attrs(dict(dataset.attrs), destination)
            _sync_variable_attrs(dataset, destination)
            return

        existing = _write_missing_variables(
            existing,
            dataset,
            destination,
            zarr_format=zarr_format,
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
            _sync_global_attrs(dict(dataset.attrs), destination)
            _sync_variable_attrs(dataset, destination)
            return

        appendable_candidate = dataset.sel(time=new_times.values)
        if _can_append_new_times(existing, appendable_candidate):
            _append_new_times(
                existing,
                appendable_candidate,
                destination,
                zarr_format=zarr_format,
            )
            _sync_global_attrs(dict(dataset.attrs), destination)
            _sync_variable_attrs(dataset, destination)
            return

        merged = _merge_time_updates(existing, dataset)
        _rewrite_dataset_via_temp(
            merged,
            destination,
            zarr_format=zarr_format,
        )
        _sync_global_attrs(dict(dataset.attrs), destination)
        _sync_variable_attrs(dataset, destination)
    finally:
        existing.close()
        gc.collect()
