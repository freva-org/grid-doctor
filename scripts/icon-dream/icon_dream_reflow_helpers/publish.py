#!/usr/bin/env python
"""Publishing helpers for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import grid_doctor as gd

from .common import (chunk_for_target_store_size, drop_surface_coords,
                     load_plan, open_existing_target, s3_map)

if TYPE_CHECKING:
    import xarray as xr

BAD = ["heightAboveGround", "surface"]


def fill_value_for_dtype(dtype: np.dtype[Any]) -> Any:
    """Return a sensible fill value for a dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    return None


def make_fill_variable(template: "xr.DataArray", name: str) -> "xr.DataArray":
    """Create a fully missing variable matching a template array."""
    import xarray as xr

    fill = fill_value_for_dtype(template.dtype)
    values = np.full(template.shape, fill, dtype=template.dtype)
    arr = xr.DataArray(
        values, dims=template.dims, coords=template.coords, attrs=template.attrs
    )
    arr.name = name
    return arr


def pad_missing_existing_vars_for_append(
    candidate: "xr.Dataset", existing: "xr.Dataset"
) -> "xr.Dataset":
    """Pad candidate data with placeholder variables required for appending."""
    for name, data in existing.data_vars.items():
        if name not in candidate.data_vars:
            candidate[name] = make_fill_variable(
                data.isel(time=slice(0, candidate.sizes.get("time", 0))), name
            )
    ordered = [name for name in existing.data_vars if name in candidate.data_vars]
    ordered.extend(name for name in candidate.data_vars if name not in ordered)
    return candidate[ordered]


def write_new_variables_full_axis(
    existing: "xr.Dataset",
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str],
    *,
    compression_level: int,
    access_pattern: str,
    strict_access_pattern: bool,
    zarr_format: Literal[2, 3],
) -> int:
    """Write variables that are missing from an existing target across the full current axis."""
    missing = [name for name in candidate.data_vars if name not in existing.data_vars]
    if not missing:
        return 0
    add_ds = candidate[missing].reindex(time=existing["time"].values)
    add_ds.to_zarr(
        s3_map(target_path, s3_options),
        mode="a",
        consolidated=True,
        zarr_format=zarr_format,
        align_chunks=True,
    )
    return len(missing)


def append_time_block(
    existing: "xr.Dataset",
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str],
    *,
    compression_level: int,
    access_pattern: str,
    strict_access_pattern: bool,
    zarr_format: Literal[2, 3],
) -> int:
    """Append strictly newer time slices to the target."""
    if "time" not in candidate.dims:
        return 0
    existing_times = set(map(str, existing["time"].values))
    new_time_values = [
        value for value in candidate["time"].values if str(value) not in existing_times
    ]
    if not new_time_values:
        return 0
    append_ds = pad_missing_existing_vars_for_append(
        candidate.sel(time=new_time_values), existing
    )
    append_ds.to_zarr(
        s3_map(target_path, s3_options),
        mode="a",
        append_dim="time",
        consolidated=True,
        zarr_format=zarr_format,
        align_chunks=True,
    )
    return int(append_ds.sizes.get("time", 0))


def write_static_dataset(
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str],
    *,
    compression_level: int,
    access_pattern: str,
    strict_access_pattern: bool,
    zarr_format: Literal[2, 3],
) -> None:
    """Write a static dataset from scratch."""
    candidate.to_zarr(
        s3_map(target_path, s3_options),
        mode="w",
        consolidated=True,
        zarr_format=zarr_format,
        align_chunks=True,
    )


def merge_level_dataset(
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str],
    *,
    overwrite_static: bool,
    replace_existing_times: bool,
    compression_level: int,
    access_pattern: str,
    strict_access_pattern: bool,
    zarr_format: Literal[2, 3],
) -> dict[str, Any]:
    """Merge one per-level candidate dataset into the final target."""
    summary = {
        "target_path": target_path,
        "time_slices_written": 0,
        "missing_variables_added": 0,
        "static_writes": 0,
        "overlapping_times_rewritten": 0,
    }
    existing = open_existing_target(target_path, s3_options)
    if existing is None:
        write_static_dataset(
            candidate,
            target_path,
            s3_options,
            compression_level=compression_level,
            access_pattern=access_pattern,
            strict_access_pattern=strict_access_pattern,
            zarr_format=zarr_format,
        )
        summary["static_writes"] = 1
        summary["time_slices_written"] = int(candidate.sizes.get("time", 0))
        return summary

    if "time" not in candidate.dims:
        if overwrite_static:
            write_static_dataset(
                candidate,
                target_path,
                s3_options,
                compression_level=compression_level,
                access_pattern=access_pattern,
                strict_access_pattern=strict_access_pattern,
                zarr_format=zarr_format,
            )
            summary["static_writes"] = 1
        return summary

    summary["missing_variables_added"] = write_new_variables_full_axis(
        existing,
        candidate,
        target_path,
        s3_options,
        compression_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
        zarr_format=zarr_format,
    )
    summary["time_slices_written"] = append_time_block(
        existing,
        candidate,
        target_path,
        s3_options,
        compression_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
        zarr_format=zarr_format,
    )

    if replace_existing_times:
        candidate_times = list(map(str, candidate["time"].values))
        existing_time_set = set(map(str, existing["time"].values))
        overlap_times = [time for time in candidate_times if time in existing_time_set]
        if overlap_times:
            rewrite_ds = candidate.sel(time=overlap_times).reindex(
                time=existing["time"].values
            )
            rewrite_ds.to_zarr(
                s3_map(target_path, s3_options),
                mode="a",
                consolidated=True,
                zarr_format=zarr_format,
                encoding=dataset_encoding(
                    rewrite_ds,
                    compression_level=compression_level,
                    access_pattern=access_pattern,
                    strict_access_pattern=strict_access_pattern,
                ),
                align_chunks=True,
            )
            summary["overlapping_times_rewritten"] = len(overlap_times)

    return summary


def combine_worker_level_outputs(level_paths: list[str]) -> "xr.Dataset":
    """Open and combine all temporary outputs for one HEALPix level."""
    import xarray as xr

    datasets = [xr.open_dataset(path) for path in level_paths]
    return (
        datasets[0]
        if len(datasets) == 1
        else xr.combine_by_coords(datasets, combine_attrs="drop_conflicts")
    )


def finalize_outputs(
    worker_results: dict[str, Any],
    *,
    s3_endpoint: str,
    s3_credentials_file: str,
    overwrite_static: bool,
    replace_existing_times: bool,
    compression_level: int,
    access_pattern: Literal["map", "time_series"],
    strict_access_pattern: bool,
    zarr_format: Literal[2, 3],
    run_dir: str,
) -> dict[str, Any]:
    """Merge all temporary outputs and publish them to the final S3 target."""
    import xarray as xr

    plan = load_plan(run_dir)
    s3_options = gd.get_s3_options(s3_endpoint, s3_credentials_file)

    level = worker_results["level"]
    chunks = chunk_for_target_store_size(level=level)
    candidate = xr.open_mfdataset(
        worker_results["level_paths"],
        preprocess=drop_surface_coords,
        combine="nested",
        concat_dim="time",
        chunks=chunks,
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",  # only if non-time dims really match
        combine_attrs="override",
    ).sortby("time")
    candidate = candidate.drop_duplicates(dim="time", keep="first")
    return merge_level_dataset(
        candidate.chunk(chunks),
        f"{plan['target_root']}/level_{level}.zarr",
        s3_options,
        overwrite_static=overwrite_static,
        replace_existing_times=replace_existing_times,
        compression_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
        zarr_format=zarr_format,
    )
