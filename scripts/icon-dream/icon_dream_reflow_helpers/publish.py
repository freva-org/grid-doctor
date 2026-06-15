#!/usr/bin/env python
"""Publishing helpers for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import grid_doctor as gd

from .common import (
    chunk_for_target_store_size,
    drop_surface_coords,
    load_plan,
    open_existing_target,
    s3_map,
)

if TYPE_CHECKING:
    import xarray as xr

BAD = ["heightAboveGround", "surface"]


def _build_compressor(compression_level: int, zarr_format: Literal[2, 3]) -> Any:
    """Return a zstd compressor for the requested Zarr format.

    ``compression_level <= 0`` disables explicit compression (the store
    default is used). The codec object differs between Zarr v2
    (``numcodecs``) and v3 (``zarr.codecs``).
    """
    if compression_level <= 0:
        return None
    if zarr_format == 2:
        from numcodecs import Blosc

        return Blosc(
            cname="zstd", clevel=int(compression_level), shuffle=Blosc.SHUFFLE
        )
    from zarr.codecs import BloscCodec, BloscShuffle

    return BloscCodec(
        cname="zstd", clevel=int(compression_level), shuffle=BloscShuffle.shuffle
    )


def _store_encoding(
    ds: "xr.Dataset",
    chunks: dict[str, int],
    *,
    compression_level: int,
    zarr_format: Literal[2, 3],
    names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-variable Zarr encoding (chunk shape + compressor).

    Only variables in *names* are encoded when given; otherwise every
    non-scalar variable and coordinate is encoded. Encoding may only be
    supplied for variables being created (a fresh ``mode="w"`` write or
    brand-new variables on ``mode="a"``); never for variables that
    already exist in the target store.
    """
    compressor = _build_compressor(compression_level, zarr_format)
    selected = ds.variables if names is None else {n: ds[n] for n in names}

    encoding: dict[str, dict[str, Any]] = {}
    for name, var in selected.items():
        if var.ndim == 0:  # scalar (e.g. crs) needs no chunking/compression
            continue
        var_chunks = tuple(
            int(min(chunks.get(str(dim), ds.sizes[dim]), ds.sizes[dim]))
            for dim in var.dims
        )
        var_encoding: dict[str, Any] = {"chunks": var_chunks}
        if compressor is not None:
            if zarr_format == 2:
                var_encoding["compressor"] = compressor
            else:
                var_encoding["compressors"] = (compressor,)
        encoding[str(name)] = var_encoding
    return encoding


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
    s3_options: dict[str, str] | None,
    *,
    chunks: dict[str, int],
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
        encoding=_store_encoding(
            add_ds,
            chunks,
            compression_level=compression_level,
            zarr_format=zarr_format,
            names=missing,
        ),
    )
    return len(missing)


def append_time_block(
    existing: "xr.Dataset",
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str] | None,
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
    s3_options: dict[str, str] | None,
    *,
    chunks: dict[str, int],
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
        encoding=_store_encoding(
            candidate,
            chunks,
            compression_level=compression_level,
            zarr_format=zarr_format,
        ),
    )


def merge_level_dataset(
    candidate: "xr.Dataset",
    target_path: str,
    s3_options: dict[str, str] | None,
    *,
    chunks: dict[str, int],
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
            chunks=chunks,
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
                chunks=chunks,
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
        chunks=chunks,
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
                align_chunks=True,
            )
            summary["overlapping_times_rewritten"] = len(overlap_times)

    return summary


def combine_worker_level_outputs(
    level_paths: list[str],
    *,
    chunks: dict[str, int] | None = None,
) -> "xr.Dataset":
    """Open and combine all temporary outputs for one HEALPix level.

    Each temporary file holds a single variable for one source period.
    Files are therefore grouped by variable, concatenated along ``time``
    within a variable (the periods are disjoint), and only then merged
    across variables by aligning on the shared time axis.

    This avoids a data-loss pattern where concatenating *different*
    variables along ``time`` produces duplicate timestamps (each with the
    other variable NaN-filled) that a subsequent
    ``drop_duplicates("time")`` would collapse, silently discarding one
    variable per overlapping timestamp.
    """
    import xarray as xr
    from collections import defaultdict

    per_variable: dict[str, list[xr.Dataset]] = defaultdict(list)
    for path in sorted(level_paths):
        ds = drop_surface_coords(
            xr.open_dataset(path, engine="h5netcdf", chunks=chunks)
        )
        names = [str(name) for name in ds.data_vars]
        if len(names) == 1:
            per_variable[names[0]].append(ds)
        else:
            for name in names:
                per_variable[name].append(ds[[name]])

    combined_per_variable: list[xr.Dataset] = []
    for name, datasets in per_variable.items():
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = xr.concat(
                datasets,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )
        if "time" in combined.dims:
            combined = combined.sortby("time").drop_duplicates(
                dim="time", keep="first"
            )
        combined_per_variable.append(combined)

    if len(combined_per_variable) == 1:
        return combined_per_variable[0]
    return cast(
        "xr.Dataset",
        xr.merge(
            combined_per_variable,
            compat="override",
            join="outer",
            combine_attrs="override",
        ),
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
    fs_type: str,
    target_chunk_mib: float = 16.0,
    run_dir: str,
) -> dict[str, Any]:
    """Merge all temporary outputs and publish them to the final S3 target."""
    if fs_type.lower() == "s3":
        s3_options = gd.get_s3_options(s3_endpoint, s3_credentials_file)
    elif fs_type.lower() in ["posix", "file"]:
        s3_options = None
    else:
        raise ValueError(f"No such file system type: {fs_type}")
    plan = load_plan(run_dir)

    level = worker_results["level"]
    print(f"Working on {worker_results['level_paths']}")
    # A coarse map-layout chunking is fine for the lazy read of the temp files.
    candidate = combine_worker_level_outputs(
        sorted(worker_results["level_paths"]),
        chunks=chunk_for_target_store_size(level=level),
    )

    # Store chunking targets ~target_chunk_mib on disk and follows the chosen
    # access pattern; time_series needs the time length, so fall back to map
    # for static (timeless) levels.
    ntime = int(candidate.sizes.get("time", 0)) or None
    access = access_pattern if ntime else "map"
    store_chunks = chunk_for_target_store_size(
        level=level,
        access=access,
        ntime=ntime,
        target_stored_mib=target_chunk_mib,
    )

    return merge_level_dataset(
        candidate.chunk(store_chunks),
        f"{plan['target_root']}/level_{level}.zarr",
        s3_options,
        chunks=store_chunks,
        overwrite_static=overwrite_static,
        replace_existing_times=replace_existing_times,
        compression_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
        zarr_format=zarr_format,
    )
