#!/usr/bin/env python
"""Transformation helpers for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

import grid_doctor as gd

from .common import (build_paths, load_plan, maybe_start_local_client,
                     open_source_dataset, to_time_strings)

if TYPE_CHECKING:
    import xarray as xr


def rename_values_dim(ds: "xr.Dataset") -> "xr.Dataset":
    """Rename the synthetic 'values' dimension to the variable name when possible."""
    if list(ds.data_vars) == ["unknown"] and "values" in ds.dims:
        return ds.rename({"values": "unknown_value"})
    return ds


def flatten_forecast_time(ds: "xr.Dataset") -> "xr.Dataset":
    """Drop a forecast lead dimension if present."""
    if (
        "step" not in ds.coords
        or "time" not in ds.coords
        or "valid_time" not in ds.coords
    ):
        return ds
    if ds["valid_time"].ndim != 2:
        return ds
    if ds.sizes.get("step", 1) == 1:
        ds = ds.isel(step=0, drop=True)
    return ds.load()


def normalise_time_axis(ds: "xr.Dataset") -> "xr.Dataset":
    """Normalise time coordinates to a single monotonic time axis."""
    ds = flatten_forecast_time(ds)
    if "time" in ds.coords and "time" not in ds.dims:
        ds = ds.expand_dims("time")
    if "valid_time" in ds.coords and "time" in ds.dims:
        valid_time_values = np.asarray(ds["valid_time"].values).ravel()
        stacked = ds.stack(_stacked_time=("time", "step"))
        stacked = stacked.drop_vars(["valid_time", "time", "step"], errors="ignore")
        stacked = stacked.assign_coords(_stacked_time=valid_time_values)
        ds = stacked.rename({"_stacked_time": "time"})
    if "time" in ds.coords:
        ds = ds.sortby("time")
    return ds


def prepare_dataset_for_regridding(ds: "xr.Dataset") -> "xr.Dataset":
    """Apply the minimal normalisation needed for regridding."""
    return normalise_time_axis(rename_values_dim(ds)).rename({"values": "cell"})


def chunk_healpix_dataset(
    ds: "xr.Dataset", *, time_chunk: int, cell_chunk: int
) -> "xr.Dataset":
    """Apply the temporary chunk layout used for intermediate worker outputs."""
    chunk_map: dict[str, int] = {}
    if "time" in ds.dims:
        chunk_map["time"] = time_chunk
    for dim in ("cell", "cells"):
        if dim in ds.dims:
            chunk_map[dim] = cell_chunk
    return ds.chunk(chunk_map) if chunk_map else ds


def worker_output_root(paths: dict[str, Path], item_index: int) -> Path:
    """Return the temporary worker output directory."""
    return paths["temp_root"] / f"item-{item_index:05d}"


def level_output_path(output_root: Path, level: int) -> Path:
    """Return the temporary Zarr path for one HEALPix level."""
    return output_root / f"level_{level}.zarr"


def write_temp_pyramid(
    pyramid: Sequence["xr.Dataset"],
    *,
    output_root: Path,
    time_chunk: int,
    cell_chunk: int,
) -> dict[str, str]:
    """Write the temporary HEALPix pyramid produced by one worker."""
    output_root.mkdir(parents=True, exist_ok=True)
    level_paths: dict[str, str] = {}
    for level, level_ds in pyramid.items():
        target = level_output_path(output_root, level).with_suffix(".nc")
        if target.exists():
            shutil.rmtree(target)
        chunked = chunk_healpix_dataset(
            level_ds, time_chunk=time_chunk, cell_chunk=cell_chunk
        )
        chunked.to_netcdf(target, mode="w")
        level_paths[str(level)] = str(target)
    return level_paths


def convert_downloaded_item(
    downloaded: dict[str, Any],
    *,
    time_chunk: int,
    cell_chunk: int,
    zarr_format: Literal[2, 3],
    local_dask_workers: int,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Convert one raw input file into temporary per-level netCDF file."""
    import xarray as xr

    plan = load_plan(run_dir)
    source_path = Path(downloaded["local_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Missing raw input file: {source_path}")

    client = maybe_start_local_client(local_dask_workers)
    try:
        ds = prepare_dataset_for_regridding(
            open_source_dataset(
                source_path,
                engine=plan["source_engine"],
                backend_kwargs=plan["source_backend_kwargs"],
            )
        )
        weights = xr.open_dataset(plan["weights_path"])
        max_level = (
            None if plan.get("max_level") is None else int(plan.get("max_level"))
        )
        pyramid = gd.latlon_to_healpix_pyramid(ds, max_level=max_level, weights=weights)
        level_paths = write_temp_pyramid(
            pyramid,
            output_root=worker_output_root(
                build_paths(run_dir), int(downloaded["item_index"])
            ),
            time_chunk=time_chunk,
            cell_chunk=cell_chunk,
        )
        time_values = to_time_strings(ds["time"].values) if "time" in ds.dims else []
        return {
            "item_index": int(downloaded["item_index"]),
            "variable": downloaded["variable"],
            "level_paths": level_paths,
            "time_count": len(time_values),
            "time_start": time_values[0] if time_values else None,
            "time_end": time_values[-1] if time_values else None,
            "has_time": bool(time_values),
        }
    finally:
        if client is not None:
            client.close()
