#!/usr/bin/env python
"""Transformation helpers for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import grid_doctor as gd

from .common import (
    build_paths,
    drop_surface_coords,
    load_plan,
    maybe_start_local_client,
    open_source_dataset,
    to_time_strings,
)

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
    return ds


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
    return drop_surface_coords(
        normalise_time_axis(rename_values_dim(ds)).rename({"values": "cell"})
    )


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
    pyramid: dict[int, "xr.Dataset"],
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
            target.unlink()
        chunked = chunk_healpix_dataset(
            level_ds, time_chunk=time_chunk, cell_chunk=cell_chunk
        )
        chunked.to_netcdf(target, mode="w")
        level_paths[str(level)] = str(target)
    return level_paths


# Peak working-set budget per dask block, in bytes. The HEALPix kernels
# accumulate in float64, so the bound is computed against an 8-byte
# itemsize. ~1.5 GiB keeps a single finest-level time block well within
# a worker's share of node memory even at level 11.
_BLOCK_BUDGET_BYTES = 1536 * 1024 * 1024


def _safe_time_chunk(
    level: int,
    ntime: int,
    *,
    user_cap: int,
    budget_bytes: int = _BLOCK_BUDGET_BYTES,
    itemsize: int = 8,
) -> int:
    """Largest time chunk that keeps one HEALPix time block within budget.

    The number of cells at a level is ``12 * 4**level``, so the finest
    levels force a small time chunk while coarse levels are capped by the
    user-requested ``time_chunk`` and the total number of time steps.
    """
    npix = 12 * (4 ** int(level))
    budget_chunk = max(1, budget_bytes // (npix * itemsize))
    return max(1, min(int(ntime), int(user_cap), int(budget_chunk)))


def _downcast_floats(ds: "xr.Dataset", dtype: str = "float32") -> "xr.Dataset":
    """Cast floating data variables to *dtype* to halve memory and storage.

    The kernels compute in float64 for accuracy; the result is stored as
    float32, matching the chunking assumptions used when publishing.
    """
    for name in list(ds.data_vars):
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype(dtype)
    return ds


def _write_level(
    level_ds: "xr.Dataset",
    target: Path,
    *,
    time_chunk: int,
    cell_chunk: int,
) -> None:
    """Stream one HEALPix level to a temporary netCDF file.

    The dataset is (re)chunked to the on-disk layout immediately before
    writing; xarray derives the netCDF chunking from the dask chunks, so
    the write proceeds block-by-block with a bounded working set.
    """
    if target.exists():
        target.unlink()
    chunk_map: dict[str, int] = {}
    if "time" in level_ds.dims:
        chunk_map["time"] = min(time_chunk, int(level_ds.sizes["time"]))
    if "cell" in level_ds.dims:
        chunk_map["cell"] = min(cell_chunk, int(level_ds.sizes["cell"]))
    out = level_ds.chunk(chunk_map) if chunk_map else level_ds
    out.to_netcdf(target, mode="w", engine="h5netcdf")


def _build_pyramid_streaming(
    ds: "xr.Dataset",
    *,
    max_level: int | None,
    weights_path: str,
    output_root: Path,
    user_time_chunk: int,
    cell_chunk: int,
) -> dict[str, str]:
    """Regrid once at the finest level, then coarsen level-by-level on disk.

    Unlike building the whole pyramid in memory, this:

    * regrids only to ``max_level`` and streams it straight to disk, and
    * derives every coarser level by reading the previously written
      (finer) level back from disk and coarsening a single step.

    The expensive remap therefore runs exactly once, no level is ever
    fully resident in memory, and coarse levels do not re-trigger the
    finest regrid graph.
    """
    import xarray as xr

    output_root.mkdir(parents=True, exist_ok=True)
    if max_level is None:
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))

    level_paths: dict[str, str] = {}

    # --- Finest level: single streamed regrid. ---
    finest = _downcast_floats(
        gd.regrid_to_healpix(ds, max_level, weights_path=weights_path)
    )
    finest_target = level_output_path(output_root, max_level).with_suffix(".nc")
    _write_level(
        finest,
        finest_target,
        time_chunk=user_time_chunk,
        cell_chunk=cell_chunk,
    )
    level_paths[str(max_level)] = str(finest_target)
    del finest

    # --- Coarser levels: read the finer level from disk, coarsen one step. ---
    prev_target = finest_target
    for level in range(max_level - 1, -1, -1):
        prev = xr.open_dataset(prev_target, engine="h5netcdf", chunks={})
        # The coarsening core dimension is ``cell`` and must be a single
        # chunk; only ``time`` is chunked, sized against the finer level.
        rechunk: dict[str, int] = {"cell": -1}
        if "time" in prev.dims:
            rechunk["time"] = _safe_time_chunk(
                level + 1, int(prev.sizes["time"]), user_cap=user_time_chunk
            )
        prev = prev.chunk(rechunk)
        coarse = _downcast_floats(gd.coarsen_healpix(prev, level))
        target = level_output_path(output_root, level).with_suffix(".nc")
        _write_level(
            coarse,
            target,
            time_chunk=user_time_chunk,
            cell_chunk=cell_chunk,
        )
        prev.close()
        level_paths[str(level)] = str(target)
        prev_target = target

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
    """Convert one raw input file into temporary per-level netCDF files.

    The source is opened lazily (dask) and chunked along ``time`` only,
    so the remap to the finest HEALPix level and every coarsening step
    stream block-by-block. Peak memory is therefore bounded by one time
    block rather than scaling with the full target grid, which is what
    previously caused out-of-memory failures at high HEALPix levels.
    """

    plan = load_plan(run_dir)
    source_path = Path(downloaded["local_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Missing raw input file: {source_path}")

    max_level = (
        None if plan.get("max_level") is None else int(plan.get("max_level"))  # type: ignore[arg-type]
    )

    client = maybe_start_local_client(local_dask_workers)
    try:
        # ``chunks={}`` returns a dask-backed dataset (engine-preferred
        # chunking); the actual time chunking is applied after the dims
        # are normalised to ``(time, cell)``.
        ds = prepare_dataset_for_regridding(
            open_source_dataset(
                source_path,
                engine=plan["source_engine"],
                backend_kwargs=plan["source_backend_kwargs"],
                chunks={},
            )
        )

        time_values = to_time_strings(ds["time"].values) if "time" in ds.dims else []
        if "time" in ds.dims:
            finest_level = (
                max_level
                if max_level is not None
                else gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
            )
            ds = ds.chunk(
                {
                    "time": _safe_time_chunk(
                        finest_level, len(time_values), user_cap=time_chunk
                    )
                }
            )

        level_paths = _build_pyramid_streaming(
            ds,
            max_level=max_level,
            weights_path=plan["weights_path"],
            output_root=worker_output_root(
                build_paths(run_dir), int(downloaded["item_index"])
            ),
            user_time_chunk=time_chunk,
            cell_chunk=cell_chunk,
        )
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
