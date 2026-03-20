#!/usr/bin/env python
"""Compact Reflow DAG for ICON-DREAM -> HEALPix Zarr conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Annotated, Literal

from reflow import Param, Result, RunDir, Workflow

from icon_dream_reflow_helpers import (
    DEFAULT_SOURCE_ROOT,
    TIME_FREQUENCY,
    build_plan,
    convert_downloaded_item,
    default_run_dir,
    download_source_item,
    finalize_outputs,
    prepare_shared_assets,
)

wf = Workflow("icon_dream_healpix")


@wf.job(cpus=2, time="00:20:00", mem="0", partition="compute")
def gather_sources(
    s3_bucket: Annotated[str, Param(help="Target S3 bucket")],
    start: Annotated[str, Param(help="Requested UTC start time")],
    end: Annotated[str, Param(help="Requested UTC end time")],
    variables: Annotated[list[str], Param(help="Variables to process")] = ["t_2m", "tot_prec"],
    freq: Annotated[TIME_FREQUENCY, Param(help="ICON-DREAM data frequency")] = "hourly",
    source_root: Annotated[str, Param(help="Source dataset root URL")] = DEFAULT_SOURCE_ROOT,
    s3_endpoint: Annotated[str, Param(help="S3 endpoint URL")] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[str, Param(help="Path to S3 credentials JSON")] = str(Path.home() / ".s3-credentials.json"),
    source_engine: Annotated[str, Param(help="Xarray backend engine for source files")] = "cfgrib",
    source_backend_kwargs_json: Annotated[str, Param(help="JSON backend_kwargs for xarray")] = "{}",
    update_only: Annotated[bool, Param(help="Skip source chunks already covered by existing variables")] = True,
    run_dir: RunDir = RunDir(),
) -> list[dict[str, Any]]:
    """Discover source files that still need processing and persist the run plan."""
    return build_plan(
        s3_bucket=s3_bucket,
        start=start,
        end=end,
        variables=variables,
        freq=freq,
        source_root=source_root,
        s3_endpoint=s3_endpoint,
        s3_credentials_file=s3_credentials_file,
        source_engine=source_engine,
        source_backend_kwargs_json=source_backend_kwargs_json,
        update_only=update_only,
        run_dir=run_dir,
    )["source_items"]


@wf.job(cpus=16, time="08:00:00", mem="0", partition="compute", after=["gather_sources"])
def prepare_shared(
    max_level: Annotated[int | None, Param(help="Override the automatically chosen HEALPix level")] = None,
    download_timeout: Annotated[int, Param(help="HTTP timeout in seconds")] = 60,
    download_chunk_size: Annotated[int, Param(help="HTTP stream chunk size in bytes")] = 1024 * 1024,
    overwrite_downloads: Annotated[bool, Param(help="Re-download the grid file even if it exists")] = False,
    run_dir: RunDir = RunDir(),
) -> str:
    """Download the grid once and cache shared HEALPix weights."""
    return prepare_shared_assets(
        max_level=max_level,
        download_timeout=download_timeout,
        download_chunk_size=download_chunk_size,
        overwrite_downloads=overwrite_downloads,
        run_dir=run_dir,
    )


@wf.array_job(cpus=1, time="01:00:00", mem="0", partition="compute", array_parallelism=16)
def download_source(
    source_item: Annotated[dict[str, Any], Result(step="gather_sources")],
    download_timeout: Annotated[int, Param(help="HTTP timeout in seconds")] = 60,
    download_chunk_size: Annotated[int, Param(help="HTTP stream chunk size in bytes")] = 1024 * 1024,
    overwrite_downloads: Annotated[bool, Param(help="Re-download raw files even if they already exist")] = False,
    run_dir: RunDir = RunDir(),
) -> dict[str, Any]:
    """Download one raw source file, retaining the local skip logic."""
    return download_source_item(
        source_item,
        download_timeout=download_timeout,
        download_chunk_size=download_chunk_size,
        overwrite_downloads=overwrite_downloads,
        run_dir=run_dir,
    )


@wf.array_job(
    cpus=32,
    time="08:00:00",
    mem="0",
    partition="compute",
    array_parallelism=16,
    after=["prepare_shared"],
)
def convert_source(
    downloaded: Annotated[dict[str, Any], Result(step="download_source")],
    time_chunk: Annotated[int, Param(help="Time chunk size for temporary Zarr stores")] = 168,
    cell_chunk: Annotated[int, Param(help="Cell chunk size for temporary Zarr stores")] = 262144,
    zarr_format: Annotated[Literal[2, 3], Param(help="Target Zarr format version")] = 2,
    local_dask_workers: Annotated[int, Param(help="Optional local distributed workers inside one process")] = 0,
    run_dir: RunDir = RunDir(),
) -> dict[str, Any]:
    """Convert one raw file into temporary per-level HEALPix Zarr stores."""
    return convert_downloaded_item(
        downloaded,
        time_chunk=time_chunk,
        cell_chunk=cell_chunk,
        zarr_format=zarr_format,
        local_dask_workers=local_dask_workers,
        run_dir=run_dir,
    )


@wf.job(cpus=8, time="08:00:00", mem="0", partition="compute")
def finalize(
    worker_results: Annotated[list[dict[str, Any]], Result(step="convert_source")],
    s3_endpoint: Annotated[str, Param(help="S3 endpoint URL")] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[str, Param(help="Path to S3 credentials JSON")] = str(Path.home() / ".s3-credentials.json"),
    overwrite_static: Annotated[bool, Param(help="Overwrite an existing static target store")] = False,
    replace_existing_times: Annotated[bool, Param(help="Rewrite overlapping time slices for already-present variables")] = False,
    compression_level: Annotated[int, Param(help="Compression level for final Zarr encoding")] = 4,
    access_pattern: Annotated[Literal["map", "time_series"], Param(help="Chunking optimisation pattern")] = "map",
    strict_access_pattern: Annotated[bool, Param(help="Enforce strict chunking for the chosen access pattern")] = True,
    zarr_format: Annotated[Literal[2, 3], Param(help="Final Zarr format version")] = 2,
    run_dir: RunDir = RunDir(),
) -> dict[str, Any]:
    """Merge temporary outputs and publish the final pyramid to S3."""
    return finalize_outputs(
        worker_results,
        s3_endpoint=s3_endpoint,
        s3_credentials_file=s3_credentials_file,
        overwrite_static=overwrite_static,
        replace_existing_times=replace_existing_times,
        compression_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
        zarr_format=zarr_format,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(wf.cli())
