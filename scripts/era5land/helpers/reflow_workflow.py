#!/usr/bin/env python3
"""Reflow workflow for ERA5/ERA5-Land HEALPix publication.

This workflow complements ``converter.py`` by turning one requested remap run
into scheduler-managed batch jobs. Each worker handles one independent
``variable x frequency`` unit, writes its result into a private temporary output
root below the shared Reflow run directory, and the final step merges those
temporary Zarr stores into the configured publication root.

The design keeps the existing remapping logic in ``helpers.mapper`` unchanged:
workers still call ``map_grib_to_healpix()`` and the gather step reuses
``update_zarr_store()`` so final publication retains the same incremental merge
and metadata-consolidation behavior as the existing CLI workflow.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

import xarray as xr
from reflow import Param, Result, RunDir, Workflow

SCRIPT_DIR = Path(__file__).resolve().parent
ERA5LAND_DIR = SCRIPT_DIR.parent
if str(ERA5LAND_DIR) not in sys.path:
    sys.path.insert(0, str(ERA5LAND_DIR))

from converter import (
    DEFAULT_CMOR_TABLES,
    DEFAULT_SOURCE_MAPPER,
    DEFAULT_VAR_TABLE,
    extend_frequencies_for_special_variables,
    parse_frequencies,
    parse_interval,
    selected_requests,
)
from helpers.file_fetcher import SourceRecord, load_json, resolve_records
from helpers.formatter import (
    dataset_output_root,
    destination_for_level,
    existing_destinations_for_frequency,
)
from helpers.mapper import map_grib_to_healpix
from helpers.special import split_special_variables
from helpers.zarr_publisher import update_zarr_store

wf = Workflow("era5land_healpix")

LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")
SLURM_ACCOUNT = str(load_json(DEFAULT_SOURCE_MAPPER)["SLURM_ACCOUNT"])


def _worker_output_root(run_dir: Path, item_index: int, frequency: str, variable: str) -> Path:
    """Return the private temporary output root for one worker item."""

    safe_variable = re.sub(r"[^A-Za-z0-9._-]+", "_", variable).strip("_") or "var"
    safe_frequency = re.sub(r"[^A-Za-z0-9._-]+", "_", frequency).strip("_") or "freq"
    return run_dir / "worker-output" / f"{item_index:04d}-{safe_frequency}-{safe_variable}"


def _existing_temp_stores(
    *,
    temp_output_root: Path,
    dataset: str,
    frequency: str,
) -> list[tuple[int, Path]]:
    """Return all temporary stores for one frequency with parsed zoom levels."""

    stores: list[tuple[int, Path]] = []
    for store_name in existing_destinations_for_frequency(
        dataset,
        frequency,
        output_path=temp_output_root,
    ):
        store_path = Path(store_name)
        match = LEVEL_RE.search(store_path.name)
        if match is None:
            continue
        stores.append((int(match.group("level")), store_path))
    return sorted(stores, reverse=True)


def _merge_temp_store(
    *,
    source_store: Path,
    destination: str,
    clean: bool,
    zarr_format: int,
    target_chunk_mb: int,
) -> None:
    """Merge one temporary Zarr store into the final publication store."""

    dataset = xr.open_zarr(str(source_store), consolidated=(zarr_format == 2))
    try:
        update_zarr_store(
            dataset,
            destination,
            clean=clean,
            zarr_format=zarr_format,
            target_chunk_mb=target_chunk_mb,
        )
    finally:
        dataset.close()


@wf.job(cpus=2, time="00:20:00", mem="2GB", partition="compute", account=SLURM_ACCOUNT)
def gather_plan(
    dataset: Annotated[
        Literal["era5land", "era5"],
        Param(help="Dataset to process"),
    ] = "era5land",
    variables: Annotated[
        list[str] | None,
        Param(help="Variables to process; omit to use the default variable table"),
    ] = None,
    freq: Annotated[
        str,
        Param(help="Comma-separated output frequencies: 1hr,day,mon,fx"),
    ] = "all",
    interval: Annotated[
        str | None,
        Param(
            help=(
                "Date interval START,END where each token may be YYYY, YYYYMM, "
                "YYYYMMDD, or hyphenated equivalents."
            )
        ),
    ] = None,
    root: Annotated[
        str | None,
        Param(help="Override the ERA5 source root for alternate mounts or tests"),
    ] = None,
    output_path: Annotated[
        str | None,
        Param(help="Override the final publication root directory"),
    ] = None,
    zarr_format: Annotated[
        Literal[2, 3],
        Param(help="Zarr format version for both temporary and final stores"),
    ] = 2,
    chunk_size: Annotated[
        int,
        Param(help="Approximate target chunk size in megabytes"),
    ] = 16,
    weights_dir: Annotated[
        str,
        Param(help="Directory where HEALPix weight files are stored and reused"),
    ] = "/tmp/healpix-weights",
    clean: Annotated[
        bool,
        Param(help="Rewrite touched destination stores during the merge step"),
    ] = False,
    from_scratch: Annotated[
        bool,
        Param(help="Delete the whole dataset output root before merging final stores"),
    ] = False,
    highest_level_only: Annotated[
        bool,
        Param(help="Write only the finest HEALPix zoom level in each worker"),
    ] = False,
    use_inventory_cache: Annotated[
        bool,
        Param(help="Reuse cached GRIB inventories while reading source files"),
    ] = True,
    use_input_cache: Annotated[
        bool,
        Param(help="Reuse cached pickled multi-file input datasets"),
    ] = False,
    fail_on_duplicate_times: Annotated[
        bool,
        Param(help="Raise on duplicate GRIB timestamps instead of dropping them"),
    ] = False,
) -> dict[str, Any]:
    """Resolve the requested work into a serializable plan for downstream steps."""

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    variable_filter = tuple(variables) if variables else None
    frequencies = parse_frequencies(freq)
    parsed_interval = parse_interval(interval)
    _, requests = selected_requests(dataset=dataset, variables=variable_filter)
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, _special_variables = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        mapper_path=DEFAULT_SOURCE_MAPPER,
        dataset=dataset,
        variables=source_variables,
        frequencies=frequencies,
        interval=parsed_interval,
        root=root,
        glob_files=True,
    )

    work_items: list[dict[str, Any]] = []
    for item_index, record in enumerate(records):
        if not record.files:
            continue
        work_items.append(
            {
                "item_index": item_index,
                "frequency": record.frequency,
                "variable": record.variable,
                "record": record._asdict(),
            }
        )

    return {
        "dataset": dataset,
        "effective_frequencies": list(effective_frequencies),
        "fail_on_duplicate_times": fail_on_duplicate_times,
        "from_scratch": from_scratch,
        "highest_level_only": highest_level_only,
        "interval": interval,
        "output_path": output_path,
        "requested_variables": list(requested_variable_names),
        "root": root,
        "use_input_cache": use_input_cache,
        "use_inventory_cache": use_inventory_cache,
        "weights_dir": weights_dir,
        "work_items": work_items,
        "zarr_format": zarr_format,
        "chunk_size": chunk_size,
        "clean": clean,
    }


@wf.job(cpus=1, time="00:02:00", mem="1GB", partition="compute", account=SLURM_ACCOUNT)
def gather_work_items(
    plan: Annotated[dict[str, Any], Result(step="gather_plan")],
) -> list[dict[str, Any]]:
    """Extract the array-job inputs from the shared plan."""

    return list(plan["work_items"])


@wf.array_job(
    cpus=32,
    time="08:00:00",
    mem="0",
    partition="compute",
    account=SLURM_ACCOUNT,
    array_parallelism=12,
    after=["gather_plan"],
)
def convert_variable_frequency(
    item: Annotated[dict[str, Any], Result(step="gather_work_items")],
    plan: Annotated[dict[str, Any], Result(step="gather_plan", broadcast=True)],
    run_dir: RunDir = RunDir(),
) -> dict[str, Any]:
    """Remap one ``variable x frequency`` unit into a private temporary output root."""

    record = SourceRecord(**item["record"])
    temp_output_root = _worker_output_root(
        Path(run_dir),
        int(item["item_index"]),
        str(item["frequency"]),
        str(item["variable"]),
    )
    temp_output_root.mkdir(parents=True, exist_ok=True)

    map_grib_to_healpix(
        [record],
        dataset=str(plan["dataset"]),
        frequencies=(str(item["frequency"]),),
        requested_variables=(str(item["variable"]),),
        interval=parse_interval(plan["interval"]),
        zarr_format=int(plan["zarr_format"]),
        use_inventory_cache=bool(plan["use_inventory_cache"]),
        use_input_cache=bool(plan["use_input_cache"]),
        drop_duplicate_time_rows=(not bool(plan["fail_on_duplicate_times"])),
        weights_dir=str(plan["weights_dir"]),
        clean=True,
        target_chunk_mb=int(plan["chunk_size"]),
        highest_level_only=bool(plan["highest_level_only"]),
        output_path=temp_output_root,
    )

    return {
        "frequency": item["frequency"],
        "item_index": item["item_index"],
        "output_root": str(temp_output_root),
        "variable": item["variable"],
    }


@wf.job(cpus=8, time="04:00:00", mem="0", partition="compute", account=SLURM_ACCOUNT)
def finalize_outputs(
    worker_results: Annotated[list[dict[str, Any]], Result(step="convert_variable_frequency")],
    plan: Annotated[dict[str, Any], Result(step="gather_plan")],
    run_dir: RunDir = RunDir(),
) -> list[str]:
    """Merge all worker stores into the final publication root and consolidate metadata."""

    dataset = str(plan["dataset"])
    zarr_format = int(plan["zarr_format"])
    target_chunk_mb = int(plan["chunk_size"])
    clean = bool(plan["clean"])
    output_path = plan["output_path"]
    merged_destinations: list[str] = []
    cleaned_destinations: set[str] = set()

    if bool(plan["from_scratch"]):
        root_path = dataset_output_root(
            dataset,
            output_path=output_path,
        )
        if root_path.exists():
            shutil.rmtree(root_path)

    for result in sorted(
        worker_results,
        key=lambda current: (
            str(current["frequency"]),
            str(current["variable"]),
            int(current["item_index"]),
        ),
    ):
        temp_output_root = Path(str(result["output_root"]))
        frequency = str(result["frequency"])
        for level, temp_store in _existing_temp_stores(
            temp_output_root=temp_output_root,
            dataset=dataset,
            frequency=frequency,
        ):
            destination = destination_for_level(
                dataset,
                frequency,
                level,
                output_path=output_path,
            )
            _merge_temp_store(
                source_store=temp_store,
                destination=destination,
                clean=(clean and destination not in cleaned_destinations),
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            )
            cleaned_destinations.add(destination)
            merged_destinations.append(destination)

    requested_variables = tuple(str(name) for name in plan["requested_variables"])
    _source_variables, special_variables = split_special_variables(requested_variables)
    if special_variables:
        map_grib_to_healpix(
            [],
            dataset=dataset,
            frequencies=("fx",),
            requested_variables=special_variables,
            interval=parse_interval(plan["interval"]),
            zarr_format=zarr_format,
            use_inventory_cache=bool(plan["use_inventory_cache"]),
            use_input_cache=False,
            drop_duplicate_time_rows=True,
            weights_dir=str(plan["weights_dir"]),
            clean=False,
            target_chunk_mb=target_chunk_mb,
            highest_level_only=bool(plan["highest_level_only"]),
            output_path=output_path,
        )

    worker_root = Path(run_dir) / "worker-output"
    if worker_root.exists():
        shutil.rmtree(worker_root, ignore_errors=True)

    return sorted(set(merged_destinations))


if __name__ == "__main__":
    raise SystemExit(wf.cli())
