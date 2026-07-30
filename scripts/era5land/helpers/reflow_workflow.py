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

import hashlib
import json
import re
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from reflow import Param, Result, RunDir, Workflow

SCRIPT_DIR = Path(__file__).resolve().parent
ERA5LAND_DIR = SCRIPT_DIR.parent
if str(ERA5LAND_DIR) not in sys.path:
    sys.path.insert(0, str(ERA5LAND_DIR))

from converter import (
    DEFAULT_CMOR_TABLES,
    DEFAULT_VAR_TABLE,
    DEFAULT_SOURCE_MAPPER,
    batched_intervals,
    extend_frequencies_for_special_variables,
    format_interval,
    parse_cli_args,
    parse_cli_freqs,
    parse_interval,
    selected_requests,
)
from helpers.file_fetcher import (
    SourceRecord,
    batched_source_record_files,
    load_json,
    resolve_records,
)
from helpers.grib import cached_grib_inventory
from helpers.formatter import (
    dataset_output_root,
    destination_for_level,
)
from helpers.zarr_publisher import merge_zarr_stores

wf = Workflow("era5land_healpix")

LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")
SOURCE_MAPPER = load_json(DEFAULT_SOURCE_MAPPER)
REFLOW_BATCHING_POLICY = SOURCE_MAPPER.get("reflow_batching_policy", {})
DEFAULT_WEIGHTS_DIR = str(SOURCE_MAPPER["weights_path"])


def _level_policy(dataset: str, level_type: str) -> dict[str, Any]:
    """Return the configured policy for one dataset and level category."""

    return REFLOW_BATCHING_POLICY["datasets"][dataset][level_type]


REFLOW_WAVE_SIZE = max(
    int(level_policy["wave_size"])
    for dataset_policy in REFLOW_BATCHING_POLICY["datasets"].values()
    for level_policy in dataset_policy.values()
)


def _array_resources() -> dict[str, Any]:
    """Return the common resources configured for the Reflow array job."""

    resources = {
        tuple(level_policy["resources"].items())
        for dataset_policy in REFLOW_BATCHING_POLICY["datasets"].values()
        for level_policy in dataset_policy.values()
    }
    if len(resources) != 1:
        raise ValueError(
            "Reflow array resources must currently be identical for all workloads."
        )
    return dict(next(iter(resources)))


REFLOW_ARRAY_RESOURCES = _array_resources()


def _batching_level_type(record: SourceRecord) -> str:
    """Classify one resolved record for Reflow batching policy lookup.

    The CMOR tables expose raw ERA5 level types such as ``pl_an`` and
    ``sfc_fc``. Reflow batching uses the coarser categories ``pressure``,
    ``surface``, and ``fixed``.
    """

    if record.frequency == "fx":
        return "fixed"

    raw_level_type = str(record.output_attrs.get("level_type") or record.level_type)
    if raw_level_type.startswith("pl"):
        return "pressure"
    return "surface"


def _worker_output_root(
    run_dir: Path,
    item_index: int,
    frequency: str,
    variable: str,
    run_token: str,
) -> Path:
    """Return a collision-resistant temporary output root for one worker."""

    safe_variable = re.sub(r"[^A-Za-z0-9._-]+", "_", variable).strip("_") or "var"
    safe_frequency = re.sub(r"[^A-Za-z0-9._-]+", "_", frequency).strip("_") or "freq"
    identity = f"{run_token}:{item_index}:{frequency}:{variable}".encode("utf-8")
    output_hash = hashlib.sha256(identity).hexdigest()[:12]
    return run_dir / "worker-output" / (
        f"{item_index:05d}-{output_hash}-{safe_frequency}-{safe_variable}"
    )


def _batch_settings_for_item(
    dataset: str,
    frequency: str,
    level_type: str,
    batch_files: int | None,
    batch_months: int | None,
) -> tuple[int | None, int | None]:
    """Return file and calendar-month batching for one workload item."""

    if batch_files is not None:
        return batch_files, None
    if batch_months is not None:
        return None, batch_months
    settings = _level_policy(dataset, level_type)["frequencies"].get(frequency, {})
    return settings.get("batch_files"), settings.get("batch_months")


def _batched_work_items(
    *,
    records: list[SourceRecord],
    parsed_interval: tuple[object | None, object | None],
    dataset: str,
    batch_files: int | None,
    batch_months: int | None,
    pressure_level_group_size: int | None,
) -> list[dict[str, Any]]:
    """Expand resolved records into workload-aware Reflow array items."""

    work_items: list[dict[str, Any]] = []

    for record in records:
        if not record.files:
            continue

        level_type = _batching_level_type(record)
        effective_batch_files, effective_batch_months = _batch_settings_for_item(
            dataset, record.frequency, level_type, batch_files, batch_months
        )
        group_size = pressure_level_group_size
        if group_size is None:
            group_size = _level_policy(dataset, level_type)["pressure_level_group_size"]

        if effective_batch_files is not None:
            record_batches = batched_source_record_files(
                record,
                batch_files=effective_batch_files,
                fallback_interval=parsed_interval,
            )
        else:
            record_batches = tuple(
                (record, batch_interval)
                for batch_interval in batched_intervals(
                    parsed_interval,
                    batch_months=effective_batch_months,
                )
            )

        pressure_level_groups = _pressure_level_groups_for_record(
            record,
            pressure_level_group_size=int(group_size),
        )

        for batch_index, (batched_record, batch_interval) in enumerate(record_batches):
            for pressure_levels in pressure_level_groups:
                work_items.append(
                    {
                        "batch_index": batch_index,
                        "batch_interval": format_interval(batch_interval),
                        "files": list(batched_record.files),
                        "frequency": record.frequency,
                        "pressure_levels": (
                            list(pressure_levels) if pressure_levels is not None else None
                        ),
                        "variable": record.variable,
                    }
                )

    for item_index, item in enumerate(work_items):
        item["item_index"] = item_index
    return work_items


def _chunk_pressure_levels(
    levels: tuple[int, ...],
    *,
    chunk_size: int,
) -> list[tuple[int, ...]]:
    """Split pressure levels into stable groups for array execution."""

    if chunk_size <= 0:
        return [levels]
    return [
        levels[index:index + chunk_size]
        for index in range(0, len(levels), chunk_size)
    ]


def _record_pressure_levels(record: SourceRecord) -> tuple[int, ...] | None:
    """Return distinct pressure levels found in one resolved source record."""

    inventory = cached_grib_inventory(record.files)
    if "level" not in inventory:
        return None

    level_values = sorted(
        {
            int(level)
            for level in inventory["level"].dropna().tolist()
        },
        reverse=True,
    )
    return tuple(level_values) if level_values else None


def _pressure_level_groups_for_record(
    record: SourceRecord,
    *,
    pressure_level_group_size: int,
) -> list[tuple[int, ...] | None]:
    """Return worker pressure-level groups for one resolved source record."""

    if _batching_level_type(record) != "pressure":
        return [None]
    if pressure_level_group_size <= 0:
        return [None]

    pressure_levels = _record_pressure_levels(record)
    if not pressure_levels:
        return [None]
    return _chunk_pressure_levels(
        pressure_levels,
        chunk_size=pressure_level_group_size,
    )


def _record_to_payload(record: SourceRecord) -> dict[str, Any]:
    """Convert a resolved source record into a Reflow-serializable mapping."""

    return dict(record._asdict())


def _record_from_payload(payload: dict[str, Any]) -> SourceRecord:
    """Reconstruct a resolved source record from a gathered worker payload."""

    return SourceRecord(**payload)


def _load_record_cache(path: str | Path) -> dict[str, SourceRecord]:
    """Load the gathered source-record cache keyed by variable and frequency."""

    with Path(path).open(encoding="utf-8") as handle:
        payloads = json.load(handle)
    return {str(key): _record_from_payload(value) for key, value in payloads.items()}


@wf.job(cpus=2, time="05:00:00", mem="40GB", partition="shared", cache=False)
def gather_plan(
    dataset: Annotated[
        Literal["era5land", "era5"],
        Param(help="Dataset to process"),
    ] = "era5land",
    var: Annotated[
        str | None,
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
    batch_files: Annotated[
        int | None,
        Param(
            help=(
                "Optional number of source files to process per worker item. "
                "When set, this replaces calendar-month batching."
            )
        ),
    ] = None,
    batch_months: Annotated[
        int | None,
        Param(
            help=(
                "Optional number of calendar months to process per worker item. "
                "When set, this replaces the policy batching value."
            )
        ),
    ] = None,
    weights_dir: Annotated[
        str,
        Param(help="Directory where HEALPix weight files are stored and reused"),
    ] = DEFAULT_WEIGHTS_DIR,
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
    pressure_level_group_size: Annotated[
        int | None,
        Param(
            help=(
                "Number of pressure levels to process per worker item. "
                "Omit to use the dataset policy."
            )
        ),
    ] = None,
    run_dir: RunDir = RunDir(),
) -> list[dict[str, Any]]:
    """Resolve the request once and return self-contained array-job payloads.

    The returned list is deliberately the direct input to the Reflow array
    job. Each payload contains one work item and the shared run settings, so
    no downstream task needs to resolve source records or rebuild batches.
    """

    from helpers.special import split_special_variables

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")
    if batch_files is not None and batch_files <= 0:
        raise ValueError("--batch-files must be a positive integer.")
    if batch_months is not None and batch_months <= 0:
        raise ValueError("--batch-months must be a positive integer.")
    if batch_files is not None and batch_months is not None:
        raise ValueError("--batch-files and --batch-months are mutually exclusive.")

    variable_filter = parse_cli_args(var)
    frequencies = parse_cli_freqs(freq)
    parsed_interval = parse_interval(interval)
    _, requests = selected_requests(
        dataset=dataset,
        variables=variable_filter,
        var_table=DEFAULT_VAR_TABLE,
    )
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, _special_variables = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        dataset=dataset,
        variables=source_variables,
        frequencies=frequencies,
        interval=parsed_interval,
        root=root,
        glob_files=True,
    )
    work_items = _batched_work_items(
        records=records,
        parsed_interval=parsed_interval,
        dataset=dataset,
        batch_files=batch_files,
        batch_months=batch_months,
        pressure_level_group_size=pressure_level_group_size,
    )
    worker_output_token = uuid.uuid4().hex
    record_cache_path = Path(run_dir) / f"source-records-{worker_output_token}.json"
    record_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with record_cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                f"{record.variable}|{record.frequency}": _record_to_payload(record)
                for record in records
            },
            handle,
            sort_keys=True,
        )

    plan = {
        "batching_policy": SOURCE_MAPPER.get("reflow_batching_policy", {}),
        "batch_files": batch_files,
        "batch_months": batch_months,
        "dataset": dataset,
        "effective_frequencies": list(effective_frequencies),
        "fail_on_duplicate_times": fail_on_duplicate_times,
        "from_scratch": from_scratch,
        "highest_level_only": highest_level_only,
        "interval": interval,
        "output_path": output_path,
        "pressure_level_group_size": pressure_level_group_size,
        "requested_variables": list(requested_variable_names),
        "root": root,
        "record_cache_path": str(record_cache_path),
        "use_input_cache": use_input_cache,
        "use_inventory_cache": use_inventory_cache,
        "weights_dir": weights_dir,
        "work_item_count": len(work_items),
        "worker_output_token": worker_output_token,
        "zarr_format": zarr_format,
        "chunk_size": chunk_size,
        "clean": clean,
    }
    return [
        {
            "item": item,
            "plan": plan,
            "record_key": f"{item['variable']}|{item['frequency']}",
        }
        for item in work_items
    ]


@wf.array_job(
    cpus=int(REFLOW_ARRAY_RESOURCES["cpus"]),
    time=str(REFLOW_ARRAY_RESOURCES["time"]),
    mem=str(REFLOW_ARRAY_RESOURCES["mem"]),
    partition="shared",
    array_parallelism=REFLOW_WAVE_SIZE,
    after=["gather_plan"],
)
def remap_variable_frequency(
    payload: Annotated[dict[str, Any], Result(step="gather_plan")],
    run_dir: RunDir = RunDir(),
) -> dict[str, Any]:
    """Remap one gathered ``variable x frequency x interval-batch`` payload."""

    from helpers.mapper import map_grib_to_healpix

    item = payload["item"]
    plan = payload["plan"]
    records = _load_record_cache(str(plan["record_cache_path"]))
    record = records[str(payload["record_key"])]
    record = record._replace(files=tuple(str(path) for path in item["files"]))
    temp_output_root = _worker_output_root(
        Path(run_dir),
        int(item["item_index"]),
        str(item["frequency"]),
        str(item["variable"]),
        str(plan["worker_output_token"]),
    )
    temp_output_root.mkdir(parents=True, exist_ok=True)

    map_grib_to_healpix(
        [record],
        dataset=str(plan["dataset"]),
        frequencies=(str(item["frequency"]),),
        requested_variables=(str(item["variable"]),),
        interval=parse_interval(str(item["batch_interval"])),
        pressure_levels=(
            tuple(int(level) for level in item["pressure_levels"])
            if item.get("pressure_levels")
            else None
        ),
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
        "batch_index": item["batch_index"],
        "batch_interval": item["batch_interval"],
        "frequency": item["frequency"],
        "item_index": item["item_index"],
        "output_root": str(temp_output_root),
        "pressure_levels": item.get("pressure_levels"),
        "variable": item["variable"],
    }


@wf.job(cpus=137, time="20:00:00", mem="128GB", partition="shared")
def finalize_outputs(
    worker_results: Annotated[list[dict[str, Any]], Result(step="remap_variable_frequency")],
    plan_payloads: Annotated[list[dict[str, Any]], Result(step="gather_plan")],
    run_dir: RunDir = RunDir(),
) -> list[str]:
    """Merge all worker stores into the final publication root and consolidate metadata."""

    from helpers.mapper import map_grib_to_healpix
    from helpers.special import split_special_variables

    if not plan_payloads:
        raise ValueError("gather_plan returned no work payloads.")
    plan = plan_payloads[0]["plan"]
    dataset = str(plan["dataset"])
    zarr_format = int(plan["zarr_format"])
    target_chunk_mb = int(plan["chunk_size"])
    clean = bool(plan["clean"])
    output_path = plan["output_path"]
    merged_destinations: list[str] = []
    worker_roots_by_frequency: dict[str, list[Path]] = defaultdict(list)

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
        worker_roots_by_frequency[frequency].append(temp_output_root)

    for frequency, source_roots in worker_roots_by_frequency.items():
        target_dir = Path(
            destination_for_level(
                dataset,
                frequency,
                0,
                output_path=output_path,
            )
        ).parent.parent
        merged_destinations.extend(
            merge_zarr_stores(
                sources=source_roots,
                target_dir=target_dir,
                dataset=dataset,
                frequency=frequency,
                clean=clean,
                zarr_format=zarr_format,
                target_chunk_mb=target_chunk_mb,
            )
        )

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

    for result in worker_results:
        shutil.rmtree(Path(str(result["output_root"])), ignore_errors=True)

    worker_root = Path(run_dir) / "worker-output"
    try:
        worker_root.rmdir()
    except OSError:
        pass

    return sorted(set(merged_destinations))


if __name__ == "__main__":
    raise SystemExit(wf.cli())
