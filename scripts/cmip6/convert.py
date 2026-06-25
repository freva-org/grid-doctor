#!/usr/bin/env python
"""Scripting solution for converting CMIP6 data to healpix.

This version splits the regrid+upload into three stages so that each
source file is regridded independently (short wall-time per array
element), then combined and uploaded per dataset.

Intermediate results are stored as NetCDF on Lustre (one file per
source file per pyramid level) to avoid the metadata overhead of Zarr
on a parallel filesystem.

Pipeline
--------
gather_sources → create_weights → plan_regrid → regrid_file → group_for_upload → combine_and_upload
  (singleton)     (singleton)      (singleton)    (array)       (singleton)         (array)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypedDict

from reflow import Param, Result, RunDir, Workflow

import grid_doctor as gd

from cmip6_matrix import (
    DEFAULT_ENSEMBLE,
    DEFAULT_INSTANCE,
    DatabrowserClient,
    FacetMatrix,
    build_group_weights,
    group_key_str,
)

if TYPE_CHECKING:
    # Imported only for type checking; the real import is lazy and lives
    # inside each job so worker startup stays cheap on the login node.
    import xarray as xr


wf = Workflow("cmip6_healpix")

TARGET_CHUNK_BYTES = 16 * 1024**2  # 16 MiB

_CELL_DIM_CANDIDATES = ("cell", "cells", "value", "values", "pix", "ipix")


# ---------------------------------------------------------------------------
# Inter-step payload contracts
#
# reflow serialises each step's result and feeds it to the next, so these
# records cross a JSON boundary: paths arrive as ``str``. TypedDicts make
# the contract explicit and let mypy check every key access downstream.
# ---------------------------------------------------------------------------
class WeightInfo(TypedDict):
    """One entry per dataset emitted by ``create_weights``."""

    s3_path: str
    max_level: int
    group_weights: dict[str, str]  # grid-group key -> weight file path


class RegridItem(TypedDict):
    """One regrid work item (one source file) emitted by ``plan_regrid``."""

    s3_path: str
    source_file: str
    weight_file: str
    max_level: int
    output_dir: str
    file_index: int


class RegridResult(TypedDict):
    """Per-file output of ``regrid_file``.

    ``group_for_upload`` de-duplicates these to one record per dataset and
    passes the same shape on to ``combine_and_upload``.
    """

    s3_path: str
    output_dir: str
    max_level: int


def _spatial_dim(ds: xr.Dataset) -> str | None:
    """Return the HEALPix cell dimension, or the largest non-time dim."""
    for name in _CELL_DIM_CANDIDATES:
        if name in ds.dims:
            return name
    non_time = {d: s for d, s in ds.sizes.items() if d != "time"}
    if not non_time:
        return None
    return str(max(non_time, key=lambda d: non_time[d]))


def _drop_source_grid(ds: xr.Dataset) -> xr.Dataset:
    """Strip stale source-grid coords/bounds left over from regridding.

    Name-agnostic: the regridded data lives on (time, cell). Anything
    carrying a different dimension — lat/lon and their bounds (atmos
    grids), or vertices/vertices_latitude/longitude (unstructured ocean
    grids) — is source debris that differs between grids and only
    confuses the combine. Keep only vars whose dims fit (time, cell, bnds).
    """
    cell = _spatial_dim(ds)
    keep: set[str] = {"time", "bnds"}
    if cell is not None:
        keep.add(cell)

    drop = [
        name
        for name, var in ds.variables.items()
        if not {str(d) for d in var.dims} <= keep
    ]
    ds = ds.drop_vars(drop, errors="ignore")

    orphaned = [
        d
        for d in ds.dims
        if str(d) not in keep and not any(d in ds[name].dims for name in ds.variables)
    ]
    return ds.drop_dims(orphaned, errors="ignore")


def _open_level(nc_files: list[str]) -> xr.Dataset:
    """Assemble one pyramid level from its per-file NetCDFs.

    Each staging file holds a single data variable on (time, cell). We
    group the files by variable, concatenate each group along time, then
    merge across variables. This is deterministic and avoids
    ``combine_by_coords``' coordinate-ordering inference, which cannot
    order two files that share an identical time axis.

    Files are opened with ``use_cftime=True`` so every time axis decodes
    to the same type regardless of calendar or out-of-range dates (some
    CMIP6 runs extend past the datetime64[ns] limit). Ordering is taken
    from the zero-padded staging filename, not the time values, so we
    never compare timestamps whose decoded type can vary between files
    (cftime vs. an undecodable raw integer axis).

    A duplicate-timestamp guard turns the overlapping/stale-staging case
    into a loud error rather than a silently doubled time axis.
    """
    import xarray as xr

    per_var: dict[str, list[tuple[str, xr.Dataset]]] = {}
    for f in sorted(nc_files):
        ds = _drop_source_grid(xr.open_dataset(f, chunks={}, use_cftime=True))
        for name in ds.data_vars:
            # Selecting a single data var keeps its associated coords
            # (cell, latitude, longitude, crs, time) and drops the others.
            per_var.setdefault(str(name), []).append((f, ds[[str(name)]]))

    merged: list[xr.Dataset] = []
    for var_name, items in per_var.items():
        if len(items) == 1:
            merged.append(items[0][1])
            continue
        # file_{idx:05d}_level_{lvl}.nc: the zero-padded index sorts
        # lexicographically into chronological order, so we order by name
        # and never touch the (possibly mixed-type) time values.
        items.sort(key=lambda item: item[0])
        combined = xr.concat(
            [d for _, d in items],
            dim="time",
            coords="minimal",
            data_vars="minimal",
            compat="override",
        )
        if "time" in combined.indexes and not combined.indexes["time"].is_unique:
            raise ValueError(
                f"Duplicate timestamps after concatenating {len(items)} files "
                f"for variable {var_name!r}; the staging dir likely contains "
                f"overlapping or stale files. Clean it before re-running "
                f"(or add .drop_duplicates('time') if overlaps are expected)."
            )
        merged.append(combined)

    return xr.merge(merged, compat="override", join="override")


def _chunk_plan(
    ds: xr.Dataset, target_bytes: int = TARGET_CHUNK_BYTES
) -> dict[str, int]:
    """Chunk sizes so each chunk is ~target_bytes.

    Sizes the estimate on the largest *data variable's* per-timestep
    footprint, so the presence of tiny side-cars like time_bnds doesn't
    skew the budget. Keeps the cell dimension contiguous while one map
    fits target_bytes; past that, splits cell with one timestep per chunk.
    """
    cell = _spatial_dim(ds)

    def per_step_bytes(var: xr.DataArray) -> int:
        n = var.dtype.itemsize
        for dim, size in var.sizes.items():
            if dim != "time":
                n *= size
        return n

    data_vars = [v for v in ds.data_vars.values() if "time" in v.dims]
    if not data_vars:
        return {}
    map_bytes = max(1, max(per_step_bytes(v) for v in data_vars))
    ncell = ds.sizes.get(cell, 1) if cell is not None else 1

    plan: dict[str, int] = {}
    if map_bytes <= target_bytes:
        if cell is not None:
            plan[cell] = -1
        if "time" in ds.dims:
            plan["time"] = max(1, target_bytes // map_bytes)
    else:
        if "time" in ds.dims:
            plan["time"] = 1
        if cell is not None:
            n = -(-map_bytes // target_bytes)  # ceil
            plan[cell] = max(1, ncell // n)

    # A chunk plan must cover *every* dimension in the dataset. Side-cars
    # like time_bnds carry a dim (bnds) that is neither time nor cell;
    # leaving it unspecified yields a per-variable chunk spec shorter than
    # the variable's rank, which this xarray rejects with a strict-zip
    # error. Default any uncovered dim to a single full chunk.
    for d in ds.dims:
        plan.setdefault(str(d), -1)
    return plan


# ---------------------------------------------------------------------------
# Step 1: Discover source files
# ---------------------------------------------------------------------------
@wf.job(cpus=2, time="00:10:00", mem="1GB", partition="shared", version="4")  # noqa: F821
def gather_sources(
    variable: Annotated[
        list[str], Param(help="Variables every dataset must contain", short="-v")
    ] = ["tas", "pr"],
    experiment: Annotated[
        list[str],
        Param(help="Experiments; one dataset group per experiment", short="-e"),
    ] = ["ssp585", "ssp245", "ssp370", "historical"],
    freq: Annotated[list[str], Param(help="Time frequencies", short="-f")] = ["6hr"],
    models: Annotated[
        list[str],
        Param(help="Explicit model list; empty discovers all complete models"),
    ] = [],
    max_models: Annotated[
        int, Param(help="Cap on discovered models (0 = no limit)")
    ] = 0,
    ensemble: Annotated[
        str,
        Param(help="Preferred ensemble; falls back to the first complete one"),
    ] = DEFAULT_ENSEMBLE,
    instance: Annotated[str, Param(help="Databrowser base URL")] = DEFAULT_INSTANCE,
    flavour: Annotated[str, Param(help="DRS flavour")] = "freva",
) -> list[tuple[str, list[str]]]:
    """Build the (experiment, frequency, model) dataset matrix.

    Every emitted dataset is guaranteed to contain all requested
    variables, with a single ensemble member chosen per combination
    (the default if it carries all variables, else the first that does).
    """
    matrix = FacetMatrix(
        variables=variable,
        experiments=experiment,
        frequencies=freq,
        models=models or None,
        max_models=max_models or None,
        default_ensemble=ensemble,
    )
    client = DatabrowserClient(instance, flavour=flavour)

    entries = matrix.build(client)
    print(f"Matrix produced {len(entries)} complete datasets:")
    for entry in entries:
        counts = {v: len(f) for v, f in entry.files_by_variable.items()}
        print(f"  {entry.key}  (ensemble={entry.ensemble})  files={counts}")

    return [(entry.key, entry.files) for entry in entries]


# ---------------------------------------------------------------------------
# Step 2: Create / cache ESMF weight files
# ---------------------------------------------------------------------------
@wf.job(
    cpus=128,
    time="08:00:00",
    partition="compute",
    mem="0",
    version="2",
)
def create_weights(
    paths: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    weights_dir: Annotated[
        Path, Param(help="Path to the grid weight directory")
    ] = Path("/work/ks1387/healpix-weights"),
    level: Annotated[
        int,
        Param(help="Target HEALPix level; 0 = auto (finest source grid per dataset)"),
    ] = 0,
) -> list[WeightInfo]:
    """Create one ESMF weight file per distinct source grid per dataset.

    A dataset may mix source grids (e.g. ``tos`` on an ocean grid,
    ``tas``/``pr``/``uas`` on an atmosphere grid). Files are grouped by
    grid (variable + grid_label); each grid gets its own weight file,
    generated at one shared HEALPix level so the per-level NetCDFs still
    combine into a single pyramid. CMIP6 unstructured corner coordinates
    are normalised first so ocean grids (e.g. FESOM/AWI-CM) work too.
    """
    import xarray as xr

    weights_dir.mkdir(exist_ok=True, parents=True)
    out: list[WeightInfo] = []

    for s3_path, source_paths in paths:
        try:
            target_level, native_levels, group_weights = build_group_weights(
                source_paths,
                level=level,
                open_dataset=xr.open_dataset,
                resolution_level=lambda ds: gd.resolution_to_healpix_level(
                    gd.get_latlon_resolution(ds)
                ),
                make_weights=lambda ds, lvl: gd.cached_weights(
                    ds,
                    level=lvl,
                    cache_path=weights_dir,
                    # See note in module docs: HEALPix is a global target,
                    # ocean/partial grids leave land/pole cells unmapped.
                    ignore_unmapped=True,
                ),
            )
        except Exception as exc:
            # One bad grid (e.g. an unstructured ocean grid ESMF cannot
            # triangulate) must not abort weight generation for every other
            # dataset. Skip this dataset; it will be dropped downstream
            # because it is missing from the weights result.
            print(f"SKIP dataset {s3_path}: weight generation failed: {exc!r}")
            continue

        for key, weight_file in group_weights.items():
            print(
                f"{s3_path} grid={key} "
                f"native_level={native_levels[key]} target_level={target_level} "
                f"weights={weight_file}"
            )

        out.append(
            {
                "s3_path": s3_path,
                "max_level": target_level,
                "group_weights": group_weights,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Step 3: Explode into per-file work items
# ---------------------------------------------------------------------------
@wf.job(cpus=1, time="00:05:00", mem="1GB", partition="shared")
def plan_regrid(
    sources: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    weights: Annotated[list[WeightInfo], Result(step="create_weights")],
    run_dir: RunDir,
) -> list[RegridItem]:
    """Flatten into one work item per file, attaching its per-grid weight."""
    lookup: dict[str, WeightInfo] = {w["s3_path"]: w for w in weights}

    staging = run_dir / "staging"
    items: list[RegridItem] = []

    for s3_path, source_files in sources:
        info = lookup.get(s3_path)
        if info is None:
            # Dataset was skipped in create_weights (e.g. a source grid that
            # could not produce weights). Drop it here too rather than fail.
            print(f"SKIP dataset {s3_path}: no weights (incomplete), not regridding")
            continue
        group_weights = info["group_weights"]
        max_level = info["max_level"]
        safe_dir = s3_path.replace("/", "__")
        out_dir = str(staging / safe_dir)

        for idx, src in enumerate(sorted(source_files)):
            key = group_key_str(src)
            weight_file = group_weights.get(key)
            if weight_file is None:
                raise KeyError(
                    f"No weight file for grid group {key!r} of {src}; "
                    f"available groups: {sorted(group_weights)}"
                )
            items.append(
                {
                    "s3_path": s3_path,
                    "source_file": src,
                    "weight_file": weight_file,
                    "max_level": max_level,
                    "output_dir": out_dir,
                    "file_index": idx,
                }
            )

    print(f"Planned {len(items)} regrid tasks across {len(sources)} datasets")
    return items


# ---------------------------------------------------------------------------
# Step 4: Regrid one source file → full HEALPix pyramid as NetCDF
# ---------------------------------------------------------------------------
@wf.array_job(
    cpus=32,
    time="02:00:00",
    mem="0",
    partition="compute",
    array_parallelism=6,
)
def regrid_file(
    item: Annotated[RegridItem, Result(step="plan_regrid")],
) -> RegridResult:
    """Regrid a single source file and write every pyramid level to NetCDF.

    Produces one NetCDF per level::

        <output_dir>/file_00042_level_7.nc
        <output_dir>/file_00042_level_6.nc
        ...
        <output_dir>/file_00042_level_0.nc
    """
    import xarray as xr

    src = item["source_file"]
    max_level = item["max_level"]
    weight_file = item["weight_file"]
    out_dir = Path(item["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    file_idx = item["file_index"]

    print(f"Regridding {src}")
    ds = xr.open_dataset(src)

    pyramid = gd.create_healpix_pyramid(
        ds, max_level=max_level, weights_path=weight_file
    )

    for level, level_ds in pyramid.items():
        nc_path = out_dir / f"file_{file_idx:05d}_level_{level}.nc"
        level_ds.load().to_netcdf(nc_path)

    return {
        "s3_path": item["s3_path"],
        "output_dir": str(out_dir),
        "max_level": max_level,
    }


# ---------------------------------------------------------------------------
# Step 5: Group per-file results back by dataset
# ---------------------------------------------------------------------------
@wf.job(cpus=1, time="00:05:00", mem="1GB", partition="shared")
def group_for_upload(
    results: Annotated[list[RegridResult], Result(step="regrid_file")],
) -> list[RegridResult]:
    """Gather per-file outputs and group by target S3 path."""
    groups: dict[str, RegridResult] = {}
    for r in results:
        key = r["s3_path"]
        if key not in groups:
            groups[key] = {
                "s3_path": key,
                "output_dir": r["output_dir"],
                "max_level": r["max_level"],
            }

    out = sorted(groups.values(), key=lambda g: g["s3_path"])
    for g in out:
        print(f"  {g['s3_path']}: level 0-{g['max_level']}")
    return out


# ---------------------------------------------------------------------------
# Step 6: Combine per-file NetCDFs and upload as Zarr pyramid to S3
# ---------------------------------------------------------------------------
@wf.array_job(
    cpus=32,
    time="08:00:00",
    mem="0",
    partition="compute",
    array_parallelism=8,
)
def combine_and_upload(
    group: Annotated[RegridResult, Result(step="group_for_upload")],
    uri: Annotated[str, Param(help="Target S3 bucket or path to disk")] = "s3://cmip6",
    s3_endpoint: Annotated[
        str, Param(help="S3 endpoint URL")
    ] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[
        Path, Param(help="Path to S3 credentials JSON")
    ] = Path.home() / ".s3-credentials.json",
) -> None:
    """Open per-file NetCDFs at each level, concatenate, rechunk, and upload."""
    from glob import glob

    s3_path = group["s3_path"]
    out_dir = group["output_dir"]
    max_level = group["max_level"]

    print(f"Combining and uploading {s3_path}")

    # Reassemble the pyramid: for each level, open all per-file NetCDFs,
    # strip the source-grid debris, concatenate along time per variable,
    # and rechunk to the 16 MiB target so Blosc never sees a > 2 GiB buffer.
    pyramid: dict[int, xr.Dataset] = {}
    for level in range(max_level + 1):
        nc_files = sorted(glob(f"{out_dir}/*_level_{level}.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No staging files for {s3_path} level {level} in {out_dir}"
            )
        ds = _open_level(nc_files)

        # Drop the NetCDF chunk hints so they don't clash with the dask
        # chunks that zarr will use as its on-disk chunking.
        for v in ds.variables.values():
            v.encoding.pop("chunks", None)
            v.encoding.pop("preferred_chunks", None)

        pyramid[level] = ds.chunk(_chunk_plan(ds))

    s3_options = gd.get_s3_options(s3_endpoint, s3_credentials_file)
    gd.save_pyramid(
        pyramid,
        f"{uri}/{s3_path}",
        s3_options if uri.startswith("s3://") else None,
        mode="w",
    )
    print(f"Uploaded {s3_path}")


if __name__ == "__main__":
    wf.cli()
