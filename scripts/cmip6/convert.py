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
from typing import Annotated

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


wf = Workflow("cmip6_healpix")

TARGET_CHUNK_BYTES = 16 * 1024**2  # 16 MiB

_CELL_DIM_CANDIDATES = ("cell", "cells", "value", "values", "pix", "ipix")


def _spatial_dim(ds) -> str | None:
    for name in _CELL_DIM_CANDIDATES:
        if name in ds.dims:
            return name
    non_time = {d: s for d, s in ds.sizes.items() if d != "time"}
    return max(non_time, key=non_time.get) if non_time else None


def _chunk_plan(ds, target_bytes: int = TARGET_CHUNK_BYTES) -> dict:
    """Chunk sizes so each chunk is ~target_bytes.

    Keeps the HEALPix cell dimension contiguous while a single map fits
    the budget (best for the nested layout); past that point it splits
    the cell dimension and writes one timestep per chunk.
    """
    cell = _spatial_dim(ds)
    itemsize = max((v.dtype.itemsize for v in ds.data_vars.values()), default=4)
    ncell = ds.sizes.get(cell, 1)
    other = 1
    for d, s in ds.sizes.items():
        if d not in ("time", cell):
            other *= s
    map_bytes = ncell * other * itemsize  # bytes for one full timestep

    plan: dict = {}
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
) -> list[dict]:
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
    out: list[dict] = []

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
    weights: Annotated[list[dict], Result(step="create_weights")],
    run_dir: RunDir,
) -> list[dict]:
    """Flatten into one work item per file, attaching its per-grid weight."""
    lookup = {w["s3_path"]: w for w in weights}

    staging = run_dir / "staging"
    items: list[dict] = []

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
    item: Annotated[dict, Result(step="plan_regrid")],
) -> dict:
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
    results: Annotated[list[dict], Result(step="regrid_file")],
) -> list[dict]:
    """Gather per-file outputs and group by target S3 path."""
    groups: dict[str, dict] = {}
    for r in results:
        key = r["s3_path"]
        if key not in groups:
            groups[key] = {
                "output_dir": r["output_dir"],
                "max_level": r["max_level"],
            }

    out = [{"s3_path": k, **v} for k, v in sorted(groups.items())]
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
    group: Annotated[dict, Result(step="group_for_upload")],
    uri: Annotated[str, Param(help="Target S3 bucket or path to disk")] = "s3://cmip6",
    s3_endpoint: Annotated[
        str, Param(help="S3 endpoint URL")
    ] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[
        Path, Param(help="Path to S3 credentials JSON")
    ] = Path.home() / ".s3-credentials.json",
) -> None:
    """Open per-file NetCDFs at each level, concatenate, and upload."""
    from glob import glob

    import xarray as xr

    s3_path = group["s3_path"]
    out_dir = group["output_dir"]
    max_level = group["max_level"]

    print(f"Combining and uploading {s3_path}")

    # Reassemble the pyramid: for each level, open all per-file
    # NetCDFs and concatenate along time.
    pyramid: dict[int, xr.Dataset] = {}
    for level in range(max_level + 1):
        nc_files = sorted(glob(f"{out_dir}/*_level_{level}.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No staging files for {s3_path} level {level} in {out_dir}"
            )
        ds = xr.open_mfdataset(
            nc_files,
            parallel=False,
            combine="by_coords",
            coords="minimal"
            data_vars="minimal",
            compat="override",
        )
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
