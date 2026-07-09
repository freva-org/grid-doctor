#!/usr/bin/env python
"""Convert CORDEX-protocol regional data (e.g. NUKLEUS) to HEALPix.

Same divide-and-conquer architecture as the CMIP6 workflow — each
source file is regridded independently (short wall-time per array
element), staged as per-level NetCDF on Lustre, then combined and
uploaded per dataset — with the regional additions from the
"Regional Datasets" recipe:

- **Coverage masking.** A ones-field is pushed through the same
  conservative weights once per source grid; every regridded file is
  masked at the *finest* level (before coarsening, so boundary
  extrapolation never contaminates parent cells) where coverage falls
  below the threshold. ``coverage_fraction`` is published alongside the
  data.
- **Regional chunking.** Power-of-four cell chunks (each chunk = one
  coarse parent cell, spatially compact) with NaN fill, so all-empty
  chunks outside the domain are elided and storage stays proportional
  to the domain.
- **Bounding-box attributes** (wrap-aware) so viewers zoom to the
  domain instead of opening on an empty globe.

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

from cordex_matrix import (
    DEFAULT_ENSEMBLE,
    DEFAULT_INSTANCE,
    CordexMatrix,
    DatabrowserClient,
    build_group_weights,
    cordex_group_key,
    domain_bbox,
    make_ones_dataset,
)

if TYPE_CHECKING:
    import xarray as xr


wf = Workflow("cordex_healpix")

TARGET_CHUNK_BYTES = 16 * 1024**2  # 16 MiB
CELL_CHUNK_EXPONENT = 8  # 4**8 = 65536 cells per spatial chunk
COVERAGE_VAR = "coverage_fraction"

_CELL_DIM_CANDIDATES = ("cell", "cells", "value", "values", "pix", "ipix")


class WeightInfo(TypedDict):
    """One entry per dataset emitted by ``create_weights``."""

    s3_path: str
    max_level: int
    group_weights: dict[str, str]   # grid-group key -> weight file path
    group_coverage: dict[str, str]  # grid-group key -> coverage NetCDF


class RegridItem(TypedDict):
    """One regrid work item (one source file)."""

    s3_path: str
    source_file: str
    weight_file: str
    coverage_file: str
    max_level: int
    output_dir: str
    file_index: int


class RegridResult(TypedDict):
    """Per-file output; de-duplicated to one record per dataset."""

    s3_path: str
    output_dir: str
    max_level: int


# ---------------------------------------------------------------------------
# Helpers (shared shape with the CMIP6 script; duplicated so each script
# folder stays self-contained and independently runnable)
# ---------------------------------------------------------------------------
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

    Regridded data lives on (time, cell); rotated-pole leftovers
    (rlat/rlon, rotated_pole mapping variables, 2-D lat/lon and their
    bounds) only confuse the combine.
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
    # Scalar grid-mapping variables (e.g. ``rotated_pole``) have no dims
    # at all and would survive the dim test; keep only the HEALPix crs.
    drop += [
        str(name)
        for name in ds.variables
        if ds[name].ndim == 0 and str(name) != "crs"
        and "grid_mapping_name" in ds[name].attrs
    ]
    ds = ds.drop_vars(drop, errors="ignore")

    orphaned = [
        d
        for d in ds.dims
        if str(d) not in keep
        and not any(d in ds[name].dims for name in ds.variables)
    ]
    return ds.drop_dims(orphaned, errors="ignore")


def _open_level(nc_files: list[str]) -> xr.Dataset:
    """Assemble one pyramid level from its per-file NetCDFs.

    Same contract as the CMIP6 combine: group staging files by variable,
    concatenate along time in filename order (zero-padded index), merge
    across variables, guard against duplicated timestamps.  The
    time-less ``coverage_fraction`` is written by exactly one file per
    dataset, so it always lands in the single-item branch and is merged
    as-is.
    """
    import xarray as xr

    per_var: dict[str, list[tuple[str, xr.Dataset]]] = {}
    for f in sorted(nc_files):
        ds = _drop_source_grid(xr.open_dataset(f, chunks={}, use_cftime=True))
        for name in ds.data_vars:
            per_var.setdefault(str(name), []).append((f, ds[[str(name)]]))

    merged: list[xr.Dataset] = []
    for var_name, items in per_var.items():
        if len(items) == 1:
            merged.append(items[0][1])
            continue
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
                f"for variable {var_name!r}; clean the staging dir before "
                f"re-running."
            )
        merged.append(combined)

    return xr.merge(merged, compat="override", join="override")


def _regional_chunk_plan(
    ds: xr.Dataset,
    target_bytes: int = TARGET_CHUNK_BYTES,
    cell_exponent: int = CELL_CHUNK_EXPONENT,
) -> dict[str, int]:
    """Chunk plan for regional data: power-of-four cell chunks.

    Each spatial chunk of ``4**cell_exponent`` cells is exactly one
    coarse parent cell, so chunks are spatially compact and the many
    all-NaN chunks outside the domain are elided (NaN fill).  The time
    chunk is then sized so a chunk lands near *target_bytes*.  This
    replaces the CMIP6 plan, which prefers a full-globe cell chunk —
    optimal for global data, but it would defeat elision for a domain
    covering a few percent of the sphere.
    """
    cell = _spatial_dim(ds)
    plan: dict[str, int] = {}
    if cell is not None:
        ncell = int(ds.sizes[cell])
        cell_chunk = min(4**cell_exponent, ncell)
        plan[cell] = cell_chunk
        if "time" in ds.dims:
            itemsize = max(
                (v.dtype.itemsize for v in ds.data_vars.values() if cell in v.dims),
                default=4,
            )
            plan["time"] = max(1, target_bytes // (cell_chunk * itemsize))
    for d in ds.dims:
        plan.setdefault(str(d), -1)
    return plan


def _mask_with_coverage(
    ds: xr.Dataset,
    coverage: xr.DataArray,
    threshold: float,
) -> xr.Dataset:
    """NaN out under-covered cells in every cell-dimensioned data var.

    Applied per variable (not ``Dataset.where``) so side-cars without
    the cell dimension — ``time_bnds`` and friends — are not broadcast
    onto the grid.
    """
    cell = _spatial_dim(ds)
    out = ds.copy()
    keep = coverage >= threshold
    for name in out.data_vars:
        if cell is not None and cell in out[name].dims:
            attrs = dict(out[name].attrs)
            out[name] = out[name].where(keep)
            out[name].attrs = attrs
    return out


# ---------------------------------------------------------------------------
# Step 1: Discover source files via the Freva databrowser
# ---------------------------------------------------------------------------
@wf.job(cpus=2, time="00:10:00", mem="1GB", partition="shared")
def gather_sources(
    project: Annotated[
        str, Param(help="Freva project facet (CORDEX protocol)", short="-p")
    ] = "nukleus",
    product: Annotated[
        str, Param(help="Freva product facet; empty = unconstrained")
    ] = "",
    experiment: Annotated[
        list[str],
        Param(help="Experiments; empty discovers all", short="-e"),
    ] = [],
    freq: Annotated[list[str], Param(help="Time frequencies", short="-f")] = ["day"],
    variable: Annotated[
        list[str],
        Param(help="Variables; empty uses ALL available per dataset", short="-v"),
    ] = [],
    exclude_variable: Annotated[
        list[str], Param(help="Variables to skip when discovering all")
    ] = [],
    models: Annotated[
        list[str], Param(help="Explicit RCM list; empty discovers all")
    ] = [],
    driving_models: Annotated[
        list[str], Param(help="Explicit driving-model list; empty discovers all")
    ] = [],
    max_models: Annotated[int, Param(help="Cap on discovered RCMs (0 = all)")] = 0,
    ensemble: Annotated[
        str, Param(help="Preferred ensemble; falls back to first available")
    ] = DEFAULT_ENSEMBLE,
    instance: Annotated[str, Param(help="Databrowser base URL")] = DEFAULT_INSTANCE,
    flavour: Annotated[str, Param(help="DRS flavour")] = "freva",
) -> list[tuple[str, list[str]]]:
    """Build the (experiment, frequency, RCM, driving model) matrix.

    Every (RCM, driving model) pair is its own output dataset — an RCM
    driven by two GCMs must never be mixed into one store. With an
    empty ``--variable`` list, each dataset carries *all* variables the
    databrowser reports for its combination.
    """
    matrix = CordexMatrix(
        project=project,
        product=product or None,
        experiments=experiment,
        frequencies=freq,
        variables=variable,
        exclude_variables=exclude_variable,
        models=models,
        driving_models=driving_models,
        max_models=max_models or None,
        default_ensemble=ensemble,
    )
    client = DatabrowserClient(instance, flavour=flavour)

    entries = matrix.build(client)
    print(f"Matrix produced {len(entries)} datasets:")
    for entry in entries:
        counts = {v: len(f) for v, f in entry.files_by_variable.items()}
        print(f"  {entry.key}  (ensemble={entry.ensemble})  files={counts}")

    return [(entry.key, entry.files) for entry in entries]


# ---------------------------------------------------------------------------
# Step 2: Create / cache ESMF weight files + per-grid coverage fields
# ---------------------------------------------------------------------------
@wf.job(cpus=128, time="08:00:00", partition="compute", mem="0")
def create_weights(
    paths: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    run_dir: RunDir,
    weights_dir: Annotated[
        Path, Param(help="Path to the grid weight directory")
    ] = Path("/work/ks1387/healpix-weights"),
    level: Annotated[
        int, Param(help="Target HEALPix level; 0 = auto (from source resolution)")
    ] = 0,
) -> list[WeightInfo]:
    """One weight file *and one coverage field* per source grid.

    The coverage field is the ones-field remapped through the same
    conservative weights: conservative weights are normalised by the
    full destination-cell area, so the result is the fraction of each
    HEALPix cell covered by the source domain. Computed once per grid,
    reused by every file of the dataset.
    """
    import xarray as xr

    weights_dir.mkdir(exist_ok=True, parents=True)
    coverage_dir = run_dir / "coverage"
    coverage_dir.mkdir(exist_ok=True, parents=True)
    out: list[WeightInfo] = []

    for s3_path, source_paths in paths:
        try:
            target_level, native_levels, group_weights, reps = build_group_weights(
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
                    # A regional domain leaves most global cells unmapped.
                    ignore_unmapped=True,
                ),
            )
        except Exception as exc:
            print(f"SKIP dataset {s3_path}: weight generation failed: {exc!r}")
            continue

        safe = s3_path.replace("/", "__")
        group_coverage: dict[str, str] = {}
        try:
            for key, rep in reps.items():
                ones = make_ones_dataset(xr.open_dataset(rep))
                coverage = gd.regrid_to_healpix(
                    ones,
                    target_level,
                    method="conservative",
                    weights_path=group_weights[key],
                    ignore_unmapped=True,
                )[COVERAGE_VAR].fillna(0.0)
                coverage.attrs.update(
                    long_name=(
                        "fraction of HEALPix cell covered by the source domain"
                    ),
                    units="1",
                )
                cov_path = coverage_dir / f"{safe}__{key}.nc"
                coverage.to_dataset(name=COVERAGE_VAR).to_netcdf(cov_path)
                group_coverage[key] = str(cov_path)
        except Exception as exc:
            print(f"SKIP dataset {s3_path}: coverage computation failed: {exc!r}")
            continue

        for key, weight_file in group_weights.items():
            print(
                f"{s3_path} grid={key} native_level={native_levels[key]} "
                f"target_level={target_level} weights={weight_file}"
            )
        out.append(
            {
                "s3_path": s3_path,
                "max_level": target_level,
                "group_weights": group_weights,
                "group_coverage": group_coverage,
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
    """Flatten into one work item per file, attaching weight + coverage."""
    lookup: dict[str, WeightInfo] = {w["s3_path"]: w for w in weights}

    staging = run_dir / "staging"
    items: list[RegridItem] = []

    for s3_path, source_files in sources:
        info = lookup.get(s3_path)
        if info is None:
            print(f"SKIP dataset {s3_path}: no weights (incomplete)")
            continue
        safe_dir = s3_path.replace("/", "__")
        out_dir = str(staging / safe_dir)

        for idx, src in enumerate(sorted(source_files)):
            key = cordex_group_key(src)
            weight_file = info["group_weights"].get(key)
            coverage_file = info["group_coverage"].get(key)
            if weight_file is None or coverage_file is None:
                raise KeyError(
                    f"No weight/coverage for grid group {key!r} of {src}; "
                    f"available: {sorted(info['group_weights'])}"
                )
            items.append(
                {
                    "s3_path": s3_path,
                    "source_file": src,
                    "weight_file": weight_file,
                    "coverage_file": coverage_file,
                    "max_level": info["max_level"],
                    "output_dir": out_dir,
                    "file_index": idx,
                }
            )

    print(f"Planned {len(items)} regrid tasks across {len(sources)} datasets")
    return items


# ---------------------------------------------------------------------------
# Step 4: Regrid one file → mask at finest level → coarsen → per-level NetCDF
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
    coverage_threshold: Annotated[
        float,
        Param(help="Cells with domain coverage below this become NaN"),
    ] = 0.5,
) -> RegridResult:
    """Regrid one source file, mask boundary cells, write pyramid levels.

    Masking happens at the *finest* level, before coarsening: the
    default renormalize policy extrapolates cells that barely overlap
    the domain to full-cell values, and coarsening first would fold that
    boundary junk into parent means. ``coverage_fraction`` is attached
    once per dataset (file_index 0) and coarsens by mean, which is
    exactly the parent's coverage.
    """
    import numpy as np
    import xarray as xr

    src = item["source_file"]
    max_level = item["max_level"]
    out_dir = Path(item["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    file_idx = item["file_index"]

    print(f"Regridding {src}")
    ds = xr.open_dataset(src)
    coverage = xr.open_dataset(item["coverage_file"])[COVERAGE_VAR].load()

    finest = gd.regrid_to_healpix(
        ds,
        max_level,
        method="conservative",
        weights_path=item["weight_file"],
        ignore_unmapped=True,
    )
    finest = _mask_with_coverage(finest, coverage, coverage_threshold)
    if file_idx == 0:
        finest[COVERAGE_VAR] = coverage.astype(np.float32)

    pyramid: dict[int, xr.Dataset] = {max_level: finest}
    for lvl in range(max_level - 1, -1, -1):
        pyramid[lvl] = gd.coarsen_healpix(pyramid[lvl + 1], lvl)

    for lvl, level_ds in pyramid.items():
        nc_path = out_dir / f"file_{file_idx:05d}_level_{lvl}.nc"
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
    uri: Annotated[
        str, Param(help="Target S3 bucket or path to disk")
    ] = "s3://regional",
    s3_endpoint: Annotated[
        str, Param(help="S3 endpoint URL")
    ] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[
        Path, Param(help="Path to S3 credentials JSON")
    ] = Path.home() / ".s3-credentials.json",
    coverage_threshold: Annotated[
        float, Param(help="Threshold used for the bounding-box attributes")
    ] = 0.5,
) -> None:
    """Combine levels, apply regional chunking + NaN fill, upload."""
    from glob import glob

    import numpy as np

    s3_path = group["s3_path"]
    out_dir = group["output_dir"]
    max_level = group["max_level"]

    print(f"Combining and uploading {s3_path}")

    pyramid: dict[int, xr.Dataset] = {}
    bbox: dict[str, float] | None = None
    for level in range(max_level + 1):
        nc_files = sorted(glob(f"{out_dir}/*_level_{level}.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No staging files for {s3_path} level {level} in {out_dir}"
            )
        ds = _open_level(nc_files)

        for name, v in ds.variables.items():
            v.encoding.pop("chunks", None)
            v.encoding.pop("preferred_chunks", None)
            # NaN fill so all-empty chunks outside the domain are elided.
            if name in ds.data_vars and np.issubdtype(v.dtype, np.floating):
                v.encoding["_FillValue"] = np.nan

        if level == max_level and COVERAGE_VAR in ds:
            bbox = domain_bbox(
                ds["latitude"].values,
                ds["longitude"].values,
                (ds[COVERAGE_VAR].values >= coverage_threshold),
            )

        pyramid[level] = ds.chunk(_regional_chunk_plan(ds))

    if bbox is not None:
        for level_ds in pyramid.values():
            level_ds.attrs.update(bbox)

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
