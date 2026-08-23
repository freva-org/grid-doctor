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
gather_sources → create_weights → scan_staging → regrid_batch → group_for_upload → combine_and_upload
  (singleton)     (singleton)      (singleton)     (array)        (singleton)         (array)

Throughput
----------
One array task is a *batch* of source files, not a single file: with
~1 h per file and an 8 h wall-time, one task runs ``workers`` files
concurrently on a node for as many rounds as fit.  ``scan_staging``
classifies every planned item against the staging directory first, so
work already on disk from an earlier run is never redone and a
cancelled run resumes at item granularity.
"""

from __future__ import annotations

import json
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

logger = gd.log.logging.getLogger(__name__)
wf = Workflow("cordex_healpix")

TARGET_CHUNK_BYTES = 16 * 1024**2  # 16 MiB
CELL_CHUNK_EXPONENT = 8  # 4**8 = 65536 cells per spatial chunk
COVERAGE_VAR = "coverage_fraction"

_CELL_DIM_CANDIDATES = ("cell", "cells", "value", "values", "pix", "ipix")

# --- Batching / resume -----------------------------------------------------
# DEFAULT_TIME_CHUNK is the *part* boundary: staging files are named
# ``..._part_PPP_...`` where PPP indexes a slice of this many time steps.
# Changing it mid-conversion re-slices the time axis, so old and new parts
# would overlap and the combine would concatenate duplicated timestamps.
# Treat it as frozen for the lifetime of a staging directory.
DEFAULT_TIME_CHUNK = 120
DEFAULT_ITEMS_PER_BATCH = 56  # 8 workers x 7 rounds at ~1 h/item within 8 h
DEFAULT_WORKERS = 8  # concurrent items per node; see sizing note in scan log
DONE_SUFFIX = ".done"  # per-item completion marker written by _regrid_one
EXPECTED_FILE = "expected.json"  # per-dataset item count, written by the scan


class WeightInfo(TypedDict):
    """One entry per dataset emitted by ``create_weights``."""

    s3_path: str
    max_level: int
    group_weights: dict[str, str]  # grid-group key -> weight file path
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
        if ds[name].ndim == 0
        and str(name) != "crs"
        and "grid_mapping_name" in ds[name].attrs
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

    Masks **in place** on the numpy buffers: ``DataArray.where`` would
    promote to float64 and allocate a full copy — 137 GiB for one year
    of daily data at level 11 — while the in-place assignment touches
    only the masked cells. Applied per variable (never
    ``Dataset.where``) so side-cars without the cell dimension are not
    broadcast onto the grid.
    """
    import numpy as np

    cell = _spatial_dim(ds)
    if cell is None:
        return ds
    invalid = ~(np.asarray(coverage.values) >= threshold)
    for name in ds.data_vars:
        var = ds[name]
        if cell in var.dims and np.issubdtype(var.dtype, np.floating):
            var = var.transpose(..., cell)
            var.values[..., invalid] = np.nan
    return ds


# ---------------------------------------------------------------------------
# Staging bookkeeping: what is already on disk?
# ---------------------------------------------------------------------------
# An item is complete when every ``part_PPP_level_L.nc`` exists for every
# part and level.  Three tiers, cheapest first:
#
#   1. a ``file_XXXXX.done`` marker matching this run's time_chunk/max_level
#      -> complete, no I/O beyond the directory listing;
#   2. part files present but no marker (the legacy items from a run that
#      predates markers) -> verify by header read, then write the marker so
#      tier 2 is paid exactly once;
#   3. nothing present -> to do.
#
# New writes go through ``_atomic_to_netcdf``, so from this run onwards a
# file bearing its final name is by construction complete and tier 2 shrinks
# to nothing.
def _marker_path(out_dir: Path, file_index: int) -> Path:
    return out_dir / f"file_{file_index:05d}{DONE_SUFFIX}"


def _part_prefix(file_index: int) -> str:
    return f"file_{file_index:05d}_part_"


def _expected_parts(source_file: str, time_chunk: int) -> int:
    """Number of time slices this source file is cut into.

    Header read only — no decoding, no data touched.
    """
    import xarray as xr

    with xr.open_dataset(source_file, decode_times=False, decode_cf=False) as ds:
        n_time = int(ds.sizes.get("time", 0))
    return max(1, -(-n_time // time_chunk)) if n_time else 1


def _marker_ok(item: RegridItem, names: set[str], time_chunk: int) -> bool:
    """Tier 1: trust a marker only if it describes *this* part layout."""
    name = f"file_{item['file_index']:05d}{DONE_SUFFIX}"
    if name not in names:
        return False
    try:
        meta = json.loads((Path(item["output_dir"]) / name).read_text())
    except Exception:
        return False
    return (
        meta.get("time_chunk") == time_chunk
        and meta.get("max_level") == item["max_level"]
    )


def _verify_and_mark(item: RegridItem, time_chunk: int) -> bool:
    """Tier 2: prove an unmarked item is complete, then mark it.

    Write order in ``_regrid_one`` is parts outer, levels max->0 inner, so
    ``part_PPP_level_0.nc`` is the last file of its part and the final part's
    level 0 is the last file overall — the only one that can be torn by a
    ``scancel``.  Existence covers the rest; the last one is opened.
    """
    import xarray as xr

    out_dir = Path(item["output_dir"])
    idx, max_level = item["file_index"], item["max_level"]
    try:
        n_parts = _expected_parts(item["source_file"], time_chunk)
    except Exception as exc:
        print(f"  cannot read source header for {item['source_file']}: {exc!r}")
        return False

    last: Path | None = None
    for part in range(n_parts):
        for lvl in range(max_level, -1, -1):
            f = out_dir / f"file_{idx:05d}_part_{part:03d}_level_{lvl}.nc"
            try:
                if f.stat().st_size == 0:
                    return False
            except OSError:
                return False
            last = f
    if last is not None:
        try:
            with xr.open_dataset(last):
                pass
        except Exception:
            return False

    _write_marker(out_dir, idx, n_parts, time_chunk, max_level)
    return True


def _write_marker(
    out_dir: Path, file_index: int, parts: int, time_chunk: int, max_level: int
) -> None:
    _marker_path(out_dir, file_index).write_text(
        json.dumps(
            {
                "parts": parts,
                "time_chunk": time_chunk,
                "max_level": max_level,
            }
        )
    )


def _purge_partial(out_dir: Path, file_index: int) -> None:
    """Drop any staging debris for this item before rewriting it.

    A re-run under a different part layout must not leave orphaned
    ``part_003`` files behind for the combine's glob to pick up.
    """
    for stale in out_dir.glob(f"{_part_prefix(file_index)}*"):
        stale.unlink(missing_ok=True)
    _marker_path(out_dir, file_index).unlink(missing_ok=True)


def _atomic_to_netcdf(ds: xr.Dataset, path: Path, **kwargs: object) -> None:
    """Write via a temp name and rename, so a killed job leaves no torn file.

    ``os.replace`` is atomic within a directory on Lustre, so a staging file
    that exists under its final name is complete by construction.
    """
    import os

    tmp = path.with_name(path.name + ".tmp")
    ds.to_netcdf(tmp, **kwargs)  # type: ignore[arg-type]
    os.replace(tmp, path)


def _balanced_batches(
    items: list[RegridItem], size: int
) -> list[list[RegridItem]]:
    """Greedy longest-first bin packing into batches of roughly equal cost.

    A batch finishes when its slowest round does, so mixing decadal files
    with single-year ones into arbitrary batches wastes wall-time.  Source
    file size is a free and good-enough proxy for regrid cost.  Ties break
    toward the emptier bin, which keeps items of one dataset — and hence one
    weight file — loosely together.
    """
    if not items:
        return []
    n_bins = max(1, -(-len(items) // size))

    def cost(item: RegridItem) -> int:
        try:
            return Path(item["source_file"]).stat().st_size
        except OSError:
            return 0

    bins: list[list[RegridItem]] = [[] for _ in range(n_bins)]
    load = [0] * n_bins
    for item in sorted(items, key=cost, reverse=True):
        j = min(range(n_bins), key=lambda k: (load[k], len(bins[k])))
        bins[j].append(item)
        load[j] += cost(item)
    for b in bins:
        b.sort(key=lambda i: (i["s3_path"], i["weight_file"], i["file_index"]))
    return [b for b in bins if b]


# ---------------------------------------------------------------------------
# Step 1: Discover source files via the Freva databrowser
# ---------------------------------------------------------------------------
@wf.job(cpus=2, time="00:10:00", mem="1GB", partition="shared", version="2")
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

    if not entries:
        # An empty matrix would let the whole run "succeed" doing nothing
        # (the array fan-out gets zero elements). Fail loudly instead, and
        # show what the databrowser actually offers for this project so
        # the facet mismatch is visible in the task log.
        diagnostics: dict[str, list[str]] = {}
        base = {"project": project, **({"product": product} if product else {})}
        for facet in (
            "experiment",
            "time_frequency",
            "model",
            "driving_model",
            "ensemble",
            "variable",
        ):
            try:
                diagnostics[facet] = client.facet_values(facet, **base)[:25]
            except Exception as exc:  # diagnostics must never mask the error
                diagnostics[facet] = [f"<query failed: {exc!r}>"]
        lines = "\n".join(f"  {k}: {v}" for k, v in diagnostics.items())
        raise RuntimeError(
            f"Search matched no datasets for project={project!r} "
            f"product={product!r} experiments={experiment or 'ALL'} "
            f"freq={freq}. Available facet values (constrained only by "
            f"project/product):\n{lines}"
        )

    return [(entry.key, entry.files) for entry in entries]


# ---------------------------------------------------------------------------
# Step 2: Create / cache ESMF weight files + per-grid coverage fields
# ---------------------------------------------------------------------------
@wf.job(cpus=128, time="08:00:00", partition="compute", mem="0", version="3")
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
            logger.error(
                f"SKIP dataset {s3_path}: weight generation failed", exc_info=exc
            )
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
                    # "propagate" returns the raw weight-row sums, which
                    # for a ones-field ARE the coverage fractions. The
                    # default "renormalize" divides by the weight sum and
                    # would turn every touched cell into exactly 1.0,
                    # making the mask a no-op.
                    missing_policy="propagate",
                )[COVERAGE_VAR].fillna(0.0)
                coverage.attrs.update(
                    long_name=("fraction of HEALPix cell covered by the source domain"),
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
    if paths and not out:
        raise RuntimeError(
            f"Weight/coverage generation failed for all {len(paths)} "
            "datasets (see the SKIP lines above for the individual "
            "reasons; a missing ESMF/esmpy in this partition's environment "
            "fails every dataset the same way)."
        )
    return out


# ---------------------------------------------------------------------------
# Step 3: Explode into per-file work items, drop what is staged, batch the rest
# ---------------------------------------------------------------------------
def _plan_items(
    sources: list[tuple[str, list[str]]],
    weights: list[WeightInfo],
    run_dir: Path,
) -> list[RegridItem]:
    """Flatten into one work item per file, attaching weight + coverage.

    Pure: depends only on its arguments, never on the filesystem.  The
    ``file_index`` assigned here is what the staging filenames are keyed on,
    so ``sorted(source_files)`` must stay stable across runs — it is the
    contract that lets a cancelled run resume.
    """
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
    return items


@wf.job(cpus=8, time="00:30:00", mem="8GB", partition="shared", version="1")
def scan_staging(
    sources: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    weights: Annotated[list[WeightInfo], Result(step="create_weights")],
    run_dir: RunDir,
    time_chunk: Annotated[
        int,
        Param(help="Time steps per staging part; FROZEN per staging directory"),
    ] = DEFAULT_TIME_CHUNK,
    items_per_batch: Annotated[
        int, Param(help="Source files per array task")
    ] = DEFAULT_ITEMS_PER_BATCH,
) -> list[list[RegridItem]]:
    """Classify every planned item against staging; batch only the outstanding.

    This is the one step whose result depends on the filesystem rather than
    on its declared inputs, so it must never be served from the Merkle
    cache — a replayed scan would re-dispatch work that is already on disk.
    ``version`` is bumped by hand whenever the classification logic changes;
    if reflow grows a per-step ``cache=False``, use that instead.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = _plan_items(sources, weights, run_dir)
    print(f"Planned {len(items)} regrid items across {len(sources)} datasets")
    if sources and not items:
        raise RuntimeError(
            "Planned zero regrid tasks although sources exist — every "
            "dataset was dropped for missing weights. Fix create_weights "
            "and retry."
        )

    # One directory listing per dataset, not one stat per item.
    listings: dict[str, set[str]] = {}
    for item in items:
        d = item["output_dir"]
        if d not in listings:
            p = Path(d)
            listings[d] = {f.name for f in p.iterdir()} if p.is_dir() else set()

    fast_done: list[RegridItem] = []
    ambiguous: list[RegridItem] = []
    todo: list[RegridItem] = []
    for item in items:
        names = listings[item["output_dir"]]
        if _marker_ok(item, names, time_chunk):
            fast_done.append(item)
        elif any(n.startswith(_part_prefix(item["file_index"])) for n in names):
            ambiguous.append(item)
        else:
            todo.append(item)

    # Metadata-bound rather than CPU-bound: threads, and Lustre rewards the
    # concurrency. This tier is empty on every run after the first.
    verified: list[bool] = []
    if ambiguous:
        print(f"Verifying {len(ambiguous)} unmarked items by header read...")
        with ThreadPoolExecutor(max_workers=32) as pool:
            verified = list(
                pool.map(lambda it: _verify_and_mark(it, time_chunk), ambiguous)
            )
        todo += [it for it, ok in zip(ambiguous, verified) if not ok]

    n_done = len(fast_done) + sum(verified)

    # The combine needs to know how many items a complete dataset has, so it
    # can refuse to publish a store with silent time gaps.
    per_dataset: dict[str, tuple[str, int]] = {}
    for item in items:
        s3_path, out_dir = item["s3_path"], item["output_dir"]
        _, count = per_dataset.get(s3_path, (out_dir, 0))
        per_dataset[s3_path] = (out_dir, count + 1)
    for s3_path, (out_dir, count) in per_dataset.items():
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / EXPECTED_FILE).write_text(
            json.dumps({"s3_path": s3_path, "items": count})
        )

    batches = _balanced_batches(todo, items_per_batch)
    print(
        f"scan: {n_done} items already staged, {len(todo)} outstanding, "
        f"{len(batches)} batches of <= {items_per_batch} "
        f"({len(ambiguous)} verified by header read)"
    )
    levels = sorted({w["max_level"] for w in weights})
    print(
        f"max_level in play: {levels} — size --workers against "
        f"12*4**level*8 bytes * time_chunk peak per item"
    )
    return batches


# ---------------------------------------------------------------------------
# Node budget: memory and wall-clock
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    """Print with an explicit flush and a timestamp.

    Slurm block-buffers a redirected stdout, so an un-flushed print is lost
    entirely when the step is SIGKILLed at the wall — which is how a job
    that ran for eight hours came back with a single line of output and no
    indication of where it had got to.
    """
    import sys
    import time as _time

    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stderr.flush()


def _mem_used_bytes() -> int:
    """Current cgroup memory usage, or 0 when unreadable."""
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return 0


def _node_mem_bytes() -> int:
    """Memory this step may actually use, from the cgroup Slurm put us in.

    ``mem=0`` asks for the whole node, so the cgroup limit is the node's
    RAM; reading it rather than hard-coding 256 GB keeps the sizing honest
    on a partition with different hardware.
    """
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw and raw != "max":
            value = int(raw)
            # v1 reports a sentinel near 2**63 when unlimited.
            if 0 < value < (1 << 62):
                return value
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 256 * 1024**3


def _peak_bytes_per_item(max_level: int, time_chunk: int) -> int:
    """Rough peak RSS of one concurrent ``_regrid_one``.

    The regrid returns float64 over ``12 * 4**level`` cells for
    ``time_chunk`` steps; the per-variable float32 cast transiently holds
    source and destination, so the data term is ~1.5x the float64 array.
    On top of that each worker holds its *own* copy of the sparse
    conservative operator — batches share a weight *file*, but not the
    in-memory matrix — which is the term that made 8 workers OOM.
    """
    cells = 12 * (4**int(max_level))
    data = cells * int(time_chunk) * 8  # float64 regrid output
    data = int(data * 1.5)  # + float32 copy during the cast
    weights = max(2 * 1024**3, cells * 40)  # sparse operator, per process
    return data + weights


def _auto_workers(
    batch: list[RegridItem], time_chunk: int, requested: int, reserve_bytes: int
) -> int:
    """Clamp ``requested`` to what the node's memory actually allows.

    Sizing is done against the *deepest* level in the batch: one OOM kill
    breaks the whole ``ProcessPoolExecutor``, so the safe worker count is
    set by the worst item, not the average.
    """
    max_level = max(int(i["max_level"]) for i in batch)
    per_item = _peak_bytes_per_item(max_level, time_chunk)
    budget = max(0, _node_mem_bytes() - reserve_bytes)
    fits = max(1, budget // per_item)
    chosen = max(1, min(int(requested), int(fits)))
    _log(
        f"memory budget: {budget / 1024**3:.0f} GiB usable, "
        f"~{per_item / 1024**3:.1f} GiB/item at level {max_level} "
        f"(time_chunk={time_chunk}) -> {fits} fit, "
        f"{requested} requested, using {chosen}"
    )
    return chosen


def _deadline(safety_seconds: float) -> float:
    """Epoch time after which no new item may be started.

    Slurm exports ``SLURM_JOB_END_TIME`` (epoch seconds); without it fall
    back to the step's own start plus a conservative 8 h.  Starting an item
    that cannot finish wastes an hour of node time and leaves debris the
    next run has to purge, so the scheduler stops feeding the pool once the
    remaining time is under one item's worth.
    """
    import os
    import time

    raw = os.environ.get("SLURM_JOB_END_TIME", "")
    try:
        end = float(raw)
    except ValueError:
        end = time.time() + 8 * 3600.0
    return end - safety_seconds


# ---------------------------------------------------------------------------
# Step 4: Regrid one file → mask at finest level → coarsen → per-level NetCDF
# ---------------------------------------------------------------------------
@wf.array_job(
    cpus=128,
    time="08:00:00",
    mem="0",
    partition="compute",
    array_parallelism=0,
    version="5",
)
def regrid_batch(
    batch: Annotated[list[RegridItem], Result(step="scan_staging")],
    coverage_threshold: Annotated[
        float,
        Param(help="Cells with domain coverage below this become NaN"),
    ] = 0.5,
    time_chunk: Annotated[
        int,
        Param(help="Time steps per staging part; FROZEN per staging directory"),
    ] = DEFAULT_TIME_CHUNK,
    workers: Annotated[
        int, Param(help="Max concurrent items; clamped by node memory")
    ] = DEFAULT_WORKERS,
    reserve_gb: Annotated[
        float, Param(help="Node memory held back from the worker budget")
    ] = 48.0,
    item_minutes: Annotated[
        int, Param(help="Assumed per-item runtime for the deadline guard")
    ] = 90,
    recycle_after: Annotated[
        int, Param(help="Fork fresh workers after this many rounds")
    ] = 1,
) -> list[RegridResult]:
    """Regrid a batch of source files, N at a time, on one node.

    One array task is a batch rather than a single file: the cluster grants
    only a handful of concurrent tasks, so throughput is work-per-task times
    tasks-in-flight and the first factor is the one under our control.

    Three failure modes are handled explicitly, all of them observed:

    - **OOM.** ``workers`` is clamped to what the cgroup can hold, sized
      against the deepest level in the batch. A single OOM kill breaks the
      whole executor, so the count must be safe for the worst item.
    - **A broken pool.** If a worker dies anyway, ``BrokenProcessPool``
      fails every in-flight *and* every queued future at once. Rather than
      logging 50 spurious failures, the remaining items are requeued on a
      fresh, smaller pool.
    - **The wall.** No item is started that cannot finish before
      ``SLURM_JOB_END_TIME``; the leftovers are simply not attempted and
      the next run's scan picks them up. Work already staged always
      survives, because staging writes are atomic and marked.
    """
    import time
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    done: list[RegridResult] = []
    pending: list[RegridItem] = []
    for item in batch:
        if _already_staged(item, time_chunk):
            done.append(
                {
                    "s3_path": item["s3_path"],
                    "output_dir": item["output_dir"],
                    "max_level": item["max_level"],
                }
            )
        else:
            pending.append(item)
    if done:
        _log(f"Skipping {len(done)} already-staged items in this batch")
    if not pending:
        _log(f"batch complete: {len(done)} staged, 0 failed, 0 deferred")
        return done

    n_workers = _auto_workers(
        pending, time_chunk, workers, int(reserve_gb * 1024**3)
    )
    # deadline: stop *starting* items that cannot finish.
    # hard_stop: stop *waiting* and exit cleanly, so the step returns its
    # results instead of being SIGKILLed at the wall with an unflushed
    # buffer and no record of what completed.
    deadline = _deadline(safety_seconds=item_minutes * 60.0)
    hard_stop = _deadline(safety_seconds=300.0)
    failures: list[str] = []
    abandoned = False
    _log(
        f"{len(pending)} item(s) to run, {n_workers} worker(s), "
        f"{(deadline - time.time()) / 3600:.1f} h of submit budget, "
        f"hard stop in {(hard_stop - time.time()) / 3600:.1f} h"
    )

    while pending and time.time() < deadline and not abandoned:
        queue = list(pending)
        pending = []
        broken = False
        recycling = False
        retired_here = 0
        try:
            pool = _pool(n_workers)
            try:
                futures: dict[object, RegridItem] = {}
                while queue or futures:
                    while queue and len(futures) < n_workers and not recycling:
                        if time.time() >= deadline:
                            break
                        item = queue.pop(0)
                        fut = pool.submit(
                            _regrid_one,
                            item,
                            coverage_threshold,
                            time_chunk,
                            max(1, 128 // n_workers),
                        )
                        futures[fut] = item
                    if not futures:
                        break
                    finished, _ = wait(
                        list(futures), timeout=60, return_when=FIRST_COMPLETED
                    )
                    if not finished:
                        # Nothing completed in the last minute: report where
                        # we are, so a stalled batch is visible in the log
                        # rather than being inferred from silence.
                        _log(
                            f"waiting on {len(futures)} item(s); "
                            f"cgroup at {_mem_used_bytes() / 1024**3:.0f} GiB; "
                            f"{(deadline - time.time()) / 60:.0f} min of "
                            "budget left"
                        )
                    if time.time() >= hard_stop:
                        # The wall is close. Abandon in-flight work rather
                        # than letting Slurm SIGKILL the step: staged files
                        # are atomic and marked, so whatever finished is
                        # kept and the next scan resumes from there.
                        _log(
                            f"hard stop: abandoning {len(futures)} in-flight "
                            f"and {len(queue)} queued item(s)"
                        )
                        queue.extend(futures.values())
                        futures.clear()
                        abandoned = True
                        break
                    for fut in finished:
                        item = futures.pop(fut)
                        try:
                            done.append(fut.result())
                            retired_here += 1
                        except BrokenProcessPool:
                            # The pool is gone; nothing queued behind this
                            # will run either. Salvage and restart smaller.
                            queue.append(item)
                            broken = True
                        except Exception as exc:
                            failures.append(item["source_file"])
                            _log(f"FAILED {item['source_file']}: {exc!r}")
                            retired_here += 1
                    if broken:
                        queue.extend(futures.values())
                        futures.clear()
                        break
                    # Stop feeding this pool once it has done a full sweep;
                    # let it drain, then the outer loop forks fresh workers.
                    if queue and retired_here >= recycle_after * n_workers:
                        recycling = True
            finally:
                _shutdown(pool, graceful=not abandoned)
        except BrokenProcessPool:
            broken = True

        if abandoned:
            # The `with` block already called shutdown(wait=True), which is
            # what we want: workers get to finish the NetCDF they are on,
            # and the atomic rename either lands or leaves a .tmp.
            pending = queue
            break

        pending = queue
        if not broken:
            if not pending:
                break
            _log(f"Recycling workers; {len(pending)} items left in this batch")
            continue
        if n_workers == 1:
            _log("Pool broke at 1 worker — memory model is wrong; deferring rest")
            break
        n_workers = max(1, n_workers // 2)
        _log(
            f"Worker died (OOM?); {len(pending)} items requeued on a pool of "
            f"{n_workers}. Lower --workers or raise --reserve-gb for the rerun."
        )

    if pending:
        _log(
            f"Deferring {len(pending)} items: not enough wall-time left to "
            "start another. The next run's scan will pick them up."
        )
    _log(
        f"batch complete: {len(done)} staged, {len(failures)} failed, "
        f"{len(pending)} deferred"
        + (f" (failures: {failures})" if failures else "")
    )
    return done


def _shutdown(pool, graceful: bool, grace_seconds: float = 120.0) -> None:
    """Tear down a pool without ever blocking indefinitely.

    ``ProcessPoolExecutor.__exit__`` calls ``shutdown(wait=True)``, which
    waits forever on a worker that is stuck — so a batch that correctly
    detected the wall approaching would still be SIGKILLed while waiting to
    tidy up. Here workers get ``grace_seconds`` to land their current
    atomic rename, then are terminated.
    """
    import time

    # shutdown() sets _processes to None, so snapshot the children first.
    children = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    if graceful:
        end = time.time() + grace_seconds
        while time.time() < end and any(p.is_alive() for p in children):
            time.sleep(1.0)
    for proc in children:
        if proc.is_alive():
            proc.terminate()
    for proc in children:
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.kill()


def _pool(n_workers: int):
    """A plain fork-context pool.

    ``max_tasks_per_child=1`` would be the obvious way to hand each item's
    peak RSS back to the OS, but CPython rejects it with the *fork* start
    method, and the alternatives (spawn, forkserver) both re-import
    ``__main__`` — which here is a reflow-dispatched workflow script, so
    importing it re-enters the CLI. The scheduler recycles the pool itself
    instead; see ``recycle_after``.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("fork"),
    )


def _already_staged(item: RegridItem, time_chunk: int) -> bool:
    """Marker-or-verify check, for use inside the worker node."""
    out_dir = Path(item["output_dir"])
    if not out_dir.is_dir():
        return False
    names = {f.name for f in out_dir.iterdir()}
    if _marker_ok(item, names, time_chunk):
        return True
    if not any(n.startswith(_part_prefix(item["file_index"])) for n in names):
        return False
    return _verify_and_mark(item, time_chunk)


def _regrid_one(
    item: RegridItem,
    coverage_threshold: float,
    time_chunk: int,
    threads: int = 0,
) -> RegridResult:
    """Regrid one source file, mask boundary cells, write pyramid levels.

    Memory discipline (a year of daily data at level 11 is ~74 GB in
    float32 for the finest level alone):

    - the file is processed in ``time_chunk``-step slices, each written
      as ``file_XXXXX_part_PPP_level_L.nc`` (the combine step's sorted
      glob concatenates parts transparently);
    - regrid output is cast to float32 immediately (the sparse matmul
      returns float64 — double the footprint for no benefit after the
      conservative average has been taken);
    - masking is in place (see ``_mask_with_coverage``);
    - the pyramid is streamed: write a level, coarsen to the next, drop
      the previous — never all levels in memory at once;
    - staging NetCDFs are zlib-compressed: cells outside the domain are
      constant NaN and compress by orders of magnitude, keeping the
      staging volume proportional to the domain instead of the globe.

    Masking happens at the *finest* level, before coarsening, so the
    renormalize policy's boundary extrapolation never contaminates
    parent cells. ``coverage_fraction`` is attached once per dataset
    (file_index 0, first slice) and coarsens by mean, which is exactly
    the parent's coverage.
    """
    import numpy as np
    import xarray as xr

    # Set in the child, not the parent: grid_doctor is imported at module
    # scope so OpenMP is already initialised, and every worker would
    # otherwise grab all 128 cores and thrash. threadpoolctl retargets the
    # live pools; OMP_NUM_THREADS at this point would be a no-op.
    if threads > 0:
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(limits=threads)
        except ImportError:
            pass

    src = item["source_file"]
    max_level = item["max_level"]
    out_dir = Path(item["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    file_idx = item["file_index"]

    # Any debris from an interrupted earlier attempt must go, or the
    # combine's glob would pick up orphaned parts alongside the new ones.
    _purge_partial(out_dir, file_idx)

    print(f"Regridding {src}")
    ds = xr.open_dataset(src)
    coverage = xr.open_dataset(item["coverage_file"])[COVERAGE_VAR].load()

    n_time = int(ds.sizes.get("time", 0))
    slices = (
        [slice(a, min(a + time_chunk, n_time)) for a in range(0, n_time, time_chunk)]
        if n_time
        else [slice(None)]
    )

    def _encoding(level_ds: xr.Dataset) -> dict[str, dict[str, object]]:
        cell = _spatial_dim(level_ds)
        return {
            str(v): {"zlib": True, "complevel": 1}
            for v in level_ds.data_vars
            if cell is not None and cell in level_ds[str(v)].dims
        }

    # Grid-geometry side-cars (cell bounds, grid-mapping scalars) must not
    # enter the regrid: conservative remapping of lat/lon_vertices onto
    # 50M cells is ~1.6 GB of meaningless output per part, and the
    # rotated-pole mapping variable is obsolete on the HEALPix grid.
    geometry_vars = [
        str(name)
        for name in ds.data_vars
        if {"vertices", "nv", "nvertex", "bnds"} & {str(d) for d in ds[name].dims}
        and str(name) != "time_bnds"
        or "grid_mapping_name" in ds[name].attrs
    ]
    if geometry_vars:
        print(f"Dropping grid-geometry side-cars: {sorted(geometry_vars)}")
        ds = ds.drop_vars(geometry_vars)

    for part, tsel in enumerate(slices):
        chunk_ds = ds.isel(time=tsel) if n_time else ds
        finest = gd.regrid_to_healpix(
            chunk_ds,
            max_level,
            method="conservative",
            weights_path=item["weight_file"],
            ignore_unmapped=True,
        )
        cell = _spatial_dim(finest)
        # Per-variable cast: Dataset.astype with a dict demands an entry
        # for EVERY data variable and raises 'exact match required'
        # otherwise; casting one variable at a time also frees each
        # float64 buffer before the next is converted.
        for v in list(map(str, finest.data_vars)):
            if (
                cell is not None
                and cell in finest[v].dims
                and finest[v].dtype == np.float64
            ):
                attrs = dict(finest[v].attrs)
                finest[v] = finest[v].astype(np.float32)
                finest[v].attrs = attrs
        finest = _mask_with_coverage(finest, coverage, coverage_threshold)
        if file_idx == 0 and part == 0:
            finest[COVERAGE_VAR] = coverage.astype(np.float32)

        current = finest
        for lvl in range(max_level, -1, -1):
            nc_path = out_dir / f"file_{file_idx:05d}_part_{part:03d}_level_{lvl}.nc"
            # Cell coordinates are a pure function of the index and cost
            # ~805 MB per part file at level 11; drop them from staging
            # and re-attach once in the combine step.
            slim = current.drop_vars(["latitude", "longitude", "cell"], errors="ignore")
            _atomic_to_netcdf(slim.load(), nc_path, encoding=_encoding(slim))
            del slim
            if lvl:
                coarser = gd.coarsen_healpix(current, lvl - 1)
                del current
                current = coarser
        del current, finest

    # Written last: its presence means every part and level landed.
    _write_marker(out_dir, file_idx, len(slices), time_chunk, max_level)

    return {
        "s3_path": item["s3_path"],
        "output_dir": str(out_dir),
        "max_level": max_level,
    }


# ---------------------------------------------------------------------------
# Step 5: Group per-file results back by dataset
# ---------------------------------------------------------------------------
@wf.job(cpus=1, time="00:10:00", mem="2GB", partition="shared", version="3")
def group_for_upload(
    weights: Annotated[list[WeightInfo], Result(step="create_weights")],
    results: Annotated[list[list[RegridResult]], Result(step="regrid_batch")],
    run_dir: RunDir,
) -> list[RegridResult]:
    """Enumerate every dataset that has weights, not just the ones re-run.

    The group list comes from ``create_weights``, because a dataset whose
    items were all staged by an earlier run produces no regrid output this
    time and would otherwise never be uploaded.  The ``results`` edge is
    kept purely for ordering — it makes the combine wait for the regrids.
    Completeness is enforced in ``combine_and_upload`` against the counts
    ``scan_staging`` recorded, not by counting what came back here.
    """
    staged = 0
    for chunk in results:
        staged += len(chunk) if isinstance(chunk, list) else 1
    print(f"regrid_batch returned {staged} staged items")

    staging = run_dir / "staging"
    out: list[RegridResult] = [
        {
            "s3_path": w["s3_path"],
            "output_dir": str(staging / w["s3_path"].replace("/", "__")),
            "max_level": w["max_level"],
        }
        for w in weights
    ]
    out.sort(key=lambda g: g["s3_path"])
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
    array_parallelism=0,
    version="2",
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

    # Datasets now reach this step whether or not anything was regridded for
    # them this run, so the "did every item land?" check has to be explicit.
    # Without it a dataset missing 3 of 40 source files sails through — the
    # per-level glob is non-empty — and publishes a store with silent time
    # gaps, which is far worse than a failed job.
    expected_path = Path(out_dir) / EXPECTED_FILE
    if not expected_path.exists():
        raise FileNotFoundError(
            f"{s3_path}: no {EXPECTED_FILE} in {out_dir}; re-run scan_staging "
            "so the expected item count is recorded before uploading."
        )
    expected = int(json.loads(expected_path.read_text())["items"])
    actual = len(list(Path(out_dir).glob(f"file_*{DONE_SUFFIX}")))
    if actual != expected:
        raise RuntimeError(
            f"{s3_path}: {actual}/{expected} items staged; refusing to upload "
            "an incomplete store. Re-run to fill the gaps (staged items are "
            "skipped), or delete the staging dir to start this dataset over."
        )

    print(f"Combining and uploading {s3_path} ({expected} items staged)")

    pyramid: dict[int, xr.Dataset] = {}
    bbox: dict[str, float] | None = None
    for level in range(max_level + 1):
        nc_files = sorted(glob(f"{out_dir}/*_level_{level}.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No staging files for {s3_path} level {level} in {out_dir}"
            )
        ds = _open_level(nc_files)

        # Staging files carry no cell coordinates (see regrid_file);
        # re-attach them for levels the store will materialise.
        if "latitude" not in ds.coords:
            try:
                from grid_doctor.helpers import WRITE_COORDS_MAX_LEVEL
            except ImportError:  # older grid-doctor
                WRITE_COORDS_MAX_LEVEL = 10
            if level <= WRITE_COORDS_MAX_LEVEL:
                from grid_doctor.remap import _attach_healpix_coords

                ds = _attach_healpix_coords(
                    ds,
                    level=level,
                    nest=True,
                    method=str(ds.attrs.get("grid_doctor_method", "conservative")),
                )

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
