"""Fan-out writes of a pyramid: manifest state, worker writes, completeness.

The init step
([`init_pyramid_store`][grid_doctor.init_pyramid_store])
creates empty level stores and a
[`PyramidManifest`][grid_doctor.PyramidManifest].  Workers call
[`write_pyramid_task`][grid_doctor.write_pyramid_task] for one
[`WriteTask`][grid_doctor.WriteTask] each, independently and in any
order.  [`check_pyramid_store`][grid_doctor.check_pyramid_store] then
tells whether every task has finished and the store still has the
layout the manifest describes.

State that is not part of the zarr data lives next to the level stores::

    <path>/
        level_9.zarr/ ... level_0.zarr/
        _grid_doctor/
            init.json              # only while an init runs (or was killed)
            manifest.json          # written last by the init step
            done/<task_id>.json    # one marker per finished task

Completion is tracked with markers rather than by looking at chunks,
because zarr does not store chunks that consist only of fill values:
for regional or masked data many chunks are legitimately absent.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import socket
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import dask.array as da
import numpy as np
import s3fs
import xarray as xr
import zarr

from ..types import (
    BoolArray,
    EncodingDict,
    GenericArray,
    IndexSlice,
    S3Options,
    ZarrArray,
    ZarrStore,
)
from ._manifest import (
    DONE_DIR,
    INIT_FILE,
    MANIFEST_FILE,
    SPATIAL_DIMS,
    STATE_DIR,
    ArraySpec,
    LevelSpec,
    ManifestError,
    PyramidManifest,
    WriteTask,
    plan_tasks,
)
from ._zarr import (
    DeferredWrite,
    encode_zarr_variable,
    get_array,
    store_all,
    stored_encodings,
)

logger = logging.getLogger(__name__)

#: What ``init_pyramid_store`` does when the target already exists.
IfExists = Literal["error", "resume", "overwrite"]


class StaleTaskError(ManifestError):
    """The task belongs to a different manifest than the one in the store."""


class StoreExistsError(ManifestError):
    """The init target already holds a pyramid (or the remains of one).

    Raised by ``init_pyramid_store(if_exists="error")``, and by
    ``if_exists="resume"`` when there is nothing it could resume.
    """


class IncompleteStoreError(RuntimeError):
    """Raised by [`StoreReport.raise_if_incomplete`][grid_doctor.StoreReport.raise_if_incomplete]."""


# ======================================================================
# Storage access
# ======================================================================


class _Target:
    """Pyramid root on local disk or S3, plus the state directory."""

    def __init__(self, path: str, s3_options: S3Options | None) -> None:
        self.path = path.rstrip("/")
        self.fs: s3fs.S3FileSystem | None = (
            s3fs.S3FileSystem(**dict(s3_options or {}))
            if self.path.startswith("s3://")
            else None
        )

    # -- zarr stores ---------------------------------------------------

    def store(self, name: str) -> ZarrStore:
        root = f"{self.path}/{name}"
        if self.fs is not None:
            store: ZarrStore = s3fs.S3Map(root=root, s3=self.fs)
            return store
        return root

    # -- plain files below the root -------------------------------------

    def _key(self, *parts: str) -> str:
        return "/".join((self.path, *parts))

    def read_text(self, *parts: str) -> str | None:
        key = self._key(*parts)
        try:
            if self.fs is not None:
                data: bytes = self.fs.cat_file(key)
                return data.decode()
            return Path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def write_text(self, text: str, *parts: str) -> None:
        key = self._key(*parts)
        if self.fs is not None:
            self.fs.pipe_file(key, text.encode())
            return
        target = Path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)  # atomic: readers never see half a marker

    def exists(self, *parts: str) -> bool:
        key = self._key(*parts)
        if self.fs is not None:
            self.fs.invalidate_cache(key)
            return bool(self.fs.exists(key))
        return Path(key).exists()

    def list_dir(self, *parts: str) -> list[str]:
        key = self._key(*parts)
        if self.fs is not None:
            # s3fs caches listings per (cached) filesystem instance; markers
            # are written by other processes, so always ask S3 again.
            self.fs.invalidate_cache(key)
            try:
                return [p.rsplit("/", 1)[-1] for p in self.fs.ls(key, detail=False)]
            except FileNotFoundError:
                return []
        directory = Path(key)
        return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []

    def remove(self, *parts: str, recursive: bool = False) -> None:
        key = self._key(*parts)
        if self.fs is not None:
            try:
                self.fs.rm(key, recursive=recursive)
            except FileNotFoundError:
                pass
            return
        target = Path(key)
        if target.is_dir():
            for child in target.iterdir():
                child.unlink()
            target.rmdir()
        elif target.exists():
            target.unlink()

    def find(self, *parts: str) -> set[str]:
        """All file keys below *parts*, relative to it, ``/``-separated."""
        key = self._key(*parts)
        if self.fs is not None:
            self.fs.invalidate_cache(key)
            prefix = self.fs._strip_protocol(key).rstrip("/") + "/"
            return {
                p.removeprefix(prefix)
                for p in self.fs.find(key)
                if p.startswith(prefix)
            }
        root = Path(key)
        if not root.is_dir():
            return set()
        found: set[str] = set()
        for dirpath, _, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            for name in filenames:
                found.add((rel / name).as_posix())
        return found


def load_manifest(
    path: str, s3_options: S3Options | None = None
) -> PyramidManifest:
    """Read the manifest that the init step stored below *path*.

    Parameters
    ----------
    path:
        Pyramid root, as passed to ``init_pyramid_store``.
    s3_options:
        Options for :class:`s3fs.S3FileSystem` (S3 targets only).

    Raises
    ------
    ManifestError
        If no manifest exists, e.g. because the init step has not
        finished or the store was not initialised with a manifest.
    """
    target = _Target(path, s3_options)
    text = target.read_text(STATE_DIR, MANIFEST_FILE)
    if text is None:
        interrupted = _interrupted_init(target)
        if interrupted:
            raise ManifestError(f"no manifest at {path}: {interrupted}")
        raise ManifestError(
            f"no manifest at {path}/{STATE_DIR}/{MANIFEST_FILE}: run "
            "init_pyramid_store first (or wait for it to finish)"
        )
    return PyramidManifest.from_json(text)


def is_current(
    item: PyramidManifest | WriteTask, s3_options: S3Options | None = None
) -> bool:
    """Whether the store at ``item.path`` still holds ``item``'s manifest.

    ``False`` if the store was deleted, never finished its init, or was
    re-initialised since *item* was created.  Meant as a cache check, for
    example reflow's ``verify=`` hook on the init step: a cached manifest
    is only reusable while the store it describes still exists.

    Parameters
    ----------
    item:
        A manifest or any of its tasks.
    s3_options:
        Options for :class:`s3fs.S3FileSystem` (S3 targets only).
    """
    text = _Target(item.path, s3_options).read_text(STATE_DIR, MANIFEST_FILE)
    if text is None:
        return False
    try:
        stored_id = json.loads(text).get("manifest_id")
    except json.JSONDecodeError:
        return False
    return bool(stored_id == item.manifest_id)


def is_task_done(task: WriteTask, s3_options: S3Options | None = None) -> bool:
    """Whether *task* has a completion marker for the current manifest.

    Equivalent to ``task.task_id in check_pyramid_store(...).done_tasks``
    but reads a single small file, so it is cheap enough to call per
    task, e.g. as reflow's ``verify=`` hook on the array job.

    Parameters
    ----------
    task:
        The work unit.
    s3_options:
        Options for :class:`s3fs.S3FileSystem` (S3 targets only).
    """
    if not is_current(task, s3_options):
        return False
    text = _Target(task.path, s3_options).read_text(
        STATE_DIR, DONE_DIR, f"{task.task_id}.json"
    )
    if text is None:
        return False
    try:
        marker = json.loads(text)
    except json.JSONDecodeError:
        return False
    return bool(
        marker.get("manifest_id") == task.manifest_id
        and marker.get("task_id") == task.task_id
    )


# ======================================================================
# Init: describe the written stores and persist the manifest
# ======================================================================


def write_unit(array: ZarrArray) -> tuple[int, ...]:
    """Smallest independently writable block of a zarr array."""
    shards = getattr(array, "shards", None)
    return tuple(int(n) for n in (shards or array.chunks))


def describe_level(
    level: int,
    store_name: str,
    store: ZarrStore,
    template: xr.Dataset,
    task_vars: Sequence[str],
    *,
    zarr_format: Literal[2, 3],
) -> LevelSpec:
    """Build the [`LevelSpec`][grid_doctor.io.LevelSpec] of a freshly written store.

    Shapes and dims come from *template*; chunking and on-disk dtype are
    read back from the store, which is the ground truth workers write to.
    """
    group = zarr.open_group(store, mode="r", zarr_format=zarr_format)
    specs: list[ArraySpec] = []
    for name in task_vars:
        var = template[name]
        array = group[name]
        if not isinstance(array, zarr.Array):
            raise ManifestError(f"{name!r} in {store_name} is not a zarr array")
        specs.append(
            ArraySpec(
                name=name,
                dims=tuple(str(d) for d in var.dims),
                shape=tuple(int(n) for n in var.shape),
                chunks=write_unit(array),
                dtype=str(array.dtype),
            )
        )
    coords = tuple(str(n) for n in template.variables if str(n) not in task_vars)
    return LevelSpec(
        level=level, store=store_name, coords=coords, data_vars=tuple(specs)
    )


def begin_init(path: str, s3_options: S3Options | None) -> None:
    """Invalidate an existing manifest and its markers before re-initialising.

    Removing the manifest first means workers of an earlier run fail
    fast instead of writing into the new stores.  An ``init.json`` records
    who started the init; it is removed once the manifest is written, so
    if it outlives the init (e.g. the job hit its time limit) later calls
    can say so.
    """
    target = _Target(path, s3_options)
    target.remove(STATE_DIR, MANIFEST_FILE)
    target.remove(STATE_DIR, DONE_DIR, recursive=True)
    info = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    target.write_text(json.dumps(info, indent=2), STATE_DIR, INIT_FILE)


def _interrupted_init(target: _Target) -> str | None:
    """Describe an init that started but never wrote its manifest."""
    text = target.read_text(STATE_DIR, INIT_FILE)
    if text is None:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        info = {}
    job = info.get("slurm_job_id")
    return (
        f"an init started {info.get('started', '?')} on host "
        f"{info.get('host', '?')}"
        + (f" (Slurm job {job})" if job else "")
        + " has not finished; it is still running or was killed"
    )


@dataclass(frozen=True)
class ExistingTarget:
    """What an init would find at its target."""

    manifest: PyramidManifest | None
    manifest_error: str | None
    stores: tuple[str, ...]
    interrupted: str | None

    @property
    def exists(self) -> bool:
        """Whether anything of a pyramid is there."""
        return bool(
            self.manifest or self.manifest_error or self.stores or self.interrupted
        )

    def describe(self) -> str:
        """Summarise what is there, for error messages."""
        parts = []
        if self.manifest is not None:
            parts.append(f"manifest {self.manifest.manifest_id}")
        if self.manifest_error:
            parts.append(f"an unreadable manifest ({self.manifest_error})")
        if self.stores:
            parts.append("level stores " + ", ".join(self.stores))
        if self.interrupted:
            parts.append(self.interrupted)
        return "; ".join(parts)


def inspect_target(
    path: str, s3_options: S3Options | None, store_names: Iterable[str]
) -> ExistingTarget:
    """Look for a manifest, level stores and an unfinished init at *path*."""
    target = _Target(path, s3_options)
    manifest: PyramidManifest | None = None
    error: str | None = None
    text = target.read_text(STATE_DIR, MANIFEST_FILE)
    if text is not None:
        try:
            manifest = PyramidManifest.from_json(text)
        except ManifestError as exc:
            error = str(exc)
    stores = tuple(name for name in store_names if target.exists(name))
    return ExistingTarget(manifest, error, stores, _interrupted_init(target))


def resume_init(
    existing: PyramidManifest,
    path: str,
    templates: Mapping[int, xr.Dataset],
    task_vars: Mapping[int, Sequence[str]],
    s3_options: S3Options | None,
    *,
    zarr_format: Literal[2, 3],
    split: Mapping[str, int] | None,
    variables_per_task: int | None,
    materialised: Sequence[str],
) -> PyramidManifest:
    """Return *existing* if the requested init would produce the same plan.

    The plan is recomputed from *templates* (the pyramid as it would be
    written) and the chunking of the stores that already exist, and
    compared with the stored manifest: path, zarr format, levels, array
    layouts, split, materialised variables and every task's variables,
    region and levels.  The coordinate values the init wrote are
    compared too (see `coordinate_differences`), so a pyramid of another
    time range with the same shape is not resumed into the store.
    Nothing is written.

    Raises
    ------
    StoreExistsError
        Listing the differences, if the plans do not match.
    """
    target = _Target(path, s3_options)
    differences: list[str] = []
    if existing.path != path:
        differences.append(f"path {existing.path!r} != {path!r}")
    if existing.zarr_format != zarr_format:
        differences.append(f"zarr_format {existing.zarr_format} != {zarr_format}")
    if existing.materialised != tuple(materialised):
        differences.append(
            f"materialise {list(existing.materialised)} != {list(materialised)}"
        )
    stored_levels = {spec.level for spec in existing.levels}
    if stored_levels != set(templates):
        differences.append(f"levels {sorted(stored_levels)} != {sorted(templates)}")

    levels: list[LevelSpec] = []
    if not differences:
        for level in sorted(templates, reverse=True):
            name = existing.level(level).store
            template = templates[level]
            try:
                spec = describe_level(
                    level,
                    name,
                    target.store(name),
                    template,
                    task_vars[level],
                    zarr_format=zarr_format,
                )
            except (KeyError, FileNotFoundError, ManifestError) as error:
                differences.append(f"level {level}: {error!r}")
                continue
            if spec != existing.level(level):
                differences.append(
                    f"level {level}: layout {_layout_difference(existing.level(level), spec)}"
                )
            differences.extend(
                _chunk_differences(
                    level, template, spec, target.store(name), zarr_format
                )
            )
            differences.extend(
                f"level {level}: {problem}"
                for problem in coordinate_differences(
                    template,
                    target.store(name),
                    existing.level(level).coords,
                    zarr_format=zarr_format,
                )
            )
            levels.append(spec)

    if not differences:
        effective, planned = plan_tasks(
            levels, split=split, variables_per_task=variables_per_task
        )
        if effective != existing.split:
            differences.append(f"split {dict(existing.split)} != {dict(effective)}")
        old = [(t.variables, t.region, t.levels) for t in existing.tasks]
        new = [(p.variables, p.region, p.levels) for p in planned]
        if old != new:
            differences.append(
                f"tasks: {len(old)} stored, {len(new)} planned"
                + ("" if len(old) != len(new) else ", same count but different")
            )

    if differences:
        raise StoreExistsError(
            f"cannot resume {path}: the stored manifest {existing.manifest_id} "
            "was planned differently (" + "; ".join(differences) + "). Use "
            "if_exists='overwrite' to discard it and start over."
        )
    logger.info(
        "Resuming %s with manifest %s (%d task(s))",
        path,
        existing.manifest_id,
        len(existing.tasks),
    )
    return existing


def coordinate_differences(
    dataset: xr.Dataset,
    store: ZarrStore,
    names: Iterable[str],
    *,
    zarr_format: Literal[2, 3],
    region: Mapping[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Compare *dataset*'s coordinates with the ones the init stored.

    The init writes coordinates completely; workers only write data
    variables and trust that their labels match.  A pyramid built from
    another time range (or other levels, members, ...) with the same
    shape would otherwise put its values under the stored labels.

    Checked are the arrays in *names* (a level's ``coords``) that exist in
    both *dataset* and the store and do not span a spatial dimension:
    ``cell``, ``latitude`` and ``longitude`` are a pure function of the
    HEALPix index and can be very large.  Values are encoded exactly as
    they would be stored (same units, dtype, packing) and compared with
    the stored bytes, so the check is exact for times, integers and
    floats alike and independent of the in-memory dtype.

    Parameters
    ----------
    dataset:
        One pyramid level.  Along a dimension in *region* it may cover
        the full stored length or exactly the region.
    store:
        The level store.
    names:
        Arrays to compare.
    zarr_format:
        Zarr format of the store.
    region:
        ``{dim: (start, stop)}``; only this part is compared.  ``None``
        compares everything.

    Returns
    -------
    list[str]
        One human-readable line per mismatching coordinate.
    """
    region = dict(region or {})
    group = zarr.open_group(store, mode="r", zarr_format=zarr_format)
    members = set(group.array_keys())
    encodings: EncodingDict | None = None
    decoded: xr.Dataset | None = None
    problems: list[str] = []
    for name in names:
        if name not in dataset.variables or name not in members:
            continue
        var = dataset.variables[name]
        if set(var.dims) & set(SPATIAL_DIMS):
            continue
        stored = group[name]
        if not isinstance(stored, zarr.Array):
            continue
        stored_dims = _array_dims(stored)
        if stored_dims is not None and set(stored_dims) != set(var.dims):
            problems.append(
                f"coordinate {name!r}: dims {var.dims} != stored {stored_dims}"
            )
            continue
        if stored_dims is not None and tuple(stored_dims) != var.dims:
            var = var.transpose(*stored_dims)
        selection: dict[str, IndexSlice] = {}
        stored_index: list[slice[int, int, None]] = []
        for dim, have, full in zip(var.dims, var.shape, stored.shape):
            start, stop = region.get(str(dim), (0, int(full)))
            if have == full and (start, stop) != (0, full):
                selection[str(dim)] = slice(start, stop)
            elif have not in (full, stop - start):
                problems.append(
                    f"coordinate {name!r}: length {have} along {dim!r}, store "
                    f"has {full}"
                    + (f", task region {stop - start}" if dim in region else "")
                )
                break
            stored_index.append(slice(start, stop))
        else:
            if encodings is None:
                encodings = stored_encodings(store, zarr_format=zarr_format)
            mine = var.isel(selection) if selection else var
            mine = mine.copy(deep=False)
            mine.encoding = dict(encodings.get(name, {}))
            raw_mine = np.asarray(
                encode_zarr_variable(mine, name=name, zarr_format=zarr_format).values
            )
            raw_stored = np.asarray(stored[tuple(stored_index)])
            if raw_mine.shape != raw_stored.shape:
                problems.append(
                    f"coordinate {name!r}: shape {raw_mine.shape} != stored "
                    f"{raw_stored.shape}"
                )
                continue
            equal = _equal_elements(raw_mine, raw_stored)
            if equal.all():
                continue
            position = tuple(int(i) for i in np.argwhere(~equal)[0])
            if decoded is None:
                decoded = xr.open_zarr(store, zarr_format=zarr_format, chunks=None)
            theirs = (
                decoded[name]
                .isel({d: s for d, s in zip(var.dims, stored_index)})
                .values[position]
            )
            ours = mine.values[position]
            where = ", ".join(
                f"{d}={s.start + i}"
                for d, s, i in zip(var.dims, stored_index, position)
            )
            problems.append(
                f"coordinate {name!r} differs at {where or 'its only element'}: "
                f"pyramid has {ours}, store has {theirs}"
            )
    if decoded is not None:
        decoded.close()
    return problems


def _array_dims(array: ZarrArray) -> tuple[str, ...] | None:
    names = getattr(array.metadata, "dimension_names", None)
    if names:
        return tuple(str(n) for n in names)
    dims = array.attrs.get("_ARRAY_DIMENSIONS")
    return tuple(str(d) for d in dims) if isinstance(dims, list) else None


def _equal_elements(a: GenericArray, b: GenericArray) -> BoolArray:
    """Elementwise equality that treats NaN == NaN (and NaT == NaT)."""
    equal = np.asarray(a == b)
    if a.dtype.kind in "fcmM" and b.dtype.kind in "fcmM":
        equal = equal | (np.isnan(a) & np.isnan(b))
    return equal


def _layout_difference(old: LevelSpec, new: LevelSpec) -> str:
    if old.coords != new.coords:
        return f"coords {list(old.coords)} != {list(new.coords)}"
    old_vars = {a.name: a for a in old.data_vars}
    new_vars = {a.name: a for a in new.data_vars}
    if old_vars.keys() != new_vars.keys():
        return f"variables {sorted(old_vars)} != {sorted(new_vars)}"
    for name, spec in old_vars.items():
        if spec != new_vars[name]:
            return f"{name!r}: {spec} != {new_vars[name]}"
    return "differs"


def _chunk_differences(
    level: int,
    template: xr.Dataset,
    spec: LevelSpec,
    store: ZarrStore,
    zarr_format: Literal[2, 3],
) -> list[str]:
    """Chunks the init would now choose vs. the chunks already stored.

    The stored chunks drive the plan, but a difference means the pyramid
    is no longer built the way it was at init time.  Only dask-backed
    variables have a predictable chunking (their first chunk per dim);
    for sharded arrays this compares the inner chunks, not the shards.
    """
    group = zarr.open_group(store, mode="r", zarr_format=zarr_format)
    result = []
    for array in spec.data_vars:
        var = template[array.name]
        if var.chunks is None:
            continue
        wanted = tuple(int(c[0]) for c in var.chunksizes.values())
        node = group[array.name]
        if not isinstance(node, zarr.Array):
            continue
        stored = tuple(int(n) for n in node.chunks)
        if wanted != stored:
            result.append(
                f"level {level}: {array.name!r} chunks {wanted} != stored {stored}"
            )
    return result


def finish_init(
    path: str,
    s3_options: S3Options | None,
    levels: Sequence[LevelSpec],
    *,
    zarr_format: Literal[2, 3],
    split: Mapping[str, int] | None,
    variables_per_task: int | None,
    materialised: Sequence[str],
) -> PyramidManifest:
    """Plan the tasks, assemble the manifest and store it below *path*."""
    from .. import __version__

    effective, planned = plan_tasks(
        levels, split=split, variables_per_task=variables_per_task
    )
    manifest_id = uuid.uuid4().hex
    width = max(6, len(str(len(planned))))
    tasks = tuple(
        WriteTask(
            task_id=f"task-{i:0{width}d}",
            manifest_id=manifest_id,
            path=path.rstrip("/"),
            zarr_format=zarr_format,
            variables=p.variables,
            region=p.region,
            levels=p.levels,
        )
        for i, p in enumerate(planned)
    )
    manifest = PyramidManifest(
        manifest_id=manifest_id,
        path=path.rstrip("/"),
        zarr_format=zarr_format,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        grid_doctor_version=__version__,
        split=effective,
        levels=tuple(levels),
        tasks=tasks,
        materialised=tuple(materialised),
    )
    target = _Target(path, s3_options)
    target.write_text(manifest.to_json(), STATE_DIR, MANIFEST_FILE)
    target.remove(STATE_DIR, INIT_FILE)
    logger.info(
        "Initialised %s: %d level(s), %d task(s), manifest %s",
        path,
        len(levels),
        len(tasks),
        manifest_id,
    )
    return manifest


# ======================================================================
# Worker side
# ======================================================================


def write_pyramid_task(
    pyramid: Mapping[int, xr.Dataset],
    task: WriteTask,
    s3_options: S3Options | None = None,
) -> WriteTask:
    """Write one task's region of *pyramid* into an initialised store.

    This is the function an array-job element calls.  It is safe to run
    any number of tasks of the same manifest concurrently, on any number
    of nodes, without locks: the planner guarantees that no two tasks
    share a zarr chunk (or shard).  Re-running a task overwrites the
    same region with the same data, so retries are harmless.

    Parameters
    ----------
    pyramid:
        Mapping of level to dataset, built the same way as for the init
        step (same levels, variables, dimension lengths).  It may cover
        either the **full** extent of the split dimensions, in which case
        the task region is selected here, or **only the task region**,
        e.g. because the source was subset with
        ``source.isel(task.isel())`` before regridding.  The second form
        is usually much cheaper, since only the task's slice of the
        source is read and regridded.  Datasets may contain more
        variables or levels than the task needs; they are ignored.
    task:
        The work unit, as taken from ``manifest.tasks``.
    s3_options:
        Options for :class:`s3fs.S3FileSystem`.  Credentials are never
        stored in the manifest, so every worker passes its own.

    Returns
    -------
    WriteTask
        *task* itself, after the data and the completion marker have
        been written.  Returning the task (rather than nothing) makes
        the result usable as a cache key: reflow can check a cached
        result with ``verify=is_task_done``.

    Raises
    ------
    StaleTaskError
        The store was re-initialised after *task* was planned.
    ManifestError
        *task* is not identical to the task of the same id in the stored
        manifest (only tasks the manifest validated are written), or
        *pyramid* does not match the manifest (missing level or variable,
        different dimension lengths, a region the data cannot cover), or
        its coordinate values in the task region differ from the ones the
        init stored (e.g. another time range of the same length).
    """
    started = time.monotonic()
    target = _Target(task.path, s3_options)
    manifest = load_manifest(task.path, s3_options)
    if manifest.manifest_id != task.manifest_id:
        raise StaleTaskError(
            f"{task.task_id} belongs to manifest {task.manifest_id}, but "
            f"{task.path} was re-initialised with manifest "
            f"{manifest.manifest_id}; plan the tasks again from the new manifest"
        )
    try:
        planned = manifest.task(task.task_id)
    except KeyError:
        raise ManifestError(
            f"{task.task_id} is not part of manifest {manifest.manifest_id}"
        ) from None
    if planned != task:
        # The manifest validated that its own tasks are chunk-aligned and
        # disjoint; a task that differs (edited or hand-built) has no such
        # guarantee and could race with another worker.
        raise ManifestError(
            f"{task.task_id} differs from the task stored in the manifest "
            f"({task} != {planned}); use the tasks of the stored manifest"
        )

    writes: list[DeferredWrite] = []
    for level in task.levels:
        if level not in pyramid:
            raise ManifestError(
                f"{task.task_id} writes level {level}, but the pyramid only has "
                f"levels {sorted(pyramid)}"
            )
        spec = manifest.level(level)
        store = target.store(spec.store)
        group = zarr.open_group(store, mode="r+", zarr_format=task.zarr_format)
        encodings = stored_encodings(store, zarr_format=task.zarr_format)
        for name in task.variables:
            array_spec = spec.array(name)
            if name not in pyramid[level]:
                raise ManifestError(
                    f"{task.task_id}: variable {name!r} missing on level {level}"
                )
            var = _select_region(pyramid[level][name].variable, array_spec, task)
            var = var.copy(deep=False)
            var.encoding = encodings.get(name, {})
            encoded = encode_zarr_variable(var, name=name, zarr_format=task.zarr_format)
            offsets = _region_offsets(array_spec, task)
            source = _aligned(encoded.data, array_spec.chunks, offsets, name)
            region = tuple(
                slice(off, off + n) for off, n in zip(offsets, encoded.shape)
            )
            writes.append(DeferredWrite(source, get_array(group, name), region))

        mismatches = coordinate_differences(
            pyramid[level],
            store,
            spec.coords,
            zarr_format=task.zarr_format,
            region={dim: (start, stop) for dim, start, stop in task.region},
        )
        if mismatches:
            raise ManifestError(
                f"{task.task_id}: the pyramid does not match the coordinates "
                f"stored by the init on level {level} ("
                + "; ".join(mismatches)
                + ").  It was built from other data than the store; nothing "
                "was written."
            )

    store_all(writes)
    # The manifest was checked before writing.  If the store was
    # re-initialised meanwhile (an overwrite while this task ran), the
    # data above may have landed in the new stores: fail loudly instead
    # of reporting success, and never write a marker for it.
    if not is_current(task, s3_options):
        raise StaleTaskError(
            f"{task.path} was re-initialised while {task.task_id} was writing; "
            "its data may have been written into the new stores.  Do not "
            "re-initialise a store while a conversion into it is running."
        )
    marker = {
        "manifest_id": task.manifest_id,
        "task_id": task.task_id,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - started, 3),
        "host": socket.gethostname(),
        "variables": list(task.variables),
        "levels": list(task.levels),
        "region": {d: [a, b] for d, a, b in task.region},
    }
    target.write_text(
        json.dumps(marker, indent=2), STATE_DIR, DONE_DIR, f"{task.task_id}.json"
    )
    logger.info(
        "%s done (%s, %d level(s)) in %.1fs",
        task.task_id,
        ", ".join(task.variables),
        len(task.levels),
        marker["elapsed_s"],
    )
    return task


def _select_region(var: xr.Variable, spec: ArraySpec, task: WriteTask) -> xr.Variable:
    """Bring *var* to the task's region, accepting full or pre-subset data."""
    if set(var.dims) != set(spec.dims):
        raise ManifestError(
            f"{task.task_id}: {spec.name!r} has dims {var.dims}, the store "
            f"expects {spec.dims}"
        )
    if var.dims != spec.dims:
        var = var.transpose(*spec.dims)
    region = {dim: (start, stop) for dim, start, stop in task.region}
    selection: dict[str, IndexSlice] = {}
    for dim in spec.dims:
        have, full = var.sizes[dim], spec.size(dim)
        if dim in region:
            start, stop = region[dim]
            if have == full:
                selection[dim] = slice(start, stop)
            elif have != stop - start:
                raise ManifestError(
                    f"{task.task_id}: {spec.name!r} has {have} elements along "
                    f"{dim!r}; expected the full length {full} or the task "
                    f"region length {stop - start} ([{start}, {stop}))"
                )
        elif have != full:
            raise ManifestError(
                f"{task.task_id}: {spec.name!r} has {have} elements along "
                f"{dim!r}, the store has {full}"
            )
    return var.isel(selection) if selection else var


def _region_offsets(spec: ArraySpec, task: WriteTask) -> tuple[int, ...]:
    starts = {dim: start for dim, start, _ in task.region}
    return tuple(starts.get(dim, 0) for dim in spec.dims)


def _aligned(
    data: da.Array | GenericArray,
    units: tuple[int, ...],
    offsets: tuple[int, ...],
    name: str,
) -> da.Array:
    """Return *data* as dask array whose blocks never split a write unit.

    ``da.store`` writes every block independently.  Two blocks inside one
    zarr chunk would race on read-modify-write of that chunk, so blocks
    must end on write-unit boundaries (counted from the array origin, so
    the region offset matters).
    """
    if not isinstance(data, da.Array):
        clipped = tuple(max(1, min(u, n)) for u, n in zip(units, data.shape))
        return cast(
            da.Array,
            da.from_array(data, chunks=clipped),  # type: ignore[no-untyped-call,unused-ignore]
        )
    for block_sizes, unit, offset in zip(data.chunks, units, offsets):
        position = offset
        for size in block_sizes[:-1]:
            position += size
            if position % unit:
                logger.warning(
                    "Rechunking %r to the store's write units %s: its dask "
                    "chunks %s would split zarr chunks",
                    name,
                    units,
                    data.chunksize,
                )
                clipped = tuple(max(1, min(u, n)) for u, n in zip(units, data.shape))
                return cast(
                    da.Array,
                    data.rechunk(clipped),  # type: ignore[no-untyped-call,unused-ignore]
                )
    return data


# ======================================================================
# Completeness
# ======================================================================


@dataclass(frozen=True)
class StoreReport:
    """Result of [`check_pyramid_store`][grid_doctor.check_pyramid_store].

    Attributes
    ----------
    manifest_id:
        Manifest the store was checked against.
    n_tasks:
        Number of tasks in the manifest.
    done_tasks:
        Ids of tasks with a valid completion marker.
    missing_tasks:
        Ids of tasks without one.  Feed them to
        [`PyramidManifest.select_tasks`][grid_doctor.PyramidManifest.select_tasks]
        to retry only those.
    problems:
        Structural mismatches between store and manifest (missing level
        store or array, changed shape, chunking or dtype, a store that
        was re-initialised with a different manifest).
    stale_markers:
        Markers that belong to another manifest or name unknown tasks.
        They are ignored.
    absent_chunks:
        Only with ``inspect_chunks=True``: ``(task_id, absent, expected)``
        for finished tasks whose region has chunks that are not stored.
        This is informational: zarr skips chunks that contain only fill
        values, so absent chunks are expected for masked or regional
        data.
    """

    manifest_id: str
    n_tasks: int
    done_tasks: tuple[str, ...]
    missing_tasks: tuple[str, ...]
    problems: tuple[str, ...] = ()
    stale_markers: int = 0
    absent_chunks: tuple[tuple[str, int, int], ...] = ()

    @property
    def complete(self) -> bool:
        """``True`` if every task finished and the layout matches."""
        return not self.missing_tasks and not self.problems

    def __bool__(self) -> bool:
        """Return [`complete`][grid_doctor.StoreReport.complete]."""
        return self.complete

    def summary(self) -> str:
        """One human-readable paragraph."""
        state = "complete" if self.complete else "INCOMPLETE"
        lines = [
            (
                f"Store {state}: {len(self.done_tasks)}/{self.n_tasks} task(s) "
                f"done (manifest {self.manifest_id})."
            )
        ]
        if self.missing_tasks:
            shown = ", ".join(self.missing_tasks[:10])
            more = len(self.missing_tasks) - 10
            lines.append(
                f"Missing: {shown}" + (f" and {more} more." if more > 0 else ".")
            )
        lines.extend(f"Problem: {p}" for p in self.problems)
        if self.stale_markers:
            lines.append(f"Ignored {self.stale_markers} stale marker(s).")
        if self.absent_chunks:
            absent = sum(a for _, a, _ in self.absent_chunks)
            lines.append(
                f"{absent} chunk(s) not stored in {len(self.absent_chunks)} finished "
                "task(s) (all-fill-value chunks are skipped by zarr; this is "
                "expected for masked or regional data)."
            )
        return "\n".join(lines)

    def raise_if_incomplete(self) -> None:
        """Raise :class:`IncompleteStoreError` unless the store is complete."""
        if not self.complete:
            raise IncompleteStoreError(self.summary())


def check_pyramid_store(
    manifest: PyramidManifest | str,
    s3_options: S3Options | None = None,
    *,
    inspect_chunks: bool = False,
) -> StoreReport:
    """Check whether a store is complete according to its manifest.

    The check is cheap: it reads the level metadata and lists one small
    marker file per finished task.  It does **not** read any data.

    A store is complete when

    1. the stored manifest is the one being checked (the store was not
       re-initialised in between),
    2. every level store and every array exists with the shape, write
       unit and dtype recorded in the manifest, and
    3. every task has written its completion marker.

    Parameters
    ----------
    manifest:
        The manifest returned by
        [`init_pyramid_store`][grid_doctor.init_pyramid_store],
        or the pyramid root path, in which case the stored manifest is
        loaded.
    s3_options:
        Options for :class:`s3fs.S3FileSystem` (S3 targets only).
    inspect_chunks:
        Also list the stored chunk keys and report, per finished task,
        how many chunks of its region are absent.  This lists every
        object of every array and can be slow on large S3 stores.  It
        never makes a store incomplete; see
        [`StoreReport.absent_chunks`][grid_doctor.StoreReport].

    Returns
    -------
    StoreReport
        Use ``report.complete`` (or ``bool(report)``),
        ``report.missing_tasks`` and ``report.summary()``; call
        ``report.raise_if_incomplete()`` to fail a workflow step.
    """
    if isinstance(manifest, str):
        manifest = load_manifest(manifest, s3_options)
    target = _Target(manifest.path, s3_options)
    problems: list[str] = []

    stored_text = target.read_text(STATE_DIR, MANIFEST_FILE)
    if stored_text is None:
        interrupted = _interrupted_init(target)
        problems.append(
            f"no manifest stored at {manifest.path}/{STATE_DIR}"
            + (f": {interrupted}" if interrupted else "")
        )
    else:
        stored_id = json.loads(stored_text).get("manifest_id")
        if stored_id != manifest.manifest_id:
            problems.append(
                f"store was re-initialised: it holds manifest {stored_id}, "
                f"not {manifest.manifest_id}"
            )

    groups: dict[int, zarr.Group] = {}
    for spec in manifest.levels:
        try:
            group = zarr.open_group(
                target.store(spec.store), mode="r", zarr_format=manifest.zarr_format
            )
        except Exception as error:  # noqa: BLE001 - report, do not raise
            problems.append(f"level {spec.level}: cannot open {spec.store}: {error}")
            continue
        groups[spec.level] = group
        members = set(group.array_keys())
        for name in spec.coords:
            if name not in members:
                problems.append(f"level {spec.level}: coordinate {name!r} missing")
        for array_spec in spec.data_vars:
            problems.extend(_check_array(spec.level, group, members, array_spec))

    known = {t.task_id for t in manifest.tasks}
    done: set[str] = set()
    stale = 0
    for filename in target.list_dir(STATE_DIR, DONE_DIR):
        if not filename.endswith(".json") or filename.startswith("."):
            continue
        text = target.read_text(STATE_DIR, DONE_DIR, filename)
        try:
            marker = json.loads(text or "")
        except json.JSONDecodeError:
            stale += 1
            continue
        task_id = marker.get("task_id")
        if marker.get("manifest_id") == manifest.manifest_id and task_id in known:
            done.add(str(task_id))
        else:
            stale += 1

    order = [t.task_id for t in manifest.tasks]
    done_tasks = tuple(t for t in order if t in done)
    missing = tuple(t for t in order if t not in done)

    absent: list[tuple[str, int, int]] = []
    if inspect_chunks:
        absent = _absent_chunks(
            manifest, target, groups, [manifest.task(t) for t in done_tasks]
        )

    return StoreReport(
        manifest_id=manifest.manifest_id,
        n_tasks=len(manifest.tasks),
        done_tasks=done_tasks,
        missing_tasks=missing,
        problems=tuple(problems),
        stale_markers=stale,
        absent_chunks=tuple(absent),
    )


def _check_array(
    level: int, group: zarr.Group, members: set[str], spec: ArraySpec
) -> list[str]:
    if spec.name not in members:
        return [f"level {level}: array {spec.name!r} missing"]
    array = get_array(group, spec.name)
    problems = []
    if tuple(array.shape) != spec.shape:
        problems.append(
            f"level {level}: {spec.name!r} has shape {tuple(array.shape)}, "
            f"manifest says {spec.shape}"
        )
    if write_unit(array) != spec.chunks:
        problems.append(
            f"level {level}: {spec.name!r} has write unit {write_unit(array)}, "
            f"manifest says {spec.chunks}"
        )
    if str(array.dtype) != spec.dtype:
        problems.append(
            f"level {level}: {spec.name!r} has dtype {array.dtype}, "
            f"manifest says {spec.dtype}"
        )
    return problems


def _absent_chunks(
    manifest: PyramidManifest,
    target: _Target,
    groups: Mapping[int, zarr.Group],
    tasks: Iterable[WriteTask],
) -> list[tuple[str, int, int]]:
    listings: dict[tuple[int, str], set[str]] = {}
    result: list[tuple[str, int, int]] = []
    for task in tasks:
        region = {dim: (start, stop) for dim, start, stop in task.region}
        absent = expected = 0
        for level in task.levels:
            if level not in groups:
                continue
            spec = manifest.level(level)
            for name in task.variables:
                array_spec = spec.array(name)
                key = (level, name)
                if key not in listings:
                    listings[key] = target.find(spec.store, name)
                array = get_array(groups[level], name)
                ranges = []
                for dim, size, unit in zip(
                    array_spec.dims, array_spec.shape, array_spec.chunks
                ):
                    start, stop = region.get(dim, (0, size))
                    ranges.append(range(start // unit, math.ceil(stop / unit)))
                for coords in itertools.product(*ranges):
                    expected += 1
                    if array.metadata.encode_chunk_key(coords) not in listings[key]:
                        absent += 1
        if absent:
            result.append((task.task_id, absent, expected))
    return result


__all__ = [
    "ExistingTarget",
    "IfExists",
    "IncompleteStoreError",
    "StaleTaskError",
    "StoreExistsError",
    "StoreReport",
    "begin_init",
    "check_pyramid_store",
    "describe_level",
    "finish_init",
    "inspect_target",
    "resume_init",
    "is_current",
    "is_task_done",
    "load_manifest",
    "write_pyramid_task",
    "write_unit",
]
