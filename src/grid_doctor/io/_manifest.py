"""Serialisable description of an initialised pyramid store.

A [`PyramidManifest`][grid_doctor.PyramidManifest] is what
[`init_pyramid_store`][grid_doctor.init_pyramid_store] returns:
the layout of every level store (array names, dimensions, shapes,
chunking) plus a list of independent
[`WriteTask`][grid_doctor.WriteTask] work units.  Each task names the
variables and the index region one worker has to write.  Regions are
planned so that no two tasks ever touch the same zarr chunk (or shard),
which is what makes uncoordinated parallel writes safe.

Everything in here is pure data: no I/O, no xarray objects.  The
classes are frozen dataclasses built from ``str``, ``int`` and
``tuple`` only, so they

- round-trip through plain JSON (``to_json`` / ``from_json``), and
- are accepted unchanged as return values and inputs of reflow tasks,
  whose manifest codec serialises dataclasses and tuples natively.

Every instance validates itself in ``__post_init__``, however it was
created: parsed from JSON, rebuilt by reflow's codec (which calls the
constructor directly and never goes through ``from_json``), or built by
hand.  Field types and shapes are checked per class;
[`PyramidManifest`][grid_doctor.PyramidManifest] additionally checks that
its tasks fit the level layouts and partition every data variable into
chunk-aligned, non-overlapping regions, which is the property parallel
writes rely on.  Violations raise
[`ManifestError`][grid_doctor.io.ManifestError] naming the offending
field.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, cast

logger = logging.getLogger(__name__)

#: Version of the manifest JSON layout.  Bumped on incompatible changes.
MANIFEST_SCHEMA_VERSION = 1

#: Directory below the pyramid root that holds the manifest and the
#: per-task completion markers.
STATE_DIR = "_grid_doctor"

#: File name of the persisted manifest inside :data:`STATE_DIR`.
MANIFEST_FILE = "manifest.json"

#: Directory inside :data:`STATE_DIR` with one marker file per finished task.
DONE_DIR = "done"

#: File inside :data:`STATE_DIR` that exists while an init is running.  If
#: it is still there without a manifest, the init was interrupted.
INIT_FILE = "init.json"

#: Dimensions that must never be split across tasks.  Pyramid levels
#: are derived from the finest level by aggregating child cells, so a
#: ``cell`` region of the finest level maps to a 4**k times smaller,
#: generally chunk-misaligned region on coarser levels.
SPATIAL_DIMS: tuple[str, ...] = ("cell",)

Region = tuple[tuple[str, int, int], ...]
"""Index region as ``((dim, start, stop), ...)``, half-open like ``slice``."""


class ManifestError(ValueError):
    """The manifest cannot be built, read, or does not match the store."""


@dataclass(frozen=True)
class ArraySpec:
    """Layout of one data variable in one level store.

    Attributes
    ----------
    name:
        Variable name.
    dims:
        Dimension names, in storage order.
    shape:
        Full array shape.
    chunks:
        Size of the smallest independently writable unit along every
        dimension: the shard shape for sharded zarr v3 arrays, the chunk
        shape otherwise.  Task regions start on multiples of these.
    dtype:
        On-disk dtype (after CF encoding, e.g. ``int16`` for packed data).
    """

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        """Validate and normalise the fields (lists become tuples)."""
        where = f"array {self.name!r}"
        dims = _str_tuple(self.dims, where, "dims", unique=True)
        shape = _int_tuple(self.shape, where, "shape", minimum=0)
        chunks = _int_tuple(self.chunks, where, "chunks", minimum=1)
        if not len(dims) == len(shape) == len(chunks):
            _fail(
                where,
                f"dims {dims}, shape {shape} and chunks {chunks} must have "
                "the same length",
            )
        _set(
            self,
            name=_name(self.name, where, "name"),
            dims=dims,
            shape=shape,
            chunks=chunks,
            dtype=_as_str(self.dtype, where, "dtype"),
        )

    def size(self, dim: str) -> int:
        """Length of *dim*."""
        return self.shape[self.dims.index(dim)]

    def chunk(self, dim: str) -> int:
        """Write-unit size along *dim*."""
        return self.chunks[self.dims.index(dim)]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {
            "name": self.name,
            "dims": list(self.dims),
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, where: str = "array") -> ArraySpec:
        """Inverse of [`to_dict`][grid_doctor.io.ArraySpec.to_dict].

        *where* names the location in error messages.
        """
        data = _mapping(data, where)
        return cls(
            name=_field(data, "name", where),
            dims=_field(data, "dims", where),
            shape=_field(data, "shape", where),
            chunks=_field(data, "chunks", where),
            dtype=_field(data, "dtype", where),
        )


@dataclass(frozen=True)
class LevelSpec:
    """Layout of one level store (``<path>/level_<n>.zarr``).

    Attributes
    ----------
    level:
        HEALPix refinement level.
    store:
        Store name relative to the pyramid root, e.g. ``"level_7.zarr"``.
    coords:
        Coordinate (and other non-task) arrays written completely by the
        init step.  Workers never touch them.
    data_vars:
        Data variables that are written by tasks.  Variables listed in
        ``materialise`` at init time are written by the init step and
        appear in ``coords`` instead.
    """

    level: int
    store: str
    coords: tuple[str, ...]
    data_vars: tuple[ArraySpec, ...]

    def __post_init__(self) -> None:
        """Validate and normalise the fields (lists become tuples)."""
        where = f"level {self.level!r}"
        coords = _str_tuple(self.coords, where, "coords", unique=True)
        data_vars = _as_tuple(self.data_vars, where, "data_vars")
        for spec in data_vars:
            if not isinstance(spec, ArraySpec):
                _fail(
                    where,
                    "data_vars must contain ArraySpec objects, got "
                    f"{type(spec).__name__}",
                )
        names = [spec.name for spec in data_vars]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            _fail(where, f"duplicate data variables {duplicates}")
        both = sorted(set(names) & set(coords))
        if both:
            _fail(where, f"{both} listed as coordinate and data variable")
        _set(
            self,
            level=_as_int(self.level, where, "level", minimum=0),
            store=_as_str(self.store, where, "store", safe=True),
            coords=coords,
            data_vars=data_vars,
        )

    def array(self, name: str) -> ArraySpec:
        """Return the spec of data variable *name*."""
        for spec in self.data_vars:
            if spec.name == name:
                return spec
        raise KeyError(f"level {self.level} has no data variable {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {
            "level": self.level,
            "store": self.store,
            "coords": list(self.coords),
            "data_vars": [spec.to_dict() for spec in self.data_vars],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, where: str = "level") -> LevelSpec:
        """Inverse of [`to_dict`][grid_doctor.io.LevelSpec.to_dict].

        *where* names the location in error messages.
        """
        data = _mapping(data, where)
        raw_vars = _as_tuple(_field(data, "data_vars", where), where, "data_vars")
        return cls(
            level=_field(data, "level", where),
            store=_field(data, "store", where),
            coords=_field(data, "coords", where),
            data_vars=tuple(
                ArraySpec.from_dict(v, where=f"{where}.data_vars[{i}]")
                for i, v in enumerate(raw_vars)
            ),
        )


@dataclass(frozen=True)
class WriteTask:
    """One self-contained unit of work for a single worker.

    A task carries everything a worker needs besides the data itself and
    the storage credentials: where the store lives, which variables to
    write on which levels, and the index region along the split
    dimensions.  Dimensions not listed in ``region`` are written in full.

    Attributes
    ----------
    task_id:
        Identifier, unique within the manifest (``"task-000042"``).
    manifest_id:
        Id of the manifest this task belongs to.  Workers refuse to run a
        task whose id does not match the manifest stored in the target,
        which protects a re-initialised store from stale array jobs.
    path:
        Pyramid root, as passed to ``init_pyramid_store``.
    zarr_format:
        Zarr format of the level stores.
    variables:
        Data variables to write.
    region:
        ``((dim, start, stop), ...)``.  Empty for variables that have
        none of the split dimensions (e.g. a static land-sea mask).
    levels:
        Levels the task writes, finest first.
    """

    task_id: str
    manifest_id: str
    path: str
    zarr_format: Literal[2, 3]
    variables: tuple[str, ...]
    region: Region
    levels: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate and normalise the fields (lists become tuples).

        A task on its own can only be checked for well-formed fields.
        Whether its region is chunk-aligned is checked by the manifest
        that contains it, and by
        [`write_pyramid_task`][grid_doctor.write_pyramid_task], which
        refuses tasks that differ from the stored manifest.
        """
        where = f"task {self.task_id!r}"
        _set(
            self,
            task_id=_as_str(self.task_id, where, "task_id", safe=True),
            manifest_id=_as_str(self.manifest_id, where, "manifest_id", safe=True),
            path=_as_str(self.path, where, "path"),
            zarr_format=_zarr_format(self.zarr_format, where),
            variables=_str_tuple(
                self.variables, where, "variables", unique=True, nonempty=True
            ),
            region=_region(self.region, where),
            levels=_int_tuple(
                self.levels, where, "levels", minimum=0, unique=True, nonempty=True
            ),
        )
        for name in self.variables:
            _name(name, where, "variables")

    @property
    def slices(self) -> dict[str, slice]:
        """The region as an ``isel``-ready mapping, e.g. ``{"time": slice(0, 24)}``."""
        return {dim: slice(start, stop) for dim, start, stop in self.region}

    def isel(self) -> dict[str, slice]:
        """Return the region as a mapping for ``Dataset.isel``.

        Same as [`slices`][grid_doctor.WriteTask.slices]; use it as
        ``ds.isel(task.isel())``.
        """
        return self.slices

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation with all fields."""
        return {
            "task_id": self.task_id,
            "manifest_id": self.manifest_id,
            "path": self.path,
            "zarr_format": self.zarr_format,
            "variables": list(self.variables),
            "region": _region_to_json(self.region),
            "levels": list(self.levels),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        where: str = "task",
    ) -> WriteTask:
        """Inverse of [`to_dict`][grid_doctor.WriteTask.to_dict].

        *defaults* supplies ``manifest_id``, ``path`` and ``zarr_format``
        when they are absent, as inside a manifest file.  *where* names
        the location in error messages.
        """
        merged = {**(defaults or {}), **_mapping(data, where)}
        return cls(
            task_id=_field(merged, "task_id", where),
            manifest_id=_field(merged, "manifest_id", where),
            path=_field(merged, "path", where),
            zarr_format=_field(merged, "zarr_format", where),
            variables=_field(merged, "variables", where),
            region=_field(merged, "region", where),
            levels=_field(merged, "levels", where),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str | bytes) -> WriteTask:
        """Parse a task written by [`to_json`][grid_doctor.WriteTask.to_json]."""
        return cls.from_dict(_loads(text, "task"))


@dataclass(frozen=True)
class PyramidManifest:
    """Layout of an initialised pyramid and the work needed to fill it.

    Returned by
    [`init_pyramid_store`][grid_doctor.init_pyramid_store] and
    stored as ``<path>/_grid_doctor/manifest.json``.

    Attributes
    ----------
    manifest_id:
        Random id generated at init.  Every task and every completion
        marker carries it.
    path:
        Pyramid root.
    zarr_format:
        Zarr format of all level stores.
    created:
        ISO 8601 UTC timestamp of the init.
    grid_doctor_version:
        Version of grid-doctor that wrote the manifest.
    split:
        Effective task size per split dimension, in index units, after
        rounding up to whole write units.  Empty when nothing was split.
    levels:
        One [`LevelSpec`][grid_doctor.io.LevelSpec] per level, finest first.
    tasks:
        All work units.  Together they cover every data variable on
        every level exactly once.
    materialised:
        Data variables written completely by the init step.
    schema_version:
        Manifest layout version, see :data:`MANIFEST_SCHEMA_VERSION`.
    """

    manifest_id: str
    path: str
    zarr_format: Literal[2, 3]
    created: str
    grid_doctor_version: str
    split: tuple[tuple[str, int], ...]
    levels: tuple[LevelSpec, ...]
    tasks: tuple[WriteTask, ...]
    materialised: tuple[str, ...] = ()
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the fields and the consistency of levels and tasks.

        Besides field types this enforces what parallel writes depend
        on, for every data variable on every level:

        - every task region lies inside the array and starts and ends on
          write-unit boundaries (or the array end),
        - the tasks writing it partition it exactly: no gaps, no
          overlaps, no duplicates,
        - every data variable is written by at least one task.
        """
        where = "manifest"
        _check_schema_version(self.schema_version)
        levels = _as_tuple(self.levels, where, "levels")
        tasks = _as_tuple(self.tasks, where, "tasks")
        for item, cls, name in (
            (levels, LevelSpec, "levels"),
            (tasks, WriteTask, "tasks"),
        ):
            for obj in item:
                if not isinstance(obj, cls):
                    _fail(
                        where,
                        f"{name} must contain {cls.__name__} objects, got "
                        f"{type(obj).__name__}",
                    )
        _set(
            self,
            manifest_id=_as_str(self.manifest_id, where, "manifest_id", safe=True),
            path=_as_str(self.path, where, "path"),
            zarr_format=_zarr_format(self.zarr_format, where),
            created=_as_str(self.created, where, "created"),
            grid_doctor_version=_as_str(
                self.grid_doctor_version, where, "grid_doctor_version"
            ),
            split=_split(self.split, where),
            levels=levels,
            tasks=tasks,
            materialised=_str_tuple(
                self.materialised, where, "materialised", unique=True
            ),
        )
        _check_consistency(self)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def level(self, level: int) -> LevelSpec:
        """Return the spec of *level*."""
        for spec in self.levels:
            if spec.level == level:
                return spec
        raise KeyError(f"manifest has no level {level}")

    def task(self, task_id: str) -> WriteTask:
        """Return the task with id *task_id*."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"manifest has no task {task_id!r}")

    def select_tasks(self, task_ids: Iterable[str]) -> list[WriteTask]:
        """Return the tasks with the given ids, in manifest order.

        Handy for re-submitting only what
        [`check_pyramid_store`][grid_doctor.check_pyramid_store] reported
        as missing::

            retry = manifest.select_tasks(report.missing_tasks)
        """
        wanted = set(task_ids)
        unknown = wanted - {t.task_id for t in self.tasks}
        if unknown:
            raise KeyError(f"unknown task ids: {sorted(unknown)}")
        return [t for t in self.tasks if t.task_id in wanted]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation.

        Tasks are written without the fields they share with the
        manifest (``manifest_id``, ``path``, ``zarr_format``).
        """
        shared = {"manifest_id", "path", "zarr_format"}
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "path": self.path,
            "zarr_format": self.zarr_format,
            "created": self.created,
            "grid_doctor_version": self.grid_doctor_version,
            "split": dict(self.split),
            "materialised": list(self.materialised),
            "levels": [spec.to_dict() for spec in self.levels],
            "tasks": [
                {k: v for k, v in task.to_dict().items() if k not in shared}
                for task in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PyramidManifest:
        """Inverse of [`to_dict`][grid_doctor.PyramidManifest.to_dict].

        The schema version is checked first, so a manifest from an
        incompatible grid-doctor fails with a clear message rather than
        with a missing field.
        """
        where = "manifest"
        data = _mapping(data, where)
        _check_schema_version(data.get("schema_version"))
        defaults = {
            key: _field(data, key, where)
            for key in ("manifest_id", "path", "zarr_format")
        }
        raw_split = _field(data, "split", where)
        if not isinstance(raw_split, Mapping):
            _fail(where, f"split must be an object, got {type(raw_split).__name__}")
        raw_levels = _as_tuple(_field(data, "levels", where), where, "levels")
        raw_tasks = _as_tuple(_field(data, "tasks", where), where, "tasks")
        return cls(
            manifest_id=defaults["manifest_id"],
            path=defaults["path"],
            zarr_format=defaults["zarr_format"],
            created=_field(data, "created", where),
            grid_doctor_version=_field(data, "grid_doctor_version", where),
            split=tuple(raw_split.items()),
            materialised=data.get("materialised", ()),
            levels=tuple(
                LevelSpec.from_dict(lvl, where=f"levels[{i}]")
                for i, lvl in enumerate(raw_levels)
            ),
            tasks=tuple(
                WriteTask.from_dict(task, defaults=defaults, where=f"tasks[{i}]")
                for i, task in enumerate(raw_tasks)
            ),
            schema_version=data["schema_version"],
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, text: str | bytes) -> PyramidManifest:
        """Parse a manifest written by [`to_json`][grid_doctor.PyramidManifest.to_json]."""
        return cls.from_dict(_loads(text, "manifest"))


# ======================================================================
# Planning
# ======================================================================


@dataclass(frozen=True)
class _PlannedTask:
    variables: tuple[str, ...]
    region: Region
    levels: tuple[int, ...]


@dataclass
class _VariablePlacement:
    split_dims: tuple[str, ...]
    levels: list[int] = field(default_factory=list)


def plan_tasks(
    levels: Sequence[LevelSpec],
    *,
    split: Mapping[str, int] | None = None,
    variables_per_task: int | None = None,
) -> tuple[tuple[tuple[str, int], ...], list[_PlannedTask]]:
    """Cut the data variables of *levels* into chunk-aligned tasks.

    Parameters
    ----------
    levels:
        Level layouts, as read back from the initialised stores.
    split:
        Requested task size per dimension in index units, e.g.
        ``{"time": 24}``.  Each size is rounded **up** to a multiple of
        the least common multiple of all write-unit sizes along that
        dimension (over every variable on every level), so a region
        boundary is a chunk boundary in every array it cuts.
    variables_per_task:
        Maximum number of variables per task.  ``None`` keeps all
        variables with the same split dimensions (and levels) together,
        so a worker computes the shared regridding once.

    Returns
    -------
    tuple
        The effective split as ``((dim, step), ...)`` and the planned
        tasks.
    """
    split = dict(split or {})
    if variables_per_task is not None and variables_per_task < 1:
        raise ManifestError("variables_per_task must be a positive integer")

    for dim, step in split.items():
        if dim in SPATIAL_DIMS:
            raise ManifestError(
                f"cannot split along {dim!r}: coarser pyramid levels aggregate "
                "child cells, so a cell region of the finest level is not "
                "chunk-aligned on all levels.  Split along a non-spatial "
                "dimension such as 'time' instead."
            )
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ManifestError(
                f"split size for {dim!r} must be a positive int, got {step!r}"
            )

    all_specs = [spec for lvl in levels for spec in lvl.data_vars]
    effective: list[tuple[str, int]] = []
    boundaries: dict[str, list[tuple[int, int]]] = {}
    for dim, requested in split.items():
        having = [spec for spec in all_specs if dim in spec.dims]
        if not having:
            raise ManifestError(
                f"split dimension {dim!r} is not a dimension of any data variable"
            )
        lengths = {spec.size(dim) for spec in having}
        if len(lengths) != 1:
            detail = ", ".join(f"{spec.name}={spec.size(dim)}" for spec in having[:6])
            raise ManifestError(
                f"dimension {dim!r} has different lengths across variables/levels "
                f"({detail}); all levels must share the split dimension"
            )
        length = lengths.pop()
        unit = math.lcm(*(spec.chunk(dim) for spec in having))
        step = math.ceil(requested / unit) * unit
        if step != requested:
            logger.info(
                "split[%r]=%d rounded up to %d (multiple of the %d-element "
                "write unit shared by all arrays)",
                dim,
                requested,
                step,
                unit,
            )
        step = min(step, max(length, 1))
        effective.append((dim, step))
        boundaries[dim] = [
            (start, min(start + step, length)) for start in range(0, length, step)
        ] or [(0, 0)]

    # Which split dims does every variable carry, and on which levels?
    placements: dict[str, _VariablePlacement] = {}
    for lvl in levels:
        for spec in lvl.data_vars:
            dims = tuple(d for d in split if d in spec.dims)
            placement = placements.setdefault(spec.name, _VariablePlacement(dims))
            if placement.split_dims != dims:
                raise ManifestError(
                    f"variable {spec.name!r} has split dimensions "
                    f"{placement.split_dims} on one level and {dims} on level "
                    f"{lvl.level}"
                )
            placement.levels.append(lvl.level)

    groups: dict[tuple[tuple[str, ...], tuple[int, ...]], list[str]] = {}
    for name, placement in placements.items():
        key = (placement.split_dims, tuple(placement.levels))
        groups.setdefault(key, []).append(name)

    planned: list[_PlannedTask] = []
    for (dims, group_levels), names in groups.items():
        size = variables_per_task or len(names)
        starts = range(0, len(names), size)
        batches = [tuple(names[start:][:size]) for start in starts]
        regions = itertools.product(*(boundaries[d] for d in dims))
        for bounds in regions:
            region: Region = tuple(
                (dim, start, stop) for dim, (start, stop) in zip(dims, bounds)
            )
            for batch in batches:
                planned.append(_PlannedTask(batch, region, group_levels))
    return tuple(effective), planned


# ======================================================================
# Validation
# ======================================================================

#: Identifiers that end up in file names (task ids, manifest ids, store
#: names) must be plain names: no separators, no leading dot.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _fail(where: str, message: str) -> NoReturn:
    raise ManifestError(f"{where}: {message}")


def _set(obj: object, **values: Any) -> None:
    """Assign normalised values on a frozen dataclass."""
    for key, value in values.items():
        object.__setattr__(obj, key, value)


def _loads(text: str | bytes, where: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        _fail(where, f"invalid JSON ({error})")


def _mapping(data: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        _fail(where, f"expected an object, got {type(data).__name__}")
    return data


def _field(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        _fail(where, f"missing field {key!r}")
    return data[key]


def _as_str(value: Any, where: str, name: str, *, safe: bool = False) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, f"{name} must be a non-empty string, got {value!r}")
    if safe and not _SAFE_NAME.fullmatch(value):
        _fail(
            where,
            f"{name} {value!r} may only contain letters, digits, '.', '_' "
            "and '-', and must not start with '.' or '-'",
        )
    return str(value)


def _name(value: Any, where: str, name: str) -> str:
    """Variable names: any string that is a single zarr key."""
    text = _as_str(value, where, name)
    if "/" in text or "\\" in text or text in {".", ".."}:
        _fail(where, f"{name} {text!r} is not a valid array name")
    return text


def _as_int(value: Any, where: str, name: str, *, minimum: int = 0) -> int:
    # bool is an int subclass; True as a shape entry is always a bug.
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(where, f"{name} must be an integer, got {value!r}")
    if value < minimum:
        _fail(where, f"{name} must be >= {minimum}, got {value}")
    return value


def _as_tuple(value: Any, where: str, name: str) -> tuple[Any, ...]:
    # A str is a sequence too; "time" must not become ("t", "i", "m", "e").
    if not isinstance(value, (list, tuple)):
        _fail(where, f"{name} must be a list, got {type(value).__name__}")
    return tuple(value)


def _unique(items: tuple[Any, ...], where: str, name: str) -> None:
    seen: set[Any] = set()
    repeated: set[str] = set()
    for item in items:
        if item in seen:
            repeated.add(str(item))
        seen.add(item)
    duplicates = sorted(repeated)
    if duplicates:
        _fail(where, f"{name} has duplicates {duplicates}")


def _str_tuple(
    value: Any,
    where: str,
    name: str,
    *,
    unique: bool = False,
    nonempty: bool = False,
) -> tuple[str, ...]:
    items = _as_tuple(value, where, name)
    if nonempty and not items:
        _fail(where, f"{name} must not be empty")
    result = tuple(_as_str(item, where, f"{name} entry") for item in items)
    if unique:
        _unique(result, where, name)
    return result


def _int_tuple(
    value: Any,
    where: str,
    name: str,
    *,
    minimum: int,
    unique: bool = False,
    nonempty: bool = False,
) -> tuple[int, ...]:
    items = _as_tuple(value, where, name)
    if nonempty and not items:
        _fail(where, f"{name} must not be empty")
    result = tuple(
        _as_int(item, where, f"{name} entry", minimum=minimum) for item in items
    )
    if unique:
        _unique(result, where, name)
    return result


def _zarr_format(value: Any, where: str) -> Literal[2, 3]:
    if isinstance(value, bool) or value not in (2, 3):
        _fail(where, f"zarr_format must be 2 or 3, got {value!r}")
    return cast(Literal[2, 3], value)


def _check_schema_version(value: Any) -> None:
    if isinstance(value, bool) or value != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema version {value!r} is not supported "
            f"(expected {MANIFEST_SCHEMA_VERSION}); re-initialise the store "
            "with this grid-doctor version"
        )


def _region(value: Any, where: str) -> Region:
    """Accept ``((dim, start, stop), ...)`` or the JSON ``{dim: [start, stop]}``."""
    if isinstance(value, Mapping):
        entries = []
        for dim, bounds in value.items():
            pair = _as_tuple(bounds, where, f"region[{dim!r}]")
            if len(pair) != 2:
                _fail(where, f"region[{dim!r}] must be [start, stop], got {bounds!r}")
            entries.append((dim, *pair))
    else:
        entries = [
            _as_tuple(entry, where, "region entry")
            for entry in _as_tuple(value, where, "region")
        ]
    region: list[tuple[str, int, int]] = []
    for entry in entries:
        if len(entry) != 3:
            _fail(where, f"region entries must be (dim, start, stop), got {entry!r}")
        dim = _as_str(entry[0], where, "region dimension")
        start = _as_int(entry[1], where, f"region start of {dim!r}")
        stop = _as_int(entry[2], where, f"region stop of {dim!r}")
        if stop < start:
            _fail(where, f"region of {dim!r} is [{start}, {stop}): stop < start")
        region.append((dim, start, stop))
    _unique(tuple(dim for dim, _, _ in region), where, "region dimensions")
    return tuple(region)


def _split(value: Any, where: str) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for entry in _as_tuple(value, where, "split"):
        pair = _as_tuple(entry, where, "split entry")
        if len(pair) != 2:
            _fail(where, f"split entries must be (dim, size), got {entry!r}")
        dim = _as_str(pair[0], where, "split dimension")
        if dim in SPATIAL_DIMS:
            _fail(where, f"split along spatial dimension {dim!r} is not supported")
        result.append((dim, _as_int(pair[1], where, f"split[{dim!r}]", minimum=1)))
    _unique(tuple(dim for dim, _ in result), where, "split dimensions")
    return tuple(result)


def _check_consistency(manifest: PyramidManifest) -> None:
    """Check tasks against the level layouts; see ``PyramidManifest.__post_init__``."""
    levels: dict[int, LevelSpec] = {}
    for spec in manifest.levels:
        if spec.level in levels:
            _fail("manifest", f"level {spec.level} is listed twice")
        levels[spec.level] = spec
    _unique(tuple(spec.store for spec in manifest.levels), "manifest", "level stores")
    _unique(tuple(t.task_id for t in manifest.tasks), "manifest", "task ids")

    split_dims = {dim for dim, _ in manifest.split}
    coverage: dict[tuple[int, str], list[Region]] = {}
    for task in manifest.tasks:
        where = f"task {task.task_id!r}"
        for name in ("manifest_id", "path", "zarr_format"):
            if getattr(task, name) != getattr(manifest, name):
                _fail(
                    where,
                    f"{name} {getattr(task, name)!r} differs from the manifest's "
                    f"{getattr(manifest, name)!r}",
                )
        extra = sorted({dim for dim, _, _ in task.region} - split_dims)
        if extra:
            _fail(where, f"region dimensions {extra} are not split dimensions")
        for level in task.levels:
            if level not in levels:
                _fail(where, f"level {level} is not in the manifest")
            for name in task.variables:
                try:
                    array = levels[level].array(name)
                except KeyError:
                    _fail(where, f"level {level} has no data variable {name!r}")
                _check_task_region(where, level, array, task.region)
                coverage.setdefault((level, name), []).append(task.region)

    for level, spec in levels.items():
        for array in spec.data_vars:
            if (level, array.name) not in coverage:
                _fail(
                    "manifest",
                    f"data variable {array.name!r} on level {level} is not "
                    "written by any task",
                )
    for (level, name), regions in coverage.items():
        _check_partition(
            level,
            levels[level].array(name),
            regions,
            [dim for dim, _ in manifest.split],
        )


def _check_task_region(where: str, level: int, spec: ArraySpec, region: Region) -> None:
    """One task's region: inside the array, on write-unit boundaries."""
    for dim, start, stop in region:
        if dim not in spec.dims:
            _fail(where, f"{spec.name!r} on level {level} has no dimension {dim!r}")
        size, unit = spec.size(dim), spec.chunk(dim)
        context = f"{spec.name!r} on level {level} along {dim!r}"
        if stop > size:
            _fail(where, f"region [{start}, {stop}) exceeds length {size} of {context}")
        if start % unit:
            _fail(
                where,
                f"region start {start} is not a multiple of the write unit "
                f"{unit} of {context}",
            )
        if stop % unit and stop != size:
            _fail(
                where,
                f"region stop {stop} is neither a multiple of the write unit "
                f"{unit} nor the end ({size}) of {context}",
            )


def _check_partition(
    level: int,
    spec: ArraySpec,
    regions: list[Region],
    split_order: Sequence[str],
) -> None:
    """Check that the regions writing one array form a grid that tiles it once.

    Each region is a box: one ``[start, stop)`` interval per split
    dimension, the full extent elsewhere.  Regions are read as
    ``{dim: interval}`` mappings, so the order of the entries inside a
    region does not matter, and dimensions are checked in the order of
    the manifest's ``split``.

    Accepted are exactly the grids the planner produces:

    1. all regions split the same dimensions,
    2. per dimension, the distinct intervals tile ``[0, size)`` end to
       end without overlap, and
    3. every combination of those intervals occurs exactly once.

    Together these imply that every element is written by exactly one
    task.  The check is deliberately stricter than "exact tiling": an
    irregular tiling that is not a grid (e.g. different ``lev`` cuts for
    different time ranges) is rejected even though it does not overlap,
    because the planner never produces one.  It is linear in the number
    of tasks.
    """
    where = f"manifest: {spec.name!r} on level {level}"
    boxes = [{dim: (start, stop) for dim, start, stop in region} for region in regions]
    dim_sets = {frozenset(box) for box in boxes}
    if len(dim_sets) > 1:
        raise ManifestError(
            f"{where} is split along different dimensions by different "
            f"tasks: {sorted(sorted(d) for d in dim_sets)}"
        )
    present = dim_sets.pop()
    dims = [dim for dim in split_order if dim in present]
    expected = 1
    for dim in dims:
        size = spec.size(dim)
        intervals = sorted({box[dim] for box in boxes})
        position = 0
        for start, stop in intervals:
            if start < position:
                raise ManifestError(
                    f"{where}: task intervals along {dim!r} overlap at index "
                    f"{start}; regions must form a grid of non-overlapping "
                    "intervals per dimension"
                )
            if start > position:
                raise ManifestError(
                    f"{where}: tasks leave a gap along {dim!r} at index {position}"
                )
            position = stop
        if position != size and not (size == 0 and intervals == [(0, 0)]):
            raise ManifestError(
                f"{where}: tasks cover {dim!r} up to {position}, not its length {size}"
            )
        expected *= len(intervals)
    cells = [tuple(box[dim] for dim in dims) for box in boxes]
    distinct = len(set(cells))
    if distinct != len(cells):
        raise ManifestError(
            f"{where} is written by more than one task in the same region"
        )
    if distinct != expected:
        raise ManifestError(
            f"{where}: {distinct} task region(s) do not form the full grid of "
            f"{expected} region(s); some combinations are missing"
        )


# ======================================================================
# Small helpers
# ======================================================================


def _region_to_json(region: Region) -> dict[str, list[int]]:
    return {dim: [start, stop] for dim, start, stop in region}


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ArraySpec",
    "LevelSpec",
    "ManifestError",
    "PyramidManifest",
    "Region",
    "WriteTask",
    "plan_tasks",
]
