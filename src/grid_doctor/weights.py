"""Descriptions of the source, target and weights used for HEALPix remapping.

These lightweight, hashable descriptions identify a remapping problem
(source geometry, HEALPix target grid, method). The source digest names the
weight file and target and method its directories, so equal problems
share weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Literal,
    cast,
    get_args,
)

import numpy as np
import xarray as xr

from .metadata import (
    GRID_MAPPING_NAME,
    INDEXING_SCHEMES,
    HealpixGrid,
    IndexingScheme,
)
from .metadata import _keys as keys
from .remap_backend import (
    PolygonMesh,
    _classify_source_kind,
    _get_latlon_names,
    _get_vertex_names,
    _looks_global,
    _materialise_spectral_source,
    _source_mesh,
    normalise_dataset,
)
from .types import RemapMethod, SourceKind, SourceUnits

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem
    from fsspec.mapping import FSMap

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False, repr=False)
class SourceDescription:
    """Resolved description of the source representation.

    Built by [`describe_source`][grid_doctor.weights.describe_source]
    and consumed by the weight-generation entry point and
    [`WeightsDescription`][grid_doctor.weights.WeightsDescription].

    Two descriptions are equal when their
    [`digest`][grid_doctor.weights.SourceDescription.digest] is, i.e. when
    they describe the same normalised geometry.

    Attributes:
        dataset: Materialised gridpoint dataset used for weight
            generation.
        geometry: Normalised geometry dataset (degrees, longitudes in
            ``[-180, 180)``) the mesh is built from.
        kind: Source representation class.

    The source mesh and everything derived from it are built lazily on
    first access, so describing a source to look its weights up is cheap.
    """

    dataset: xr.Dataset
    geometry: xr.Dataset
    kind: SourceKind

    @cached_property
    def _mesh_and_dims(self) -> tuple[PolygonMesh, tuple[str, ...]]:
        return _source_mesh(self.geometry, source_units="deg")

    @property
    def source_mesh(self) -> PolygonMesh:
        """Source mesh stored as compact node/connectivity arrays."""
        return self._mesh_and_dims[0]

    @property
    def source_dims(self) -> tuple[str, ...]:
        """Source dimensions in flattening order."""
        return self._mesh_and_dims[1]

    @cached_property
    def looks_global(self) -> bool:
        """Whether the source appears to cover the whole sphere (diagnostic only)."""
        return _looks_global(self.geometry, source_units="deg")

    @property
    def ignore_unmapped(self) -> bool:
        """Recommended default for ESMF unmapped handling; always ``True``.

        The HEALPix target tiles the whole sphere, so a source that does
        not (masked, regional, polar gaps) leaves destination cells
        without overlap. They must become missing values rather than abort
        weight generation; callers can pass ``ignore_unmapped=False`` to
        require full coverage. A source that looks global yet leaves cells
        unmapped may point at a grid problem (see ``looks_global``).
        """
        return True

    @cached_property
    def metadata(self) -> dict[str, str | int | float | bool]:
        """Source attributes to persist into the weight file."""
        return {
            keys.SOURCE_KIND_ATTR: self.kind,
            keys.SOURCE_DIMS_ATTR: json.dumps(list(self.source_dims)),
            keys.SOURCE_SIZE_ATTR: self.source_mesh.face_count,
            keys.SOURCE_UNITS_ATTR: "deg",
            keys.SOURCE_LOOKS_GLOBAL_ATTR: int(self.looks_global),
        }

    @cached_property
    def digest(self) -> str:
        """Hash (sha256 hex digest) of the normalised source geometry.

        Independent of the target and the method, so one grid has one
        digest at every level, indexing scheme and method.
        """
        return _source_hash(self.geometry)

    def __hash__(self) -> int:  # noqa: D105
        return hash(self.digest)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, SourceDescription) and self.digest == other.digest


def describe_source(
    source: xr.Dataset | str | Path,
    *,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    source_units: SourceUnits = "auto",
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
    workdir: Path | None = None,
) -> SourceDescription:
    """Describe a source for HEALPix weight generation.

    Normalises the source geometry and classifies the grid. The mesh
    and the weight-file metadata are built lazily, so describing a
    source to look its weights up does not mesh it.

    Args:
        source: Source dataset or source path. A path is mainly
            useful for spectral inputs that need an external transform.
        grid: Optional grid dataset that supplies geometry when the
            data file itself only contains values on dimensions such
            as ``cell``.
        source_kind: Explicit source representation. Use ``"auto"``
            to infer it from the available coordinates.
        source_units: Angular unit convention of the source
            coordinates, used to normalise the geometry to degrees.
        spectral_transform_command: External command used when
            ``source_kind="spectral"``. The command may use
            ``{input}`` and ``{output}`` placeholders.
        workdir: Working directory for the spectral transform;
            a temporary directory when omitted.

    Raises:
        ValueError: When the geometry cannot be normalised (see
            ``normalise_dataset``). In particular, cell centres with
            NaN coordinates (masked grids) are rejected.
    """
    dataset = source if isinstance(source, xr.Dataset) else xr.load_dataset(source)
    kind = source_kind
    if kind == "spectral":
        with tempfile.TemporaryDirectory(prefix="grid_doctor_source_") as scratch:
            dataset = _materialise_spectral_source(  # loads eagerly, scratch can go
                source,
                transform_command=spectral_transform_command,
                workdir=workdir or Path(scratch),
            )
        kind = "auto"

    # Only the geometry source is normalised: grid if given, else dataset.
    geometry = normalise_dataset(grid if grid is not None else dataset, source_units)
    kind = _classify_source_kind(geometry, explicit_kind=kind)
    return SourceDescription(dataset=dataset, geometry=geometry, kind=kind)


@dataclass(frozen=True, eq=False, repr=False)
class WeightsDescription:
    """Description of the weights of a remapping problem.

    Two descriptions are equal when they describe the same source
    geometry (by [`digest`][grid_doctor.weights.SourceDescription.digest]),
    target and method.

    Attributes:
        source: Source description.
        target: HEALPix target grid.
        method: Remapping method.
    """

    source: SourceDescription
    target: HealpixGrid
    method: RemapMethod

    @property
    def ref(self) -> WeightsRef:
        """Reference (relative path) under which these weights are stored."""
        return WeightsRef(self.method, self.target.indexing_scheme, self.target.level, self.source.digest)

    def _identity(self) -> tuple[str, HealpixGrid, RemapMethod]:
        return (self.source.digest, self.target, self.method)

    def __hash__(self) -> int:  # noqa: D105
        return hash(self._identity())

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, WeightsDescription) and self._identity() == other._identity()


def _source_hash(ds: xr.Dataset) -> str:
    """Hash the cell centres and vertices of *ds*.

    *ds* is expected to be normalised (see
    [`normalise_dataset`][grid_doctor.remap_backend.normalise_dataset]).
    Variable and dimension names and dtypes do not enter the hash; values,
    shapes, the order of the cells and which dimensions the coordinates
    share do (separate ``lat``/``lon`` dims form a regular grid, a shared
    ``cell`` dim does not).
    """
    names = [_get_latlon_names(ds)]
    try:
        names.append(_get_vertex_names(ds))
    except ValueError:
        pass  # vertices are optional

    # Dimensions as positions of first appearance: name-free, but keeps
    # which coordinates share a dimension.
    dim_ids: dict[str, int] = {}
    digest = hashlib.sha256()
    for lat_name, lon_name in names:
        for name in (lat_name, lon_name):
            dims = tuple(dim_ids.setdefault(str(d), len(dim_ids)) for d in ds[name].dims)
            values = np.asarray(ds[name].values, dtype=np.float64) + 0.0  # -0.0 -> 0.0
            values[np.isnan(values)] = np.nan  # one NaN bit pattern
            values = np.ascontiguousarray(values, dtype="<f8")  # same bytes on any host
            digest.update(f"dims={dims};shape={values.shape};".encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


WEIGHTS_VERSION: Final = 1
"""Version of the [`WeightsRef`][grid_doctor.weights.WeightsRef] path scheme.

Not the grid-doctor package version: bump it only when the path layout changes.
"""


@dataclass(frozen=True, slots=True)
class WeightsRef:
    """Reference to a weight file, as a Hive-style relative path.

    Lowercase throughout. One ``name=value`` directory per entry of
    [`HIERARCHY`][grid_doctor.weights.WeightsRef.HIERARCHY], followed by the
    source digest as file name::

        version=1/grid=healpix/ellipsoid=sphere/indexing_scheme=nested/level=10/method=conservative/<key>.nc

    Every part that changes the weights is a directory, so the path is
    unique per weight file and the fields used for searching can be read
    from a listing without opening files. ``version`` is the version of the
    path scheme itself.

    Attributes:
        method: Remapping method.
        indexing_scheme: HEALPix indexing scheme.
        level: HEALPix level.
        key: Source digest (sha256 hex digest); see
            [`SourceDescription.digest`][grid_doctor.weights.SourceDescription.digest].
        grid: Target grid; always ``"healpix"``.
        ellipsoid: Ellipsoid of the HEALPix cell vertices; always
            ``"sphere"`` until other ellipsoids are supported.
        version: Version of the path scheme.
    """

    HIERARCHY: ClassVar[tuple[str, ...]] = (
        "version",
        "grid",
        "ellipsoid",
        "indexing_scheme",
        "level",
        "method",
    )
    """Directory levels, outermost first."""

    method: RemapMethod
    indexing_scheme: IndexingScheme
    level: int
    key: str
    grid: str = GRID_MAPPING_NAME
    ellipsoid: Literal["sphere"] = "sphere"
    version: int = WEIGHTS_VERSION

    def __post_init__(self) -> None:
        """Lowercase every string field, so one set of weights has one path."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, value.lower())

    @property
    def ref(self) -> str:
        """Relative path of the weight file."""
        dirs = (f"{name}={getattr(self, name)}" for name in self.HIERARCHY)
        return "/".join((*dirs, f"{self.key}.nc"))

    @classmethod
    def parse(cls, ref: str) -> WeightsRef:
        """Parse a relative reference path.

        Raises:
            ValueError: If *ref* does not follow the layout.
        """
        *dirs, file = ref.strip("/").split("/")
        pairs = [d.partition("=") for d in dirs]
        values = {name: value for name, _, value in pairs}
        if (
            tuple(name for name, _, _ in pairs) != cls.HIERARCHY
            or any(sep != "=" for _, sep, _ in pairs)
            or values["version"] != str(WEIGHTS_VERSION)
            or values["grid"] != GRID_MAPPING_NAME
            or values["ellipsoid"] != "sphere"
            or values["method"] not in get_args(RemapMethod)
            or values["indexing_scheme"] not in INDEXING_SCHEMES
            or not values["level"].isdigit()
            or not file.endswith(".nc")
        ):
            raise ValueError(f"Not a weight reference: {ref!r}")
        return cls(
            method=cast(RemapMethod, values["method"]),
            indexing_scheme=cast(IndexingScheme, values["indexing_scheme"]),
            level=int(values["level"]),
            key=file[: -len(".nc")],
        )


WEIGHTS_STORE_ENV: Final = "GRID_DOCTOR_WEIGHTS_STORE"
"""Environment variable naming the default [`WeightsStore`][grid_doctor.weights.WeightsStore] location."""

_LOCAL_PROTOCOLS: Final = frozenset({"file", "local"})
_SUPPORTED_PROTOCOLS: Final = _LOCAL_PROTOCOLS | {"memory"}


def local_weights_dir() -> Path:
    """Local directory holding weight files in the [`WeightsRef`][grid_doctor.weights.WeightsRef] layout.

    ``<cache_dir()>/weights``. Generated weights land here by default and
    remote [`WeightsStore`][grid_doctor.weights.WeightsStore]s download
    into it, so one set of weights has one local path.
    """
    from .utils import cache_dir

    return cache_dir() / "weights"


class WeightsStore:
    """Store of HEALPix weight files, laid out by [`WeightsRef`][grid_doctor.weights.WeightsRef].

    Callers pass a root; everything below it is grid-doctor's layout. The
    root is a local (POSIX) directory, or an in-memory fsspec filesystem
    (``memory://``, mainly for tests). ESMF and the weight readers need
    local files, so files of in-memory stores are copied once into a local
    cache that mirrors the same layout.

    Args:
        location: Root of the store: a local path, a ``file://`` or
            ``memory://`` URL, or an fsspec mapper of either. Defaults to the
            ``GRID_DOCTOR_WEIGHTS_STORE`` environment variable, else
            ``<cache_dir()>/weights``.
        cache_dir: Local directory for copies of in-memory weights and for
            weights computed while the store is read-only. Defaults to
            ``<cache_dir()>/weights``.
        read_only: Forbid [`publish`][grid_doctor.weights.WeightsStore.publish].
            When omitted, local stores are read-only if their directory is
            not writable.

    Raises:
        ValueError: For filesystems other than local and in-memory ones.

    Examples:
        ```python
        store = WeightsStore("/work/ab1234/weights")
        if store.exists(desc):
            path = store.fetch(desc)
        ```
    """

    def __init__(
        self,
        location: str | Path | FSMap | None = None,
        *,
        cache_dir: str | Path | None = None,
        read_only: bool | None = None,
    ) -> None:  # noqa: D107
        from fsspec.core import url_to_fs
        from fsspec.mapping import FSMap

        if location is None:
            location = os.environ.get(WEIGHTS_STORE_ENV) or local_weights_dir()
        fs: AbstractFileSystem
        if isinstance(location, FSMap):
            fs, root = location.fs, location.root
        elif isinstance(location, (str, os.PathLike)):
            fs, root = url_to_fs(str(location))
        else:
            raise TypeError(
                f"Unsupported weights store location {type(location).__name__!r}; "
                "expected a path, an fsspec URL or an fsspec mapper."
            )
        self.fs = fs
        self.root = str(root).rstrip("/")
        protocols = fs.protocol
        protocols = (protocols,) if isinstance(protocols, str) else tuple(protocols)
        if not _SUPPORTED_PROTOCOLS.intersection(protocols):
            raise ValueError(
                f"Unsupported weights store filesystem {protocols[0]!r}; "
                f"expected one of {sorted(_SUPPORTED_PROTOCOLS)}."
            )
        self.is_local = bool(_LOCAL_PROTOCOLS.intersection(protocols))
        self._protocol = protocols[0]
        if read_only is None:
            read_only = self.is_local and not _is_writable(Path(self.root))
        self.read_only = read_only
        self.cache_dir = Path(cache_dir) if cache_dir is not None else local_weights_dir()

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}({self.url_root!r}, read_only={self.read_only})"

    @property
    def url_root(self) -> str:
        """URL of the store root."""
        return f"{self._protocol}://{self.root}"

    @staticmethod
    def _ref(item: WeightsDescription | WeightsRef | str) -> WeightsRef:
        if isinstance(item, WeightsDescription):
            return item.ref
        if isinstance(item, WeightsRef):
            return item
        return WeightsRef.parse(item)

    def _path(self, item: WeightsDescription | WeightsRef | str) -> str:
        return f"{self.root}/{self._ref(item).ref}"

    def url(self, item: WeightsDescription | WeightsRef | str) -> str:
        """Return the full URL (``file://`` or ``memory://``) of a weight file."""
        return f"{self._protocol}://{self._path(item)}"

    def local_path(self, item: WeightsDescription | WeightsRef | str) -> Path:
        """Return where the local copy of a weight file lives.

        The file itself for local stores, its place in the local cache
        for in-memory ones. The file need not exist.
        """
        if self.is_local:
            return Path(self._path(item))
        return self.cache_dir / self._ref(item).ref

    def exists(self, item: WeightsDescription | WeightsRef | str) -> bool:
        """Return whether the store holds the weight file."""
        return bool(self.fs.exists(self._path(item)))

    def fetch(self, item: WeightsDescription | WeightsRef | str) -> Path:
        """Return a local path to the weight file, downloading it if needed.

        Raises:
            FileNotFoundError: If the store does not hold the file.
        """
        path = self._path(item)
        if self.is_local:
            if not self.fs.exists(path):
                raise FileNotFoundError(self.url(item))
            return Path(path)
        local = self.local_path(item)
        if local.exists():
            return local
        if not self.fs.exists(path):
            raise FileNotFoundError(self.url(item))
        logger.info("Copying weight file %s", self.url(item))
        _atomic_write(local, lambda tmp: self.fs.get_file(path, tmp))
        return local

    def publish(self, desc: WeightsDescription, local_file: str | Path) -> WeightsRef:
        """Store *local_file* as the weights of *desc* and return its reference.

        Readers never see a partial file: local stores write a temporary
        file and rename it.

        Raises:
            PermissionError: If the store is read-only.
        """
        if self.read_only:
            raise PermissionError(f"{self!r} is read-only")
        ref = desc.ref
        path = self._path(ref)
        logger.info("Publishing weight file %s", self.url(ref))
        if self.is_local:
            _atomic_write(Path(path), lambda tmp: shutil.copyfile(local_file, tmp))
        else:
            self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
            self.fs.put_file(str(local_file), path)
        return ref

    def fetch_or_compute(self, desc: WeightsDescription, compute: Callable[[Path], Path]) -> Path:
        """Return a local copy of the weights of *desc*, computing them if needed.

        Looks for the local copy, then the store; otherwise calls
        ``compute(path)`` and publishes the result to a writable in-memory
        store. Weights for a read-only store are computed into and kept in
        ``cache_dir`` only.
        """
        local = self.local_path(desc)
        cached = self.cache_dir / desc.ref.ref
        for path in dict.fromkeys((local, cached)):
            if path.exists():
                return path
        if self.exists(desc):
            return self.fetch(desc)
        target = local if (self.is_local and not self.read_only) else cached
        logger.info("Computing HEALPix weight file %s", target)
        target = compute(target)
        if not self.is_local and not self.read_only:
            self.publish(desc, target)
        return target

    def find(self, **filters: Any) -> list[WeightsRef]:
        """List the weight files whose reference fields match *filters*.

        Args:
            **filters: Field values of [`WeightsRef`][grid_doctor.weights.WeightsRef],
                e.g. ``level``, ``method``, ``indexing_scheme`` or ``key``.

        Raises:
            TypeError: For unknown filter names.

        Examples:
            ```python
            store.find(level=10, method="conservative")
            ```
        """
        fields = set(WeightsRef.__dataclass_fields__)
        unknown = set(filters) - fields
        if unknown:
            raise TypeError(f"Unknown filters: {sorted(unknown)}; expected {sorted(fields)}")
        filters = {k: v.lower() if isinstance(v, str) else v for k, v in filters.items()}
        dirs = (f"{name}={filters.get(name, '*')}" for name in WeightsRef.HIERARCHY)
        pattern = "/".join((*dirs, f"{filters.get('key', '*')}.nc"))
        found = []
        for path in sorted(self.fs.glob(f"{self.root}/{pattern}")):
            try:
                found.append(WeightsRef.parse(str(path).removeprefix(self.root)))
            except ValueError:
                continue
        return found


def _is_writable(path: Path) -> bool:
    """Return whether *path*, or its nearest existing parent, is writable."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return os.access(path, os.W_OK)


def _atomic_write(target: Path, write: Callable[[str], object]) -> None:
    """Call ``write(tmp)`` on a temporary file next to *target*, then rename it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
