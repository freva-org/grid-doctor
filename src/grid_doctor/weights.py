"""Descriptions of the source, target and weights used for HEALPix remapping.

These lightweight, hashable descriptions identify a remapping problem
(source geometry, HEALPix target grid, method). The source digest names the
weight file and target and method its directories, so equal problems
share weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Final, Literal, cast, get_args

import numpy as np
import xarray as xr

from .metadata import (
    GRID_MAPPING_NAME,
    INDEXING_SCHEMES,
    HealpixGrid,
    IndexingScheme,
)
from .remap_backend import (
    PolygonMesh,
    _get_latlon_names,
    _get_vertex_names,
)
from .types import RemapMethod, SourceKind


@dataclass(frozen=True, eq=False, repr=False)
class SourceDescription:
    """Resolved description of the source representation.

    Produced by [`describe_source`][grid_doctor.remap_backend.describe_source]
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
        source_dims: Source dimensions in flattening order.
        source_mesh: Source mesh stored as compact node/connectivity
            arrays.
        ignore_unmapped: Recommended default for ESMF unmapped
            handling.
        metadata: Metadata that should be persisted into the output
            weight file.
    """

    dataset: xr.Dataset
    geometry: xr.Dataset
    kind: SourceKind
    source_dims: tuple[str, ...]
    source_mesh: PolygonMesh
    ignore_unmapped: bool
    metadata: dict[str, str | int | float | bool]

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

    One ``name=value`` directory per entry of
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
