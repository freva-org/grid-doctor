"""Descriptions of the source, target and weights used for HEALPix remapping.

These lightweight, hashable descriptions identify the parts of a
remapping problem. The source digest identifies a grid independently of
the HEALPix target and the method.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import xarray as xr

from .remap_backend import (
    PolygonMesh,
    _get_latlon_names,
    _get_vertex_names,
)
from .types import SourceKind


@dataclass(frozen=True, eq=False, repr=False)
class SourceDescription:
    """Resolved description of the source representation.

    Produced by [`describe_source`][grid_doctor.remap_backend.describe_source]
    and consumed by the weight-generation entry point.

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
