"""Special type definitions.

Shared aliases, so that signatures say what a value is instead of
``Any``.  The only ``Any`` left in grid-doctor's own annotations is the
precision parameter of NumPy's generic float type (``FloatingArray``)
and keyword arguments passed through unchanged to third-party
functions.
"""

from collections.abc import Callable, Collection, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import xarray as xr
import zarr
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

FloatingArray: TypeAlias = npt.NDArray[np.floating[Any]]
"""NumPy array of any float precision (``float16`` ... ``float128``)."""

GenericArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
"""NumPy array of any dtype and shape (without ``Any`` in its type)."""

BoolArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
"""Boolean NumPy array of any shape."""

NumberArray: TypeAlias = npt.NDArray[np.number[Any]]
"""NumPy array of any numeric dtype."""

IndexSlice: TypeAlias = "slice[int | None, int | None, int | None]"
"""An index ``slice`` as used by ``isel`` and zarr regions."""

RegridFunc = Callable[[FloatingArray], FloatingArray]
"""A function regridding one block of values."""

# ---------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------

S3Options: TypeAlias = Mapping[str, object]
"""Keyword arguments for :class:`s3fs.S3FileSystem`, e.g. ``key``,
``secret`` and ``endpoint_url``.  Never stored in a manifest."""

ZarrStore: TypeAlias = str | MutableMapping[str, bytes]
"""Where a zarr store lives: a local path or an fsspec mapping (S3)."""

ZarrArray: TypeAlias = zarr.Array[ArrayV2Metadata] | zarr.Array[ArrayV3Metadata]
"""A zarr array of format 2 or 3."""

# ---------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------

VarEncoding: TypeAlias = Mapping[str, object]
"""Encoding of one variable as accepted by ``to_zarr``, e.g.
``{"dtype": "int16", "scale_factor": 0.01, "chunks": (24, 4096)}``.

Kept open (not a ``TypedDict``) because zarr and xarray accept many
codec-specific keys (``compressors``, ``filters``, ``serializer``,
``shards``, ...)."""

LevelEncoding: TypeAlias = Mapping[str, VarEncoding]
"""Encodings of one dataset (pyramid level), keyed by variable name."""

PyramidEncoding: TypeAlias = Mapping[int, LevelEncoding]
"""Encodings of a whole pyramid, keyed by HEALPix level."""

EncodingDict: TypeAlias = dict[str, dict[str, object]]
"""A mutable per-variable encoding mapping, as built internally."""

# ---------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------

JSONValue: TypeAlias = (
    "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
)
"""Any value ``json.loads`` can return."""

RemapMethod = Literal["nearest", "conservative"]
"""Supported weight-generation methods."""

SourceUnits = Literal["auto", "deg", "rad"]
"""Angular unit convention for source coordinates."""

SourceKind = Literal["auto", "regular", "curvilinear", "unstructured", "spectral"]
"""Explicit source grid classification."""

CoarsenMode = Literal["mean", "mode", "auto"]
"""Coarsening strategy for HEALPix pyramid construction."""

ValidFraction = (
    bool | Literal["static"] | Collection[str] | Mapping[str, bool | Literal["static"]]
)
"""Which variables get a ``<name>_valid_fraction`` companion, and its shape.

``True``/``"static"`` apply to every variable with a ``cell`` dimension;
a collection of names selects variables (full shape); a mapping sets the
shape per variable.  ``"static"`` stores the fraction of the first slice
along all non-cell dimensions only, for masks that never change."""

BinAgg = Literal["mean", "mode", "min", "max", "count"]
"""Per-cell aggregation methods for point binning."""

FloatArray = npt.NDArray[np.float64]
"""Shorthand for a float64 NumPy array."""

IntArray = npt.NDArray[np.int32]
"""Shorthand for a int32 NumPy array."""

Int64Array = npt.NDArray[np.int64]
"""Shorthand for a int64 NumPy array."""

MissingPolicy = Literal["renormalize", "propagate"]
"""Missing-value handling strategy for weight application."""

ApplyBackend = Literal["auto", "scipy", "numba", "cupy"]
"""Which application backend to use."""


class WeightOptions(TypedDict, total=False):
    """Options for generating weight files.

    Shared by
    [`compute_healpix_weights`][grid_doctor.compute_healpix_weights] and
    [`regrid_to_healpix`][grid_doctor.regrid_to_healpix].  Functions that pass them through unchanged (e.g.
    [`cached_weights`][grid_doctor.utils.cached_weights]) accept them as
    keyword arguments; see ``compute_healpix_weights`` for what each one
    does.
    """

    grid: xr.Dataset | None
    source_kind: SourceKind
    ignore_unmapped: bool | None
    large_file: bool
    prefer_offline: bool | None
    nproc: int
    esmf_regrid_weightgen: str
    keep_intermediates: bool
    workdir: str | Path | None
    spectral_transform_command: list[str] | tuple[str, ...] | None


class RegridOptions(WeightOptions, total=False):
    """Keyword options of [`regrid_to_healpix`][grid_doctor.regrid_to_healpix].

    Functions that build pyramids pass them through unchanged; see
    ``regrid_to_healpix`` for what each one does.
    """

    nest: bool
    method: RemapMethod
    source_units: SourceUnits
    weights_path: str | Path | None
    missing_policy: MissingPolicy
    backend: ApplyBackend
    cell_chunks: int | Literal["auto"] | None


class ZarrOptions(TypedDict, total=False):
    """Definitions of possible to_zarr arguments."""

    compute: bool
    mode: Literal["a", "w", "r+"]
    zarr_format: Literal[2, 3]
    consolidated: bool
    encoding: Mapping[str, Mapping[str, object]]
