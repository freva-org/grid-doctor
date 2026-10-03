"""Special type definitions."""

from collections.abc import Collection, Mapping
from typing import Any, Callable, Dict, Literal, TypedDict

import numpy as np
import numpy.typing as npt

RegridFunc = Callable[[npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]]
regrid_core: RegridFunc

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


class ZarrOptions(TypedDict, total=False):
    """Definitions of possible to_zarr arguments."""

    compute: bool
    mode: Literal["a", "w", "r+"]
    zarr_format: Literal[2, 3]
    consolidated: bool
    encoding: Dict[str, Dict[str, Any]]
