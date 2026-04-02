"""Special type definitions."""

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
