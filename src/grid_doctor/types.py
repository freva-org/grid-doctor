"""Special type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

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


WriteMode = Literal["w", "a", "r+", "auto"]
"""Zarr write mode.  ``"auto"`` inspects the existing store and infers the
correct combination of operations to perform."""


@dataclass
class WritePlan:
    """Describes what operations are needed to bring a Zarr store up to date.

    Produced by `grid_doctor.s3._inspect_store` and consumed by
    `grid_doctor.s3._execute_write_plan`.
    """

    mode: Literal["w", "a", "r+"]
    """Base write mode resolved for this store.

    ``"w"`` is used for both the "store absent" and "store inconsistent"
    cases — in the latter, ``_inspect_store`` logs a warning and falls back
    to a full overwrite.
    """

    new_vars: List[str]
    """Data variables present in the incoming dataset but absent from the store."""

    existing_vars: List[str]
    """Data variables present in both the incoming dataset and the store."""

    n_existing_time: Optional[int]
    """Number of time steps already written to the store, or ``None`` if the
    dataset has no time dimension or the store does not yet exist."""

    append_time: bool
    """``True`` when the incoming dataset has more time steps than the store."""


class ZarrOptions(TypedDict, total=False):
    """Definitions of possible to_zarr arguments."""

    compute: bool
    mode: Literal["a", "w", "r+"]
    zarr_format: Literal[2, 3]
    consolidated: bool
    encoding: Dict[str, Dict[str, Any]]
