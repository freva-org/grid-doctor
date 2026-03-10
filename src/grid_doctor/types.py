"""Special type definitions."""

from typing import Any, Callable, Literal, TypedDict

import numpy as np
import numpy.typing as npt

RegridFunc = Callable[
    [npt.NDArray[np.floating[Any]]], npt.NDArray[np.floating[Any]]
]
regrid_core: RegridFunc


class ZarrOptions(TypedDict, total=False):
    """Definitions of possible to_zarr arguments."""

    compute: bool
    mode: Literal["a", "w", "r+"]
    zarr_format: Literal[2, 3]
    consolidated: bool
