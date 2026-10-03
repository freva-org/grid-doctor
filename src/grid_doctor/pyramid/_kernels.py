"""NumPy reduction kernels for coarsening nested HEALPix arrays.

All kernels treat the last axis as the cell axis and every leading axis
as a batch axis. In nested ordering, parent cell ``i`` at level ``L``
contains the contiguous children ``factor * i ... factor * (i + 1) - 1``
at level ``L + delta``, with ``factor = 4**delta``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ..types import FloatArray

# Guards ``ceil`` against float round-off, e.g. 0.3 * 10 -> 3.0000000000000004.
_EPS = 1e-9


def min_valid_count(min_valid_fraction: float, n_children: int) -> int:
    """Smallest number of valid children that satisfies the fraction."""
    return max(1, int(np.ceil(min_valid_fraction * n_children - _EPS)))


def coarsen_mean(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Average groups of ``factor`` nested children, ignoring NaNs.

    The result is the mean over all valid input cells of each parent.
    It is exact only when ``values`` is the finest level. Feeding the
    output of a previous call back in weights every valid parent
    equally, regardless of how many valid cells it was built from,
    and applies ``min_valid_fraction`` per step instead of
    cumulatively. Coarsen from the finest level instead
    (see grid-doctor issue #54).

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of children per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid (finite)
            children for a parent to be valid.

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    n_cells = arr.shape[-1]
    if factor < 1 or n_cells % factor:
        raise ValueError(f"Cannot coarsen {n_cells} cells by a factor of {factor}.")
    grouped = arr.reshape(*arr.shape[:-1], n_cells // factor, factor)
    valid = np.isfinite(grouped)
    counts = valid.sum(axis=-1)
    sums = np.where(valid, grouped, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            counts >= min_valid_count(min_valid_fraction, factor),
            sums / counts,
            np.nan,
        )
    return cast(FloatArray, result)
