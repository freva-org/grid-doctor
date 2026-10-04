"""NumPy reduction kernels for coarsening nested HEALPix arrays.

All kernels treat the last axis as the cell axis and every leading axis
as a batch axis. In nested ordering, parent cell ``i`` at level ``L``
contains the contiguous children ``factor * i ... factor * (i + 1) - 1``
at level ``L + delta``, with ``factor = 4**delta``.

NaN-aware means are built from *partials*: the sum of all valid
finest-level values under a cell and the number of those values. Partials
can be coarsened step by step without loss, which keeps every step a
factor-4 reduction and therefore chunkable along the cell axis (see
grid-doctor issue #54).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ..types import FloatArray, Int64Array

# Guards ``ceil`` against float round-off, e.g. 0.3 * 10 -> 3.0000000000000004.
_EPS = 1e-9


def min_valid_count(min_valid_fraction: float, n_children: int) -> int:
    """Smallest number of valid children that satisfies the fraction."""
    return max(1, int(np.ceil(min_valid_fraction * n_children - _EPS)))


def _grouped(values: npt.ArrayLike, factor: int) -> npt.NDArray[Any]:
    arr = np.asarray(values)
    n_cells = arr.shape[-1]
    if factor < 1 or n_cells % factor:
        raise ValueError(f"Cannot coarsen {n_cells} cells by a factor of {factor}.")
    return arr.reshape(*arr.shape[:-1], n_cells // factor, factor)


def block_sum(values: npt.ArrayLike, *, factor: int) -> npt.NDArray[Any]:
    """Sum groups of ``factor`` nested children, preserving the dtype."""
    return _grouped(values, factor).sum(axis=-1)


def valid_sums(values: npt.ArrayLike, *, factor: int) -> FloatArray:
    """Sum of the finite children of each parent (NaNs count as zero)."""
    grouped = _grouped(np.asarray(values, dtype=np.float64), factor)
    return cast(FloatArray, np.where(np.isfinite(grouped), grouped, 0.0).sum(axis=-1))


def valid_counts(values: npt.ArrayLike, *, factor: int) -> Int64Array:
    """Count the finite children of each parent."""
    return cast(
        Int64Array,
        np.isfinite(_grouped(values, factor)).sum(axis=-1, dtype=np.int64),
    )


def finalize_mean(
    sums: npt.ArrayLike,
    counts: npt.ArrayLike,
    *,
    n_fine: int,
    min_valid_fraction: float,
) -> FloatArray:
    """Turn partials into means; mask cells below the valid fraction.

    Args:
        sums: Sum of valid finest-level values per cell.
        counts: Number of valid finest-level values per cell.
        n_fine: Number of finest-level cells under each cell.
        min_valid_fraction: Minimum cumulative fraction of valid
            finest-level cells for a cell to be valid.
    """
    s = np.asarray(sums, dtype=np.float64)
    c = np.asarray(counts)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            c >= min_valid_count(min_valid_fraction, n_fine), s / c, np.nan
        )
    return cast(FloatArray, result)


def coarsen_mean(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Average groups of ``factor`` nested children, ignoring NaNs.

    Exact only when ``values`` is the finest level: means of means
    weight every valid parent equally. Chain partials instead
    (see [`MeanPartials`][grid_doctor.pyramid.MeanPartials]).

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of children per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid (finite)
            children for a parent to be valid.

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    return finalize_mean(
        valid_sums(values, factor=factor),
        valid_counts(values, factor=factor),
        n_fine=factor,
        min_valid_fraction=min_valid_fraction,
    )
