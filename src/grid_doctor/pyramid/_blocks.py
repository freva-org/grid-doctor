"""Chunk-aware coarsening along the HEALPix cell axis.

Every function here accepts NumPy or dask arrays with the cell axis last.
Dask arrays are processed block by block along ``cell``: nested parents
never straddle a chunk boundary as long as every chunk is a multiple of
the reduction factor, so no step needs the whole globe in one task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np

from ._kernels import block_sum, finalize_mean, valid_counts, valid_sums

Array = Any
"""A NumPy or dask array with the HEALPix cell axis last."""

STEP = 4
"""Children per parent for one HEALPix level."""


def is_dask(arr: Array) -> bool:
    """Whether *arr* is a dask array."""
    return isinstance(arr, da.Array)


def _round_down(value: int, multiple: int) -> int:
    return max(multiple, value - value % multiple)


def rechunk_cells(arr: Array, cell_chunk: int | None) -> Array:
    """Give *arr* uniform cell chunks of ``cell_chunk`` (a multiple of 4).

    NumPy arrays and ``cell_chunk=None`` are returned unchanged.
    """
    if not is_dask(arr) or cell_chunk is None:
        return arr
    size = min(_round_down(cell_chunk, STEP), arr.shape[-1])
    if all(c == size for c in arr.chunks[-1][:-1]) and arr.chunks[-1][-1] <= size:
        return arr
    return arr.rechunk({arr.ndim - 1: size})


def map_cells(
    func: Callable[..., np.ndarray],
    arr: Array,
    *,
    factor: int,
    dtype: Any,
    **kwargs: Any,
) -> Array:
    """Apply a reducing kernel ``func(block, factor=...)`` along cells.

    Each output block covers exactly the parents of its input block.
    Dask chunks that are not multiples of *factor* are realigned first.
    """
    if not is_dask(arr):
        return func(arr, factor=factor, **kwargs)
    if any(c % factor for c in arr.chunks[-1]):
        arr = rechunk_cells(arr, _round_down(arr.chunks[-1][0], factor))
    out_chunks = (*arr.chunks[:-1], tuple(c // factor for c in arr.chunks[-1]))
    return arr.map_blocks(
        func,
        factor=factor,
        chunks=out_chunks,
        dtype=dtype,
        meta=np.empty((0,) * arr.ndim, dtype=dtype),
        **kwargs,
    )


@dataclass(frozen=True)
class MeanPartials:
    """Running totals that make chained NaN-aware means exact.

    Attributes:
        sums: Sum of valid finest-level values under each cell.
        counts: Number of valid finest-level values under each cell.
        n_fine: Finest-level cells under each cell (``4**steps``).
    """

    sums: Array
    counts: Array
    n_fine: int

    @classmethod
    def from_values(cls, values: Array) -> MeanPartials:
        """Build partials one level coarser than *values* (the finest level)."""
        return cls(
            sums=map_cells(valid_sums, values, factor=STEP, dtype=np.float64),
            counts=map_cells(valid_counts, values, factor=STEP, dtype=np.int64),
            n_fine=STEP,
        )

    def coarsen(self) -> MeanPartials:
        """Coarsen the partials by one level."""
        return MeanPartials(
            sums=map_cells(block_sum, self.sums, factor=STEP, dtype=np.float64),
            counts=map_cells(block_sum, self.counts, factor=STEP, dtype=np.int64),
            n_fine=self.n_fine * STEP,
        )

    def rechunk(self, cell_chunk: int | None) -> MeanPartials:
        """Merge shrinking chunks back to ``cell_chunk`` cells."""
        return MeanPartials(
            sums=rechunk_cells(self.sums, cell_chunk),
            counts=rechunk_cells(self.counts, cell_chunk),
            n_fine=self.n_fine,
        )

    def mean(self, min_valid_fraction: float) -> Array:
        """Mean over valid finest-level cells, masked by cumulative fraction."""
        if is_dask(self.sums):
            return da.map_blocks(
                finalize_mean,
                self.sums,
                self.counts,
                dtype=np.float64,
                meta=np.empty((0,) * self.sums.ndim, dtype=np.float64),
                n_fine=self.n_fine,
                min_valid_fraction=min_valid_fraction,
            )
        return finalize_mean(
            self.sums,
            self.counts,
            n_fine=self.n_fine,
            min_valid_fraction=min_valid_fraction,
        )

    def valid_fraction(self) -> Array:
        """Fraction of finest-level cells under each cell that are valid."""
        return self.counts / self.n_fine


def coarsen_mean_steps(
    values: Array,
    steps: int,
    *,
    min_valid_fraction: float,
    cell_chunk: int | None = None,
) -> Array:
    """NaN-aware mean of *values* coarsened by ``4**steps``.

    Exact for finest-level input, chunked along cells for dask input.
    """
    partials = MeanPartials.from_values(values).rechunk(cell_chunk)
    for _ in range(steps - 1):
        partials = partials.coarsen().rechunk(cell_chunk)
    return partials.mean(min_valid_fraction)


def coarsen_mode_steps(
    values: Array,
    steps: int,
    *,
    kernel: Callable[..., np.ndarray],
    min_valid_fraction: float,
    cell_chunk: int | None = None,
) -> Array:
    """Mode coarsening by ``4**steps``, one factor-4 step at a time."""
    out = values
    for _ in range(steps):
        out = map_cells(
            kernel,
            out,
            factor=STEP,
            dtype=np.float64,
            min_valid_fraction=min_valid_fraction,
        )
        out = rechunk_cells(out, cell_chunk)
    return out
