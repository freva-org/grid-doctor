"""Sparse weight-matrix application engine.

This module provides the numerical core for applying ESMF-style sparse
weight matrices to gridded data.  It is intentionally free of
climate-specific logic (no xarray dataset introspection, no coordinate
name lookups) so that it can be tested and benchmarked in isolation.

Three application strategies are available, selected automatically based
on what is installed and the problem size:

1. **Batched SciPy** (always available) — reshapes all field slices into
   a ``(n_source, n_batch)`` matrix and performs a single sparse matmul.
   This is typically 10–50× faster than per-slice application because
   SciPy delegates to BLAS for the dense right-hand side.

2. **Numba fused kernel** (optional, ``numba`` must be installed) —
   walks the CSR structure once per target row and computes the weighted
   sum, NaN mask, and renormalization support in a single pass.  Halves
   memory traffic compared to two separate sparse matmuls.  Auto-threads
   across target rows with ``parallel=True``.

3. **CuPy / GPU** (optional, ``cupy`` must be installed) — transfers
   the weight matrix and field to the GPU and uses cuSPARSE for the
   batched matmul.  Worthwhile for large target grids (HEALPix level
   10+) with many time steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix

from .types import ApplyBackend, FloatArray, MissingPolicy

if TYPE_CHECKING:
    from scipy.sparse import csr_array

logger = logging.getLogger(__name__)


# ===================================================================
# Weight-file extraction
# ===================================================================


def extract_sparse_weights(
    row: FloatArray,
    col: FloatArray,
    values: FloatArray,
) -> tuple["csr_array", int, int]:
    """Build a CSR weight matrix from COO triplets.

    One-based indices (as produced by ESMF) are converted to
    zero-based automatically.

    Args:
        row: Destination (target) indices.
        col: Source indices.
        values: Weight values.

    Returns:
        ``(matrix, n_target, n_source)``.

    Raises:
        ValueError: When the arrays have mismatched lengths or are
            empty.
    """
    row_i = np.asarray(row, dtype=np.int64).ravel()
    col_i = np.asarray(col, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()

    if row_i.size != values.size or col_i.size != values.size:
        raise ValueError("row, col, and values arrays must have the same length.")
    if row_i.size == 0:
        raise ValueError("Weight arrays are empty.")

    if row_i.min() >= 1 and col.min() >= 1:
        row_i = row_i - 1
        col_i = col_i - 1

    n_target = int(row_i.max()) + 1
    n_source = int(col_i.max()) + 1
    matrix = coo_matrix(
        (values, (row_i, col_i)),
        shape=(n_target, n_source),
        dtype=np.float64,
    )
    return matrix.tocsr(), n_target, n_source


# ===================================================================
# Optional Numba fused kernel
# ===================================================================

_HAS_NUMBA = False

try:
    import numba

    _HAS_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover
    pass

_HAS_CUPY = False

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix

    _HAS_CUPY = True
except ModuleNotFoundError:  # pragma: no cover
    pass


def _build_numba_kernels() -> (
    tuple[
        "numba.core.registry.CPUDispatcher",
        "numba.core.registry.CPUDispatcher",
    ]
    | None
):
    """JIT-compile the fused Numba kernels on first use.

    Returns:
        ``(renormalize_kernel, propagate_kernel)`` or ``None`` when
        Numba is not available.
    """
    if not _HAS_NUMBA:
        return None

    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _numba_renormalize(
        indptr: npt.NDArray[np.int32],
        indices: npt.NDArray[np.int32],
        data: FloatArray,
        values: FloatArray,
        out: FloatArray,
    ) -> None:
        """Fused NaN-aware sparse apply with renormalization."""
        n_target = indptr.size - 1
        for i in numba.prange(n_target):
            wsum = 0.0
            sup = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                c = indices[j]
                w = data[j]
                v = values[c]
                if np.isfinite(v):
                    wsum += w * v
                    sup += w
            out[i] = wsum / sup if sup > 0.0 else np.nan

    @numba.njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
    def _numba_propagate(
        indptr: npt.NDArray[np.int32],
        indices: npt.NDArray[np.int32],
        data: FloatArray,
        values: FloatArray,
        out: FloatArray,
    ) -> None:
        """Fused sparse apply that propagates NaN."""
        n_target = indptr.size - 1
        for i in numba.prange(n_target):
            wsum = 0.0
            has_nan = False
            for j in range(indptr[i], indptr[i + 1]):
                c = indices[j]
                w = data[j]
                v = values[c]
                if np.isfinite(v):
                    wsum += w * v
                else:
                    has_nan = True
            out[i] = np.nan if has_nan else wsum

    return _numba_renormalize, _numba_propagate


# Lazy singleton — compiled on first call.
_NUMBA_KERNELS: (
    tuple[
        "numba.core.registry.CPUDispatcher",
        "numba.core.registry.CPUDispatcher",
    ]
    | None
    | Literal[False]
) = False


def _get_numba_kernels() -> (
    tuple[
        "numba.core.registry.CPUDispatcher",
        "numba.core.registry.CPUDispatcher",
    ]
    | None
):
    """Return the Numba kernels, compiling on first call."""
    global _NUMBA_KERNELS  # noqa: PLW0603
    if _NUMBA_KERNELS is False:
        _NUMBA_KERNELS = _build_numba_kernels()
    return _NUMBA_KERNELS


# ===================================================================
# Single-slice application (Numba path)
# ===================================================================


def _apply_numba_single(
    matrix: csr_matrix,
    values_1d: FloatArray,
    missing_policy: MissingPolicy,
) -> FloatArray:
    """Apply weights to a single 1-D field using the Numba kernel.

    Args:
        matrix: CSR weight matrix ``(n_target, n_source)``.
        values_1d: Source field, shape ``(n_source,)``.
        missing_policy: NaN handling strategy.

    Returns:
        Remapped field, shape ``(n_target,)``.
    """
    kernels = _get_numba_kernels()
    if kernels is None:
        raise RuntimeError("Numba is not available.")

    renorm_kernel, prop_kernel = kernels
    out = np.empty(matrix.shape[0], dtype=np.float64)

    if missing_policy == "propagate":
        prop_kernel(matrix.indptr, matrix.indices, matrix.data, values_1d, out)
    else:
        renorm_kernel(matrix.indptr, matrix.indices, matrix.data, values_1d, out)
    return out


# ===================================================================
# Batched SciPy application
# ===================================================================


def _apply_scipy_batched(
    matrix: csr_matrix,
    values_2d: FloatArray,
    missing_policy: MissingPolicy,
) -> FloatArray:
    """Apply weights to many slices in one batched sparse matmul.

    Args:
        matrix: CSR weight matrix ``(n_target, n_source)``.
        values_2d: Source fields, shape ``(n_batch, n_source)``.
        missing_policy: NaN handling strategy.

    Returns:
        Remapped fields, shape ``(n_batch, n_target)``.
    """
    valid = np.isfinite(values_2d)
    filled = np.where(valid, values_2d, 0.0)

    # Single batched sparse matmul: (n_target, n_source) @ (n_source, n_batch)
    weighted = np.asarray(matrix @ filled.T, dtype=np.float64).T

    if missing_policy == "propagate":
        missing = np.asarray(matrix @ (~valid).astype(np.float64).T, dtype=np.float64).T
        weighted[missing > 0.0] = np.nan
        return weighted

    # Renormalize — try to detect a static NaN mask.
    n_batch = values_2d.shape[0]
    if n_batch > 1 and np.array_equal(valid[0], valid[-1]):
        # Heuristic: if first and last slices share the same mask,
        # assume the mask is static (covers the common land/ocean case).
        support_1d = np.asarray(matrix @ valid[0].astype(np.float64), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted /= support_1d[np.newaxis, :]
        weighted[:, support_1d <= 0.0] = np.nan
    else:
        support = np.asarray(matrix @ valid.astype(np.float64).T, dtype=np.float64).T
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted /= support
        weighted[support <= 0.0] = np.nan

    return weighted


# ===================================================================
# CuPy / GPU application
# ===================================================================


class _GpuMatrixCache:
    """Cache the GPU-side weight matrix to avoid repeated transfers.

    The weight matrix is constant across all slices, so transferring
    it once and reusing it across calls saves significant PCIe
    bandwidth.  The cache is keyed on the ``id()`` of the host-side
    CSR matrix.
    """

    def __init__(self) -> None:
        self._cache: dict[int, Any] = {}

    def get(self, matrix: csr_matrix) -> Any:
        """Return the GPU-side CSR matrix, transferring if needed.

        Args:
            matrix: Host-side SciPy CSR matrix.

        Returns:
            CuPy CSR matrix on the current GPU device.
        """
        key = id(matrix)
        if key not in self._cache:
            self._cache[key] = cp_csr_matrix(matrix)
        return self._cache[key]

    def clear(self) -> None:
        """Drop all cached GPU matrices."""
        self._cache.clear()


_gpu_cache = _GpuMatrixCache() if _HAS_CUPY else None


def _apply_cupy_batched(
    matrix: csr_matrix,
    values_2d: FloatArray,
    missing_policy: MissingPolicy,
) -> FloatArray:
    """Apply weights using cuSPARSE on the GPU.

    The weight matrix is cached on the GPU across calls.  Input data
    is transferred to the GPU, the sparse matmul runs via cuSPARSE,
    and the result is transferred back to host memory.

    Args:
        matrix: Host-side CSR weight matrix
            ``(n_target, n_source)``.
        values_2d: Source fields, shape ``(n_batch, n_source)``.
        missing_policy: NaN handling strategy.

    Returns:
        Remapped fields on the host, shape ``(n_batch, n_target)``.

    Raises:
        RuntimeError: When CuPy is not available.
    """
    if not _HAS_CUPY or _gpu_cache is None:
        raise RuntimeError(
            "CuPy is required for the GPU backend. "
            "Install it with: pip install cupy-cuda12x"
        )

    gpu_matrix = _gpu_cache.get(matrix)
    values_gpu = cp.asarray(values_2d, dtype=cp.float64)

    valid_gpu = cp.isfinite(values_gpu)
    filled_gpu = cp.where(valid_gpu, values_gpu, 0.0)

    # Batched sparse matmul on GPU via cuSPARSE.
    weighted_gpu = (gpu_matrix @ filled_gpu.T).T

    if missing_policy == "propagate":
        missing_gpu = (gpu_matrix @ (~valid_gpu).astype(cp.float64).T).T
        weighted_gpu[missing_gpu > 0.0] = cp.nan
        return cast(FloatArray, cp.asnumpy(weighted_gpu).astype(np.float64))

    # Renormalize — same static-mask optimisation as SciPy path.
    n_batch = values_2d.shape[0]
    if n_batch > 1 and cp.array_equal(valid_gpu[0], valid_gpu[-1]):
        support_1d = gpu_matrix @ valid_gpu[0].astype(cp.float64)
        weighted_gpu /= support_1d[cp.newaxis, :]
        weighted_gpu[:, support_1d <= 0.0] = cp.nan
    else:
        support_gpu = (gpu_matrix @ valid_gpu.astype(cp.float64).T).T
        weighted_gpu /= support_gpu
        weighted_gpu[support_gpu <= 0.0] = cp.nan

    return cast(FloatArray, cp.asnumpy(weighted_gpu).astype(np.float64))


# ===================================================================
# N-dimensional entry point
# ===================================================================


def apply_weights_nd(
    values: npt.NDArray[np.floating],
    *,
    matrix: csr_matrix,
    n_source_dims: int = 1,
    missing_policy: MissingPolicy = "renormalize",
    backend: ApplyBackend = "auto",
) -> FloatArray:
    """Apply a sparse weight matrix to an N-dimensional array.

    The last *n_source_dims* dimensions are treated as the flattened
    source grid.  All preceding dimensions are batch dimensions that
    are preserved in the output.

    Backend selection (``"auto"``):

    - ``"cupy"`` when CuPy is installed — uses cuSPARSE on the GPU
      with a cached device-side weight matrix.
    - ``"numba"`` for a single slice when Numba is installed — fused
      CSR kernel with zero intermediate arrays.
    - ``"scipy"`` otherwise — batched sparse matmul via BLAS.

    Args:
        values: Source data with shape
            ``(*batch_dims, *source_dims)``.
        matrix: CSR weight matrix ``(n_target, n_source)``.
        n_source_dims: Number of trailing dimensions that form the
            source grid.
        missing_policy: NaN handling strategy.
            ``"renormalize"`` skips NaN source cells and rescales
            the remaining weights (target is valid if at least one
            source cell contributed).  ``"propagate"`` sets the
            target to NaN if any contributing source cell is NaN.
        backend: Force a specific application backend.  ``"auto"``
            selects the best available.

    Returns:
        Remapped array with shape ``(*batch_dims, n_target)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    n_target, n_source = matrix.shape

    # Flatten source dims into a single trailing dimension.
    if n_source_dims > 1:
        batch_shape = arr.shape[:-n_source_dims]
        arr = arr.reshape(*batch_shape, -1)
    else:
        batch_shape = arr.shape[:-1]

    if arr.shape[-1] != n_source:
        raise ValueError(
            f"Trailing source size {arr.shape[-1]} does not match "
            f"weight matrix source dimension {n_source}."
        )

    # Reshape to (n_batch, n_source) for the engine.
    n_batch = int(np.prod(batch_shape)) if batch_shape else 1
    flat_2d = arr.reshape(n_batch, n_source)

    # --- Backend dispatch ---
    use_cupy = backend == "cupy" or (backend == "auto" and _HAS_CUPY)
    use_numba = not use_cupy and (
        backend == "numba" or (backend == "auto" and n_batch == 1 and _HAS_NUMBA)
    )

    if use_cupy:
        result_2d = _apply_cupy_batched(matrix, flat_2d, missing_policy)
    elif use_numba and n_batch == 1:
        result_2d = _apply_numba_single(matrix, flat_2d[0], missing_policy)[
            np.newaxis, :
        ]
    elif use_numba and n_batch > 1:
        # Numba is fastest per-slice; loop is still faster than
        # Python-level vectorize because the kernel itself is compiled.
        result_2d = np.empty((n_batch, n_target), dtype=np.float64)
        for i in range(n_batch):
            result_2d[i] = _apply_numba_single(matrix, flat_2d[i], missing_policy)
    else:
        result_2d = _apply_scipy_batched(matrix, flat_2d, missing_policy)

    return result_2d.reshape(*batch_shape, n_target)
