"""Tests for `grid_doctor.remap_apply`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from grid_doctor.remap_apply import (
    ApplyBackend,
    _apply_scipy_batched,
    apply_weights_nd,
    extract_sparse_weights,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def simple_matrix() -> csr_matrix:
    """2-target, 3-source weight matrix.

    target 0 = 0.5 * src[0] + 0.5 * src[1]
    target 1 = 1.0 * src[2]
    """
    return csr_matrix(
        np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    )


@pytest.fixture()
def identity_matrix() -> csr_matrix:
    """3×3 identity — pass-through."""
    return csr_matrix(np.eye(3, dtype=np.float64))


# ===================================================================
# extract_sparse_weights
# ===================================================================


class TestExtractSparseWeights:
    def test_one_based_to_zero_based(self) -> None:
        row = np.array([1, 2], dtype=np.int32)
        col = np.array([1, 2], dtype=np.int32)
        vals = np.array([0.25, 0.75], dtype=np.float64)
        matrix, n_target, n_source = extract_sparse_weights(row, col, vals)
        assert n_target == 2
        assert n_source == 2
        np.testing.assert_allclose(matrix.toarray(), np.diag([0.25, 0.75]))

    def test_zero_based_stays(self) -> None:
        row = np.array([0, 1], dtype=np.int32)
        col = np.array([0, 1], dtype=np.int32)
        vals = np.array([1.0, 1.0], dtype=np.float64)
        matrix, n_target, n_source = extract_sparse_weights(row, col, vals)
        assert n_target == 2
        assert n_source == 2
        np.testing.assert_allclose(matrix.toarray(), np.eye(2))

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            extract_sparse_weights(
                np.array([1, 2]), np.array([1]), np.array([1.0, 2.0])
            )

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            extract_sparse_weights(
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
            )

    def test_scrip_style_multiple_entries_per_row(self) -> None:
        row = np.array([1, 1, 2], dtype=np.int32)
        col = np.array([1, 2, 2], dtype=np.int32)
        vals = np.array([0.25, 0.75, 1.0], dtype=np.float64)
        matrix, n_target, n_source = extract_sparse_weights(row, col, vals)
        assert (n_target, n_source) == (2, 2)
        np.testing.assert_allclose(
            matrix.toarray(), np.array([[0.25, 0.75], [0.0, 1.0]])
        )


# ===================================================================
# _apply_scipy_batched
# ===================================================================


class TestApplyScipyBatched:
    def test_clean_data_renormalize(self, simple_matrix: csr_matrix) -> None:
        values = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        np.testing.assert_allclose(result, [[3.0, 6.0]])

    def test_clean_data_propagate(self, simple_matrix: csr_matrix) -> None:
        values = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
        result = _apply_scipy_batched(simple_matrix, values, "propagate")
        np.testing.assert_allclose(result, [[3.0, 6.0]])

    def test_nan_renormalize(self, simple_matrix: csr_matrix) -> None:
        values = np.array([[10.0, np.nan, 3.0]], dtype=np.float64)
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        np.testing.assert_allclose(result, [[10.0, 3.0]])

    def test_nan_propagate(self, simple_matrix: csr_matrix) -> None:
        values = np.array([[10.0, np.nan, 3.0]], dtype=np.float64)
        result = _apply_scipy_batched(simple_matrix, values, "propagate")
        assert np.isnan(result[0, 0])
        np.testing.assert_allclose(result[0, 1], 3.0)

    def test_all_nan_source_gives_nan(self, simple_matrix: csr_matrix) -> None:
        values = np.array([[np.nan, np.nan, np.nan]], dtype=np.float64)
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        assert np.all(np.isnan(result))

    def test_multi_batch(self, simple_matrix: csr_matrix) -> None:
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.5, 3.0])
        np.testing.assert_allclose(result[1], [4.5, 6.0])

    def test_static_mask_optimisation(self, simple_matrix: csr_matrix) -> None:
        """When first and last slices share the same NaN mask,
        only one support vector should be needed (internal optimisation)."""
        values = np.array(
            [
                [10.0, np.nan, 3.0],
                [20.0, np.nan, 6.0],
            ],
            dtype=np.float64,
        )
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        np.testing.assert_allclose(result[:, 0], [10.0, 20.0])
        np.testing.assert_allclose(result[:, 1], [3.0, 6.0])

    def test_varying_mask_across_batch(self, simple_matrix: csr_matrix) -> None:
        """Different NaN patterns per slice should still work."""
        values = np.array(
            [
                [10.0, np.nan, 3.0],
                [np.nan, 20.0, 6.0],
            ],
            dtype=np.float64,
        )
        result = _apply_scipy_batched(simple_matrix, values, "renormalize")
        np.testing.assert_allclose(result[0], [10.0, 3.0])
        np.testing.assert_allclose(result[1], [20.0, 6.0])


# ===================================================================
# apply_weights_nd
# ===================================================================


class TestApplyWeightsNd:
    def test_1d_single_slice(self, simple_matrix: csr_matrix) -> None:
        values = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [3.0, 6.0])

    def test_2d_batch(self, simple_matrix: csr_matrix) -> None:
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.5, 3.0])
        np.testing.assert_allclose(result[1], [4.5, 6.0])

    def test_3d_multi_batch(self, simple_matrix: csr_matrix) -> None:
        values = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        assert result.shape == (2, 3, 2)

    def test_multi_source_dims(self) -> None:
        """n_source_dims=2 should flatten (y, x) into a single source dim."""
        matrix = csr_matrix(np.eye(6, dtype=np.float64))
        values = np.arange(6, dtype=np.float64).reshape(2, 3)
        result = apply_weights_nd(
            values, matrix=matrix, n_source_dims=2, backend="scipy"
        )
        assert result.shape == (6,)
        np.testing.assert_allclose(result, np.arange(6, dtype=np.float64))

    def test_multi_source_dims_with_batch(self) -> None:
        matrix = csr_matrix(np.eye(6, dtype=np.float64))
        values = np.arange(24, dtype=np.float64).reshape(4, 2, 3)
        result = apply_weights_nd(
            values, matrix=matrix, n_source_dims=2, backend="scipy"
        )
        assert result.shape == (4, 6)

    def test_source_size_mismatch_raises(self, simple_matrix: csr_matrix) -> None:
        values = np.array([1.0, 2.0], dtype=np.float64)  # 2 != 3
        with pytest.raises(ValueError, match="does not match"):
            apply_weights_nd(values, matrix=simple_matrix, backend="scipy")

    def test_renormalize_with_nan(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="renormalize",
            backend="scipy",
        )
        np.testing.assert_allclose(result, [10.0, 3.0])

    def test_propagate_with_nan(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="propagate",
            backend="scipy",
        )
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 3.0)

    def test_identity_passthrough(self, identity_matrix: csr_matrix) -> None:
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values, matrix=identity_matrix, backend="scipy"
        )
        np.testing.assert_allclose(result, values)

    def test_scipy_backend_forced(self, simple_matrix: csr_matrix) -> None:
        values = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        np.testing.assert_allclose(result, [3.0, 6.0])


class TestApplyWeightsNdNumba:
    """Tests that run only when Numba is installed."""

    @pytest.fixture(autouse=True)
    def _require_numba(self) -> None:
        pytest.importorskip("numba")

    def test_single_slice_renormalize(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="renormalize",
            backend="numba",
        )
        np.testing.assert_allclose(result, [10.0, 3.0])

    def test_single_slice_propagate(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="propagate",
            backend="numba",
        )
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 3.0)

    def test_multi_batch_numba(self, simple_matrix: csr_matrix) -> None:
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="numba"
        )
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [1.5, 3.0])
        np.testing.assert_allclose(result[1], [4.5, 6.0])

    def test_matches_scipy(self, simple_matrix: csr_matrix) -> None:
        """Numba and SciPy backends should produce identical results."""
        values = np.array(
            [[10.0, np.nan, 3.0], [1.0, 2.0, np.nan]],
            dtype=np.float64,
        )
        scipy_result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        numba_result = apply_weights_nd(
            values, matrix=simple_matrix, backend="numba"
        )
        np.testing.assert_allclose(
            numba_result, scipy_result, equal_nan=True
        )


class TestApplyWeightsNdCupy:
    """Tests that run only when CuPy is installed."""

    @pytest.fixture(autouse=True)
    def _require_cupy(self) -> None:
        pytest.importorskip("cupy")

    def test_single_slice(self, simple_matrix: csr_matrix) -> None:
        values = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        result = apply_weights_nd(
            values, matrix=simple_matrix, backend="cupy"
        )
        np.testing.assert_allclose(result, [3.0, 6.0])

    def test_nan_renormalize(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="renormalize",
            backend="cupy",
        )
        np.testing.assert_allclose(result, [10.0, 3.0])

    def test_nan_propagate(self, simple_matrix: csr_matrix) -> None:
        values = np.array([10.0, np.nan, 3.0], dtype=np.float64)
        result = apply_weights_nd(
            values,
            matrix=simple_matrix,
            missing_policy="propagate",
            backend="cupy",
        )
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 3.0)

    def test_matches_scipy(self, simple_matrix: csr_matrix) -> None:
        values = np.array(
            [[10.0, np.nan, 3.0], [1.0, 2.0, np.nan]],
            dtype=np.float64,
        )
        scipy_result = apply_weights_nd(
            values, matrix=simple_matrix, backend="scipy"
        )
        cupy_result = apply_weights_nd(
            values, matrix=simple_matrix, backend="cupy"
        )
        np.testing.assert_allclose(
            cupy_result, scipy_result, equal_nan=True
        )
