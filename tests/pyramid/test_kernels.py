"""Coarsening kernels and partial means (``grid_doctor.pyramid._kernels``, ``_blocks``)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from grid_doctor.pyramid._blocks import (
    MeanPartials,
    coarsen_mean_steps,
)
from grid_doctor.pyramid._build import (
    coarsen_healpix,
    create_healpix_pyramid,
)
from grid_doctor.pyramid._kernels import (
    _coarsen_array,
    _coarsen_array_mode,
    coarsen_mean,
    min_valid_count,
)
from grid_doctor.xarray_accessor import DatasetAccessor

from ..helpers import with_healpix_metadata


class TestCoarsenMean:
    def test_issue_54_example(self) -> None:
        values = np.array([2.0] * 12 + [8.0, 8.0, np.nan, np.nan])
        result = coarsen_mean(values, factor=16, min_valid_fraction=0.5)
        np.testing.assert_allclose(result, [40 / 14])

    def test_rejects_non_divisible_factor(self) -> None:
        with pytest.raises(ValueError, match="factor"):
            coarsen_mean(np.zeros(10), factor=4)

    def test_threshold_round_off(self) -> None:
        # 0.3 * 10 is 3.0000000000000004 in float64; must still be 3.
        assert min_valid_count(0.3, 10) == 3
        assert min_valid_count(0.5, 4) == 2
        assert min_valid_count(0.0, 4) == 1

    def test_threshold_is_cumulative_from_finest(self) -> None:
        # Each of four children is exactly half valid: a chained 50%
        # threshold would keep the parent, but only 8 of 16 cells are
        # valid, so with 0.6 it must be masked.
        values = np.tile([1.0, 1.0, np.nan, np.nan], 4)
        assert np.isnan(coarsen_mean(values, factor=16, min_valid_fraction=0.6))[0]
        assert coarsen_mean(values, factor=16, min_valid_fraction=0.5)[0] == 1.0


class TestPyramidConsistency:
    @staticmethod
    def _dataset_with_nans(level: int = 4) -> xr.Dataset:
        npix = 12 * 4**level
        rng = np.random.default_rng(54)
        data = rng.normal(size=(2, npix))
        # clustered NaNs, like a coastline: mask whole and partial blocks
        data[:, : npix // 3] = np.nan
        data[:, rng.random(npix) < 0.3] = np.nan
        ds = xr.Dataset(
            {"tos": (("time", "cell"), data)},
            coords={"time": np.arange(2), "cell": np.arange(npix)},
        )
        return with_healpix_metadata(ds, level, method="conservative")

    def test_pyramid_levels_match_direct_coarsening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        finest = self._dataset_with_nans()
        monkeypatch.setattr(
            "grid_doctor.xarray_accessor.DatasetAccessor.regrid_to_healpix", lambda self, level, **kw: finest
        )
        pyramid = create_healpix_pyramid(
            finest, max_level=4, min_level=0, min_valid_fraction=0.0
        )
        values = finest["tos"].values
        for level in range(4):
            factor = 4 ** (4 - level)
            expected = coarsen_mean(values, factor=factor, min_valid_fraction=0.0)
            np.testing.assert_allclose(
                pyramid[level]["tos"].values, expected, equal_nan=True
            )

    def test_chaining_is_path_dependent(self) -> None:
        """Documents why the pyramid must not chain means."""
        finest = self._dataset_with_nans()
        chained = coarsen_healpix(
            coarsen_healpix(finest, 3, min_valid_fraction=0.0),
            2,
            min_valid_fraction=0.0,
        )
        direct = coarsen_healpix(finest, 2, min_valid_fraction=0.0)
        assert not np.allclose(
            chained["tos"].values, direct["tos"].values, equal_nan=True
        )


class TestCoarsenArray:
    """Tests for the low-level ``_coarsen_array`` mean kernel."""

    def test_all_valid(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        np.testing.assert_allclose(result, [[2.5, 6.5]])

    def test_below_threshold_becomes_nan(self) -> None:
        """1 of 4 valid children (25%) is below the 50% threshold."""
        data = np.array([[1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0]), "1/4 valid should be NaN"
        np.testing.assert_allclose(result[0, 1], 6.5)

    def test_at_threshold_is_valid(self) -> None:
        """2 of 4 valid children (50%) meets the 50% threshold."""
        data = np.array([[1.0, 2.0, np.nan, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        np.testing.assert_allclose(result[0, 0], 1.5)

    def test_all_nan_becomes_nan(self) -> None:
        data = np.full((1, 4), np.nan)
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_siberian_lake_killed(self) -> None:
        """A single valid pixel among 3 NaN siblings must not survive."""
        data = np.full((1, 4), np.nan)
        data[0, 2] = 15.0
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_batch_dims_preserved(self) -> None:
        data = np.ones((3, 5, 16))
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert result.shape == (3, 5, 4)

    def test_custom_threshold(self) -> None:
        """With threshold 0.25, 1 of 4 valid should survive."""
        data = np.array([[7.0, np.nan, np.nan, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.25)
        np.testing.assert_allclose(result[0, 0], 7.0)

    def test_strict_threshold(self) -> None:
        """With threshold 1.0, any NaN child kills the parent."""
        data = np.array([[1.0, 2.0, 3.0, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=1.0)
        assert np.isnan(result[0, 0])


class TestCoarsenArrayMode:
    """Tests for the low-level ``_coarsen_array_mode`` kernel."""

    def test_clear_majority(self) -> None:
        data = np.array([[5.0, 5.0, 5.0, 3.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 5.0

    def test_tie_returns_valid_class(self) -> None:
        """All unique values — any of them is acceptable."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] in {1.0, 2.0, 3.0, 4.0}

    def test_nan_excluded_from_vote(self) -> None:
        data = np.array([[3.0, 3.0, 7.0, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 3.0

    def test_below_threshold_becomes_nan(self) -> None:
        data = np.array([[5.0, np.nan, np.nan, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_at_threshold_is_valid(self) -> None:
        data = np.array([[5.0, 5.0, np.nan, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 5.0

    def test_batch_dims_preserved(self) -> None:
        data = np.ones((2, 3, 8))
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result.shape == (2, 3, 2)

    def test_does_not_average_classes(self) -> None:
        """Mode must return an actual class label, never an average."""
        data = np.array([[2.0, 2.0, 8.0, 8.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] in {2.0, 8.0}, "Result must be an existing class"
        assert result[0, 0] != 5.0, "Must not be the average"


LEVEL = 3
NPIX = 12 * 4**LEVEL


def _values_with_nans(shape: tuple[int, ...], seed: int = 54) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)
    values[..., : shape[-1] // 3] = np.nan  # a clustered "coastline"
    values[rng.random(shape) < 0.3] = np.nan
    return values


class TestMeanPartials:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_chained_partials_match_direct_mean(self, chunked: bool) -> None:
        values = _values_with_nans((2, NPIX))
        arr = da.from_array(values, chunks=(1, 64)) if chunked else values
        partials = MeanPartials.from_values(arr).rechunk(64)
        for steps in range(1, LEVEL + 1):
            expected = coarsen_mean(values, factor=4**steps, min_valid_fraction=0.4)
            np.testing.assert_allclose(
                np.asarray(partials.mean(0.4)), expected, equal_nan=True
            )
            partials = partials.coarsen().rechunk(64)

    def test_dask_stays_chunked_along_cells(self) -> None:
        arr = da.from_array(_values_with_nans((2, NPIX)), chunks=(1, 64))
        out = coarsen_mean_steps(arr, 1, min_valid_fraction=0.5, cell_chunk=64)
        assert isinstance(out, da.Array)
        assert out.chunks[-1] == (64, 64, 64)
