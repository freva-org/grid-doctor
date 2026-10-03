"""Tests for NaN-aware mean coarsening (grid-doctor issue #54)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor.helpers import coarsen_healpix, create_healpix_pyramid
from grid_doctor.pyramid import coarsen_mean, min_valid_count


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
        return xr.Dataset(
            {"tos": (("time", "cell"), data)},
            coords={"time": np.arange(2), "cell": np.arange(npix)},
            attrs={
                "healpix_nside": 2**level,
                "healpix_level": level,
                "healpix_order": "nested",
                "grid_doctor_method": "conservative",
            },
        )

    def test_pyramid_levels_match_direct_coarsening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        finest = self._dataset_with_nans()
        monkeypatch.setattr(
            "grid_doctor.helpers.regrid_to_healpix", lambda ds, level, **kw: finest
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
