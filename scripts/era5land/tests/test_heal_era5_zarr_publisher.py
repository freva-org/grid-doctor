"""Tests for Zarr publication behavior."""

import numpy as np
import xarray as xr

from heal_era5.helpers import zarr_publisher


def _pressure_dataset(levels: list[int], values: list[int]) -> xr.Dataset:
    return xr.Dataset(
        {"ta": (("plev",), values)},
        coords={"plev": levels},
    )


def test_matching_pressure_levels_in_a_different_order_do_not_require_a_rewrite():
    existing = _pressure_dataset([1000, 850, 500], [10, 8, 5])
    candidate = _pressure_dataset([500, 1000, 850], [5, 10, 8])

    assert zarr_publisher._requires_vertical_rewrite(existing, candidate) is False

    aligned = zarr_publisher._align_vertical_levels_to_existing(existing, candidate)

    np.testing.assert_array_equal(aligned["plev"].values, [1000, 850, 500])
    np.testing.assert_array_equal(aligned["ta"].values, [10, 8, 5])


def test_added_or_removed_pressure_levels_require_a_rewrite():
    existing = _pressure_dataset([1000, 850, 500], [10, 8, 5])
    candidate = _pressure_dataset([1000, 850, 250], [10, 8, 2])

    assert zarr_publisher._requires_vertical_rewrite(existing, candidate) is True
