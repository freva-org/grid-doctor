"""Tests for Zarr publication behavior."""

import numpy as np
import pandas as pd
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


def test_merge_pressure_level_selection_retains_requested_levels():
    dataset = _pressure_dataset([1000, 850, 500], [10, 8, 5])

    selected = zarr_publisher._select_merge_pressure_levels(dataset, (850, 500))

    np.testing.assert_array_equal(selected["plev"].values, [850, 500])
    np.testing.assert_array_equal(selected["ta"].values, [8, 5])


def test_time_merge_action_distinguishes_append_overlap_and_rewrite():
    def dataset(times: list[str]) -> xr.Dataset:
        return xr.Dataset({"ta": ("time", np.arange(len(times)))}, coords={"time": pd.to_datetime(times)})

    existing = dataset(["2000-01-01", "2000-01-02"])

    assert zarr_publisher._time_merge_action(existing, dataset(["2000-01-03"])) == ("append", 0, 1)
    assert zarr_publisher._time_merge_action(existing, dataset(["2000-01-02"])) == ("rewrite-overlaps", 1, 0)
    assert zarr_publisher._time_merge_action(existing, dataset(["1999-12-31"])) == ("rewrite-store", 0, 1)


def test_missing_pressure_variable_writes_pressure_coordinate(monkeypatch):
    """A newly added pressure variable must write its physical ``plev`` values."""

    class FakeArray:
        def __init__(self):
            self.attrs: dict[str, object] = {}
            self.values: np.ndarray | None = None

        def __setitem__(self, key, values):
            self.values = np.asarray(values)

    class FakeRoot(dict):
        def create_dataset(self, name, **kwargs):
            array = FakeArray()
            self[name] = array
            return array

    root = FakeRoot()
    monkeypatch.setattr(zarr_publisher.zarr, "open_group", lambda *args, **kwargs: root)
    monkeypatch.setattr(zarr_publisher.zarr, "consolidate_metadata", lambda *args, **kwargs: None)

    existing = xr.Dataset({"surface": ("plev", [1, 2])})
    candidate = _pressure_dataset([1000, 850], [10, 8]).rename({"ta": "zg"})
    zarr_publisher._write_missing_variables(existing, candidate, "/tmp/test.zarr", zarr_format=2)

    np.testing.assert_array_equal(root["plev"].values, [1000, 850])
    assert root["plev"].attrs["_ARRAY_DIMENSIONS"] == ["plev"]


def test_refreshing_public_attrs_preserves_existing_variable_metadata():
    class FakeArray:
        def __init__(self):
            self.attrs = {
                "_ARRAY_DIMENSIONS": ["time", "cell"],
                "grid_mapping": "crs",
                "original_attribute": "keep me",
                "long_name": "old name",
                "last_real_data": "2026-09-01",
            }

    array = FakeArray()

    changed = zarr_publisher._replace_public_attrs(
        array,
        {"long_name": "refreshed name", "units": "K"},
    )

    assert changed is True
    assert array.attrs == {
        "_ARRAY_DIMENSIONS": ["time", "cell"],
        "grid_mapping": "crs",
        "original_attribute": "keep me",
        "long_name": "refreshed name",
        "units": "K",
        "last_real_data": "2026-09-01",
    }


def test_rebuilding_store_preserves_existing_variable_attrs():
    existing = xr.Dataset({"tas": ("time", [1])})
    existing["tas"].attrs = {"original_attribute": "keep me", "long_name": "old name"}
    candidate = xr.Dataset({"tas": ("time", [2])})
    candidate["tas"].attrs = {"long_name": "refreshed name", "units": "K"}
    merged = candidate.combine_first(existing)

    result = zarr_publisher._preserve_update_attrs(existing, candidate, merged)

    assert result["tas"].attrs == {
        "original_attribute": "keep me",
        "long_name": "refreshed name",
        "units": "K",
    }
