"""Tests for cleanup of existing HEALPix Zarr stores."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from heal_era5.helpers import cleanup


def _pressure_dataset() -> xr.Dataset:
    """Build a small store containing pressure-level and surface variables."""

    return xr.Dataset(
        data_vars={
            "ta": (("time", "plev", "cell"), np.arange(12).reshape(2, 3, 2)),
            "hus": (("time", "plev", "cell"), np.arange(12, 24).reshape(2, 3, 2)),
            "tas": (("time", "cell"), np.arange(4).reshape(2, 2)),
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "plev": [1000, 850, 500],
            "cell": [0, 1],
        },
    )


def _write_store(destination: Path) -> None:
    _pressure_dataset().to_zarr(destination, mode="w", zarr_format=2, consolidated=True)


def test_drop_pressure_levels_rewrites_all_pressure_variables(tmp_path: Path):
    destination = tmp_path / "level_8.zarr"
    _write_store(destination)

    changed = cleanup._drop_pressure_levels_from_zarr_store(
        destination=str(destination),
        variable_names=None,
        pressure_levels=(850,),
    )

    assert changed is True
    actual = xr.open_zarr(destination, consolidated=True)
    try:
        assert actual["ta"].plev.values.tolist() == [1000, 500]
        assert actual["hus"].plev.values.tolist() == [1000, 500]
        assert actual["tas"].values.tolist() == [[0, 1], [2, 3]]
    finally:
        actual.close()


def test_drop_pressure_levels_removes_coordinate_after_deleting_all_pressure_variables(
    tmp_path: Path,
):
    destination = tmp_path / "level_8.zarr"
    _write_store(destination)

    changed = cleanup._drop_pressure_levels_from_zarr_store(
        destination=str(destination),
        variable_names=None,
        pressure_levels=None,
    )

    assert changed is True
    actual = xr.open_zarr(destination, consolidated=True)
    try:
        assert set(actual.data_vars) == {"tas"}
        assert "plev" not in actual.coords
    finally:
        actual.close()


def test_drop_pressure_levels_skips_unknown_levels(tmp_path: Path):
    destination = tmp_path / "level_8.zarr"
    _write_store(destination)

    changed = cleanup._drop_pressure_levels_from_zarr_store(
        destination=str(destination),
        variable_names=None,
        pressure_levels=(925,),
    )

    assert changed is False
    actual = xr.open_zarr(destination, consolidated=True)
    try:
        assert actual["ta"].plev.values.tolist() == [1000, 850, 500]
    finally:
        actual.close()


def test_clean_frequency_stores_targets_requested_levels_and_reports_changes(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[dict[str, object]] = []

    def selected_destinations(*_, levels, **__):
        assert levels == (7,)
        return [(7, tmp_path / "level_7.zarr")]

    monkeypatch.setattr(cleanup, "selected_level_destinations", selected_destinations)
    monkeypatch.setattr(
        cleanup,
        "_drop_pressure_levels_from_zarr_store",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    actions = cleanup.clean_frequency_stores(
        dataset="era5land",
        frequency="1hr",
        variable_names=None,
        pressure_levels=(1000, 850),
        levels=(7,),
    )

    assert calls == [
        {
            "destination": str(tmp_path / "level_7.zarr"),
            "variable_names": None,
            "pressure_levels": (1000, 850),
            "zarr_format": 2,
        }
    ]
    assert actions == [f"❌ removed pressure levels 1000,850 from {tmp_path / 'level_7.zarr'} (level 7)"]


def test_clean_frequency_stores_dry_run_does_not_change_stores(monkeypatch, tmp_path: Path):
    destination = tmp_path / "level_8.zarr"
    monkeypatch.setattr(cleanup, "selected_level_destinations", lambda *_, **__: [(8, destination)])
    monkeypatch.setattr(
        cleanup,
        "_drop_pressure_levels_from_zarr_store",
        lambda **_: pytest.fail("dry-run must not rewrite a store"),
    )

    actions = cleanup.clean_frequency_stores(
        dataset="era5land",
        frequency="1hr",
        variable_names=None,
        pressure_levels=(850,),
        levels=None,
        dry_run=True,
    )

    assert actions == [f"would remove pressure levels 850 from all pressure-level variables in {destination} (level 8)"]


def test_clean_frequency_stores_requires_a_selection():
    with pytest.raises(ValueError, match="selection is required"):
        cleanup.clean_frequency_stores(
            dataset="era5land",
            frequency="1hr",
            variable_names=None,
            levels=None,
        )
