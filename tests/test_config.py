from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import grid_doctor as gd
from grid_doctor import config, swath
from grid_doctor.metadata import HealpixGrid, HealpixMetadata
from grid_doctor.remap import _regrid as regrid_module
from grid_doctor.types import Ellipsoid


@pytest.fixture
def restore_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo any change a test makes to the process-wide settings."""
    for name in config.SETTINGS:
        monkeypatch.setattr(config, name, getattr(config, name))


@pytest.mark.usefixtures("restore_config")
class TestSettings:
    """Precedence, validation and ``set`` for the flat settings."""

    def test_defaults(self) -> None:
        assert config._read_env({}) == {}
        assert (config.ellipsoid, config.weights_store) == ("sphere", None)

    def test_env_var_names(self) -> None:
        assert [config.env_var(name) for name in config.SETTINGS] == [
            "GRID_DOCTOR_ELLIPSOID",
            "GRID_DOCTOR_WEIGHTS_STORE",
        ]

    def test_env_values_are_read_and_stripped(self) -> None:
        env = {"GRID_DOCTOR_ELLIPSOID": " WGS84 ", "GRID_DOCTOR_WEIGHTS_STORE": " /data/w "}
        assert config._read_env(env) == {"ellipsoid": "WGS84", "weights_store": "/data/w"}

    def test_empty_weights_store_counts_as_unset(self) -> None:
        assert config._read_env({"GRID_DOCTOR_WEIGHTS_STORE": " "}) == {"weights_store": None}

    def test_invalid_env_var_is_named_in_error(self) -> None:
        with pytest.raises(ValueError, match="GRID_DOCTOR_ELLIPSOID.*'wgs84'"):
            config._read_env({"GRID_DOCTOR_ELLIPSOID": "wgs84"})

    def test_assignment_overrides(self) -> None:
        gd.config.ellipsoid = "WGS84"
        assert config.ellipsoid == "WGS84"

    def test_invalid_assignment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ellipsoid"):
            gd.config.ellipsoid = "GRS80"  # type: ignore[assignment]
        assert config.ellipsoid == "sphere"

    def test_unknown_setting_is_rejected_by_set(self) -> None:
        with pytest.raises(AttributeError, match="elipsoid"), gd.config.set(elipsoid="WGS84"):
            pass

    def test_set_restores_previous_value(self) -> None:
        with gd.config.set(ellipsoid="WGS84", weights_store="/data/w"):
            inside = (config.ellipsoid, config.weights_store)
        assert inside == ("WGS84", "/data/w")
        assert (config.ellipsoid, config.weights_store) == ("sphere", None)

    def test_set_restores_on_error(self) -> None:
        with pytest.raises(RuntimeError), gd.config.set(ellipsoid="WGS84"):
            raise RuntimeError
        assert config.ellipsoid == "sphere"

    def test_set_invalid_value_changes_nothing(self) -> None:
        with pytest.raises(ValueError), gd.config.set(weights_store="/data/w", ellipsoid="GRS80"):
            pass
        assert (config.ellipsoid, config.weights_store) == ("sphere", None)

    @pytest.mark.parametrize(
        ("var", "code", "expected"),
        [
            ("GRID_DOCTOR_ELLIPSOID", "print(gd.config.ellipsoid)", "WGS84"),
            ("GRID_DOCTOR_WEIGHTS_STORE", "print(gd.config.weights_store)", "WGS84"),
        ],
    )
    def test_env_var_is_read_at_import(self, var: str, code: str, expected: str) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, var: "WGS84", "PYTHONPATH": src}
        script = f"import grid_doctor as gd; {code}"
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == expected

    def test_new_grids_use_configured_ellipsoid(self) -> None:
        sphere = gd.bin_to_healpix(_POINTS, 1, dense=False)["latitude"].values
        with gd.config.set(ellipsoid="WGS84"):
            wgs84 = gd.bin_to_healpix(_POINTS, 1, dense=False)["latitude"].values
        assert np.all(np.abs(wgs84) > np.abs(sphere))

    def test_existing_data_ignores_configured_ellipsoid(self) -> None:
        sphere = gd.bin_to_healpix(_POINTS, 1, dense=False)
        with gd.config.set(ellipsoid="WGS84"):
            again = gd.sparse_to_dense(sphere)
        assert HealpixMetadata.from_dataset(again).grid.ellipsoid == "sphere"


_POINTS = xr.Dataset(
    {"t": ("obs", np.arange(4.0))},
    coords={"lat": ("obs", [10.0, 40.0, -30.0, 60.0]), "lon": ("obs", [0.0, 90.0, 180.0, 270.0])},
)


class TestEllipsoidArgument:
    """Grid producers pass their ``ellipsoid=`` argument, else the config, down.

    The steps that use the ellipsoid are replaced by fakes recording what they
    receive, so no geometry is computed and ESMPy is not needed.
    """

    @pytest.fixture(params=[True, False], ids=["dense", "sparse"])
    def dense(self, request: pytest.FixtureRequest) -> bool:
        """Binning output layout; tests using it run for both."""
        return bool(request.param)

    @pytest.fixture
    def binning_passed(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Ellipsoid]:
        """Record the ``ellipsoid`` that binning's cell lookup and coordinate step receive."""
        passed: dict[str, Ellipsoid] = {}

        def fake_lonlat_to_healpix(lon: np.ndarray, lat: np.ndarray, **kwargs: object) -> np.ndarray:
            return np.zeros(lon.size, dtype=np.int64)  # every point in cell 0

        def fake_geo_module(grid: HealpixGrid) -> tuple[SimpleNamespace, dict[str, str]]:
            passed["cells"] = grid.ellipsoid
            return SimpleNamespace(lonlat_to_healpix=fake_lonlat_to_healpix), {}

        def fake_coords(ds: xr.Dataset, cell_ids: np.ndarray, grid: HealpixGrid, **kwargs: object) -> xr.Dataset:
            passed["coords"] = grid.ellipsoid
            return ds

        monkeypatch.setattr(swath, "_require_healpix_geo_module", fake_geo_module)
        monkeypatch.setattr(swath, "_scatter_to_dense", fake_coords)
        monkeypatch.setattr(swath, "_attach_sparse_coords", fake_coords)
        return passed

    @pytest.fixture
    def regrid_passed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Ellipsoid | None]:
        """Record the ``ellipsoid`` that regridding's weight generation and application receive.

        Application gets ``None`` unless an ellipsoid was given: it then
        takes the one the weight file was generated on.
        """
        passed: dict[str, Ellipsoid | None] = {}

        def fake_weights(*args: object, ellipsoid: Ellipsoid, **kwargs: object) -> Path:
            passed["weights"] = ellipsoid
            return tmp_path / "weights.nc"

        def fake_apply(ds: xr.Dataset, *args: object, ellipsoid: Ellipsoid | None, **kwargs: object) -> xr.Dataset:
            passed["apply"] = ellipsoid
            return ds

        monkeypatch.setattr(regrid_module, "compute_healpix_weights_backend", fake_weights)
        monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)
        return passed

    def test_binning_default_is_sphere(self, binning_passed: dict[str, Ellipsoid], dense: bool) -> None:
        # Pin the built-in default, whatever the environment says.
        with gd.config.set(ellipsoid="sphere"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_binning_none_uses_config(self, binning_passed: dict[str, Ellipsoid], dense: bool) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "WGS84", "coords": "WGS84"}

    def test_binning_argument_overrides_config(self, binning_passed: dict[str, Ellipsoid], dense: bool) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense, ellipsoid="sphere")
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_regrid_default_is_sphere(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        # Pin the built-in default, whatever the environment says.
        with gd.config.set(ellipsoid="sphere"):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "sphere", "apply": None}

    def test_regrid_none_uses_config(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "WGS84", "apply": None}

    def test_regrid_argument_overrides_config(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1, ellipsoid="sphere")
        assert regrid_passed == {"weights": "sphere", "apply": "sphere"}
