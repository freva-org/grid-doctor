from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
import pytest
import xarray as xr

import grid_doctor as gd
from grid_doctor import remap, swath
from grid_doctor.config import HealpixConfig, _Namespace
from grid_doctor.metadata import HealpixGrid, HealpixMetadata
from grid_doctor.remap import _regrid as regrid_module
from grid_doctor.types import Ellipsoid


@dataclass(slots=True)
class _ExampleConfig(_Namespace):
    namespace: ClassVar[str] = "toy"
    mode: Literal["a", "b", "c"] = "a"
    label: str = "x"



class TestNamespace:
    """Behaviour shared by every settings namespace."""

    def test_precedence_default_env_assignment(self) -> None:
        assert _ExampleConfig.from_env({}).mode == "a"
        cfg = _ExampleConfig.from_env({"GRID_DOCTOR_TOY__MODE": "b"})
        assert cfg.mode == "b"
        cfg.mode = "c"
        assert cfg.mode == "c"

    def test_env_values_are_stripped(self) -> None:
        cfg = _ExampleConfig.from_env({"GRID_DOCTOR_TOY__MODE": " b ", "GRID_DOCTOR_TOY__LABEL": " y "})
        assert (cfg.mode, cfg.label) == ("b", "y")

    def test_invalid_env_var_is_named_in_error(self) -> None:
        with pytest.raises(ValueError, match="GRID_DOCTOR_TOY__MODE.*'z'"):
            _ExampleConfig.from_env({"GRID_DOCTOR_TOY__MODE": "z"})

    def test_invalid_assignment_is_rejected(self) -> None:
        cfg = _ExampleConfig()
        with pytest.raises(ValueError, match="toy.mode"):
            cfg.mode = "z"  # type: ignore[assignment]
        assert cfg.mode == "a"

    def test_unknown_setting_is_rejected(self) -> None:
        with pytest.raises(AttributeError):
            _ExampleConfig().mod = "a"  # type: ignore[attr-defined]

    def test_set_restores_previous_value(self) -> None:
        cfg = _ExampleConfig(mode="b")
        with cfg.set(mode="c"):
            inside = cfg.mode
        assert (inside, cfg.mode) == ("c", "b")

    def test_set_restores_on_error(self) -> None:
        cfg = _ExampleConfig(mode="b")
        with pytest.raises(RuntimeError), cfg.set(mode="c"):
            raise RuntimeError
        assert cfg.mode == "b"

    def test_set_invalid_value_changes_nothing(self) -> None:
        cfg = _ExampleConfig(mode="b")
        with pytest.raises(ValueError), cfg.set(label="y", mode="z"):
            pass
        assert (cfg.mode, cfg.label) == ("b", "x")


class TestConfigSet:
    """``grid_doctor.config.set`` across namespaces."""

    @pytest.fixture
    def toy(self, monkeypatch: pytest.MonkeyPatch) -> _ExampleConfig:
        cfg = _ExampleConfig(mode="b", label="y")
        monkeypatch.setattr(gd.config, "toy", cfg, raising=False)
        return cfg

    def test_keyword_mapping_changes_only_named_settings(self, toy: _ExampleConfig) -> None:
        with gd.config.set(toy={"mode": "c"}):
            inside = (toy.mode, toy.label)
        assert inside == ("c", "y")
        assert (toy.mode, toy.label) == ("b", "y")

    def test_keyword_namespace_replaces_all_settings(self, toy: _ExampleConfig) -> None:
        with gd.config.set(toy=_ExampleConfig(mode="c")):
            inside = (toy.mode, toy.label)
        assert inside == ("c", "x")
        assert (toy.mode, toy.label) == ("b", "y")

    def test_positional_mapping(self, toy: _ExampleConfig) -> None:
        with gd.config.set({"toy": _ExampleConfig(mode="c")}):
            inside = toy.mode
        assert (inside, toy.mode) == ("c", "b")

    def test_keeps_namespace_identity(self, toy: _ExampleConfig) -> None:
        with gd.config.set(toy=_ExampleConfig(mode="c")):
            assert gd.config.toy is toy  # type: ignore[attr-defined]

    def test_unknown_namespace_is_rejected(self) -> None:
        with pytest.raises(AttributeError, match="healpx"), gd.config.set(healpx={"ellipsoid": "WGS84"}):
            pass

    def test_wrong_namespace_type_is_rejected(self, toy: _ExampleConfig) -> None:
        with pytest.raises(TypeError), gd.config.set(toy=HealpixConfig()):
            pass

    def test_invalid_value_restores_earlier_namespaces(self, toy: _ExampleConfig) -> None:
        previous = gd.config.healpix.ellipsoid
        other = "WGS84" if previous != "WGS84" else "sphere"
        with pytest.raises(ValueError), gd.config.set(healpix={"ellipsoid": other}, toy={"mode": "z"}):
            pass
        assert gd.config.healpix.ellipsoid == previous
        assert toy.mode == "b"


class TestHealpixConfig:
    def test_default_ellipsoid_is_sphere(self) -> None:
        assert HealpixConfig.from_env({}).ellipsoid == "sphere"

    def test_env_var_name(self) -> None:
        env = {"GRID_DOCTOR_HEALPIX__ELLIPSOID": "WGS84"}
        assert HealpixConfig.from_env(env).ellipsoid == "WGS84"

    def test_rejects_unknown_ellipsoid(self) -> None:
        with pytest.raises(ValueError, match="GRID_DOCTOR_HEALPIX__ELLIPSOID.*'wgs84'"):
            HealpixConfig.from_env({"GRID_DOCTOR_HEALPIX__ELLIPSOID": "wgs84"})
        with pytest.raises(ValueError, match="healpix.ellipsoid"):
            HealpixConfig().ellipsoid = "GRS80"  # type: ignore[assignment]

    def test_env_var_is_read_at_import(self) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_HEALPIX__ELLIPSOID": "WGS84", "PYTHONPATH": src}
        code = "import grid_doctor as gd; print(gd.config.healpix.ellipsoid)"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "WGS84"

    def test_new_grids_use_configured_ellipsoid(self) -> None:
        sphere = gd.bin_to_healpix(_POINTS, 1, dense=False)["latitude"].values
        with gd.config.healpix.set(ellipsoid="WGS84"):
            wgs84 = gd.bin_to_healpix(_POINTS, 1, dense=False)["latitude"].values
        assert np.all(np.abs(wgs84) > np.abs(sphere))

    def test_existing_data_ignores_configured_ellipsoid(self) -> None:
        sphere = gd.bin_to_healpix(_POINTS, 1, dense=False)
        with gd.config.healpix.set(ellipsoid="WGS84"):
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
        # HealpixConfig() holds the built-in defaults, whatever the environment says.
        with gd.config.set(healpix=HealpixConfig()):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_binning_none_uses_config(self, binning_passed: dict[str, Ellipsoid], dense: bool) -> None:
        with gd.config.healpix.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "WGS84", "coords": "WGS84"}

    def test_binning_argument_overrides_config(self, binning_passed: dict[str, Ellipsoid], dense: bool) -> None:
        with gd.config.healpix.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense, ellipsoid="sphere")
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_regrid_default_is_sphere(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        # HealpixConfig() holds the built-in defaults, whatever the environment says.
        with gd.config.set(healpix=HealpixConfig()):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "sphere", "apply": None}

    def test_regrid_none_uses_config(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        with gd.config.healpix.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "WGS84", "apply": None}

    def test_regrid_argument_overrides_config(self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset) -> None:
        with gd.config.healpix.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1, ellipsoid="sphere")
        assert regrid_passed == {"weights": "sphere", "apply": "sphere"}
