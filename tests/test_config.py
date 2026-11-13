from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest
import xarray as xr

import grid_doctor as gd
from grid_doctor.config import HealpixConfig, _Namespace


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

    def test_env_var_is_read_at_import(self) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_HEALPIX__ELLIPSOID": "WGS84", "PYTHONPATH": src}
        code = "import grid_doctor as gd; print(gd.config.healpix.ellipsoid)"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "WGS84"

    def test_geometry_uses_configured_ellipsoid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cells = np.array([0, 1, 2], dtype=np.int64)
        ds = xr.Dataset({"t": ("cell", np.zeros(3))})
        sphere = gd.attach_cell_coords(ds, cells, level=1)["latitude"].values
        monkeypatch.setattr(gd.config.healpix, "ellipsoid", "WGS84")
        wgs84 = gd.attach_cell_coords(ds, cells, level=1)["latitude"].values
        assert np.all(np.abs(wgs84) > np.abs(sphere))


_POINTS = xr.Dataset(
    {"t": ("obs", np.arange(4.0))},
    coords={"lat": ("obs", [10.0, 40.0, -30.0, 60.0]), "lon": ("obs", [0.0, 90.0, 180.0, 270.0])},
)


def _bin(ds: xr.Dataset, ellipsoid: str | None = None) -> xr.Dataset:
    return gd.bin_to_healpix(_POINTS, 1, ellipsoid=ellipsoid)


def _bin_sparse(ds: xr.Dataset, ellipsoid: str | None = None) -> xr.Dataset:
    return gd.bin_to_healpix(_POINTS, 1, dense=False, ellipsoid=ellipsoid)


def _regrid(ds: xr.Dataset, ellipsoid: str | None = None) -> xr.Dataset:
    return gd.regrid_to_healpix(ds, 1, method="nearest", ellipsoid=ellipsoid)


class TestExplicitEllipsoid:
    """An ``ellipsoid=`` argument on grid producers wins over the config."""

    @pytest.mark.parametrize("produce", [_bin, _bin_sparse, _regrid])
    def test_argument_overrides_config(
        self, produce: Callable[..., xr.Dataset], regular_ds: xr.Dataset
    ) -> None:
        with gd.config.healpix.set(ellipsoid="WGS84"):
            wgs84 = produce(regular_ds)["latitude"].values
        sphere = produce(regular_ds)["latitude"].values
        assert not np.allclose(wgs84, sphere)

        np.testing.assert_array_equal(produce(regular_ds, ellipsoid="WGS84")["latitude"].values, wgs84)
        with gd.config.healpix.set(ellipsoid="WGS84"):
            np.testing.assert_array_equal(produce(regular_ds, ellipsoid="sphere")["latitude"].values, sphere)
