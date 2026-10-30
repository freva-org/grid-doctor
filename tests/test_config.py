from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import ClassVar, Literal

import pytest

from grid_doctor.config import HealpixConfig, _Namespace


@dataclass(slots=True)
class _ExampleConfig(_Namespace):
    namespace: ClassVar[str] = "toy"
    mode: Literal["a", "b", "c"] = "a"



class TestNamespace:
    """Behaviour shared by every settings namespace."""

    def test_invalid_env_var_is_named_in_error(self) -> None:
        with pytest.raises(ValueError, match="GRID_DOCTOR_TOY__MODE.*'z'"):
            _ExampleConfig.from_env({"GRID_DOCTOR_TOY__MODE": "z"})

    def test_invalid_assignment_is_rejected(self) -> None:
        cfg = _ExampleConfig()
        with pytest.raises(ValueError, match="toy.mode"):
            cfg.mode = "z"  # type: ignore[assignment]
        assert cfg.mode == "a"


class TestHealpixConfig:
    def test_default_ellipsoid_is_sphere(self) -> None:
        assert HealpixConfig.from_env({}).ellipsoid == "sphere"

    def test_env_var_overrides_default(self) -> None:
        env = {"GRID_DOCTOR_HEALPIX__ELLIPSOID": " WGS84 "}
        assert HealpixConfig.from_env(env).ellipsoid == "WGS84"

    def test_unknown_setting_is_rejected(self) -> None:
        with pytest.raises(AttributeError):
            HealpixConfig().elipsoid = "WGS84"  # type: ignore[attr-defined]

    def test_env_var_is_read_at_import(self) -> None:
        env = {**os.environ, "GRID_DOCTOR_HEALPIX__ELLIPSOID": "WGS84"}
        code = "import grid_doctor as gd; print(gd.config.healpix.ellipsoid)"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "WGS84"
