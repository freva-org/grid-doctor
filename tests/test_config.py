from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pytest

import grid_doctor as gd
from grid_doctor.config import RemapConfig, _Namespace


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


class TestRemapConfig:
    def test_default_method_is_nearest(self) -> None:
        assert RemapConfig.from_env({}).method == "nearest"

    def test_env_var_name(self) -> None:
        env = {"GRID_DOCTOR_REMAP__METHOD": "conservative"}
        assert RemapConfig.from_env(env).method == "conservative"

    def test_env_var_is_read_at_import(self) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_REMAP__METHOD": "conservative", "PYTHONPATH": src}
        code = "import grid_doctor as gd; print(gd.config.remap.method)"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "conservative"
