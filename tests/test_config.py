from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import grid_doctor as gd
from grid_doctor.config import RemapConfig


class TestRemapConfig:
    def test_default_method_is_nearest(self) -> None:
        assert RemapConfig.from_env({}).method == "nearest"

    def test_env_var_overrides_default(self) -> None:
        assert RemapConfig.from_env({"GRID_DOCTOR_REMAP__METHOD": " conservative "}).method == "conservative"

    def test_invalid_env_var_is_named_in_error(self) -> None:
        with pytest.raises(ValueError, match="GRID_DOCTOR_REMAP__METHOD.*bilinear"):
            RemapConfig.from_env({"GRID_DOCTOR_REMAP__METHOD": "bilinear"})

    def test_invalid_assignment_is_rejected(self) -> None:
        cfg = RemapConfig()
        with pytest.raises(ValueError, match="bilinear"):
            cfg.method = "bilinear"  # type: ignore[assignment]
        assert cfg.method == "nearest"

    def test_unknown_setting_is_rejected(self) -> None:
        with pytest.raises(AttributeError):
            RemapConfig().metod = "nearest"  # type: ignore[attr-defined]

    def test_env_var_is_read_at_import(self) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_REMAP__METHOD": "conservative", "PYTHONPATH": src}
        code = "import grid_doctor as gd; print(gd.config.remap.method)"
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "conservative"
