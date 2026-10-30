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
        with pytest.raises(AttributeError, match="remp"), gd.config.set(remp={"method": "nearest"}):
            pass

    def test_wrong_namespace_type_is_rejected(self, toy: _ExampleConfig) -> None:
        with pytest.raises(TypeError), gd.config.set(toy=RemapConfig()):
            pass

    def test_invalid_value_restores_earlier_namespaces(self, toy: _ExampleConfig) -> None:
        previous = gd.config.remap.method
        other = "conservative" if previous == "nearest" else "nearest"
        with pytest.raises(ValueError), gd.config.set(remap={"method": other}, toy={"mode": "z"}):
            pass
        assert gd.config.remap.method == previous
        assert toy.mode == "b"


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
