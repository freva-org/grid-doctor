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
from grid_doctor.config import template
from grid_doctor.metadata import HealpixGrid, HealpixMetadata
from grid_doctor.remap import _esmf as esmf_module
from grid_doctor.remap import _regrid as regrid_module
from grid_doctor.types import Ellipsoid
from grid_doctor.weights import describe_source


@pytest.fixture
def restore_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo any change a test makes to the process-wide settings."""
    for name in (*config.SETTINGS, "loaded_files"):
        monkeypatch.setattr(config, name, getattr(config, name))


@pytest.mark.usefixtures("restore_config")
class TestSettings:
    """Precedence, validation and ``set`` for the flat settings."""

    def test_defaults(self) -> None:
        assert config._read_env({}) == {}
        assert [getattr(config, name) for name in config.SETTINGS] == [
            "sphere",
            None,
            None,
            None,
        ]

    def test_env_var_names(self) -> None:
        assert [config.env_var(name) for name in config.SETTINGS] == [
            "GRID_DOCTOR_ELLIPSOID",
            "GRID_DOCTOR_WEIGHTS_STORE",
            "GRID_DOCTOR_ESMF_REGRID_WEIGHTGEN",
            "GRID_DOCTOR_SCRATCH_DIR",
        ]

    def test_env_values_are_read_and_stripped(self) -> None:
        env = {
            "GRID_DOCTOR_ELLIPSOID": " WGS84 ",
            "GRID_DOCTOR_WEIGHTS_STORE": " /data/w ",
        }
        assert config._read_env(env) == {
            "ellipsoid": "WGS84",
            "weights_store": "/data/w",
        }

    def test_empty_weights_store_counts_as_unset(self) -> None:
        assert config._read_env({"GRID_DOCTOR_WEIGHTS_STORE": " "}) == {
            "weights_store": None
        }

    def test_assignment_overrides(self) -> None:
        gd.config.ellipsoid = "WGS84"
        assert config.ellipsoid == "WGS84"

    def test_invalid_assignment_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ellipsoid"):
            gd.config.ellipsoid = "GRS80"  # type: ignore[assignment]
        assert config.ellipsoid == "sphere"

    def test_unknown_setting_is_rejected_by_set(self) -> None:
        with (
            pytest.raises(AttributeError, match="elipsoid"),
            gd.config.set(elipsoid="WGS84"),
        ):
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
        with (
            pytest.raises(ValueError),
            gd.config.set(weights_store="/data/w", ellipsoid="GRS80"),
        ):
            pass
        assert (config.ellipsoid, config.weights_store) == ("sphere", None)

    @pytest.mark.parametrize(
        ("var", "code", "expected"),
        [
            ("GRID_DOCTOR_ELLIPSOID", "print(gd.config.ellipsoid)", "WGS84"),
            ("GRID_DOCTOR_WEIGHTS_STORE", "print(gd.config.weights_store)", "WGS84"),
        ],
    )
    def test_env_var_is_read_at_import(
        self, var: str, code: str, expected: str
    ) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, var: "WGS84", "PYTHONPATH": src}
        script = f"import grid_doctor as gd; {code}"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
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


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.mark.usefixtures("restore_config")
class TestSetGetRefresh:
    """``set`` without and with ``with``, from a file and one by one; ``get``; ``refresh``."""

    def test_set_without_with_stays(self) -> None:
        gd.config.set(ellipsoid="WGS84")
        assert gd.config.ellipsoid == "WGS84"

    def test_set_as_context_manager_restores(self) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            assert gd.config.ellipsoid == "WGS84"
        assert gd.config.ellipsoid == "sphere"

    def test_nested_blocks_restore_in_order(self) -> None:
        with gd.config.set(scratch_dir="/a"):
            with gd.config.set(scratch_dir="/b"):
                assert gd.config.scratch_dir == "/b"
            assert gd.config.scratch_dir == "/a"
        assert gd.config.scratch_dir is None

    def test_set_from_file_then_items(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "p.toml", 'ellipsoid = "WGS84"\nweights_store = "/w"\n'
        )
        with gd.config.set(path, weights_store="/mine"):
            assert gd.config.get() == {
                "ellipsoid": "WGS84",
                "weights_store": "/mine",  # items after the file
                "esmf_regrid_weightgen": None,
                "scratch_dir": None,
            }
            assert gd.config.loaded_files[-1] == path
        assert gd.config.get("weights_store") is None
        assert path not in gd.config.loaded_files

    def test_file_overrides_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GRID_DOCTOR_ELLIPSOID", "sphere")
        gd.config.refresh()
        gd.config.set(_write(tmp_path / "p.toml", 'ellipsoid = "WGS84"\n'))
        assert gd.config.ellipsoid == "WGS84"

    def test_set_from_file_only_changes_named_settings(self, tmp_path: Path) -> None:
        gd.config.set(scratch_dir="/keep")
        gd.config.set(_write(tmp_path / "p.toml", 'ellipsoid = "WGS84"\n'))
        assert (gd.config.ellipsoid, gd.config.scratch_dir) == ("WGS84", "/keep")

    @pytest.mark.parametrize(
        ("text", "settings", "error"),
        [
            ('ellipsoid = "WGS84"\nweight_store = "/w"\n', {}, ValueError),
            ('ellipsoid = "WGS84"\n', {"scratch_dir": 1}, ValueError),
            ('ellipsoid = "WGS84"\n', {"elipsoid": "sphere"}, AttributeError),
        ],
        ids=["bad-file", "bad-value", "unknown-setting"],
    )
    def test_invalid_set_changes_nothing(
        self,
        tmp_path: Path,
        text: str,
        settings: dict[str, object],
        error: type[Exception],
    ) -> None:
        before = gd.config.get(), gd.config.loaded_files
        with pytest.raises(error):
            gd.config.set(_write(tmp_path / "p.toml", text), **settings)
        assert (gd.config.get(), gd.config.loaded_files) == before

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="missing.toml"):
            gd.config.set(tmp_path / "missing.toml")

    def test_file_argument_is_positional_only(self) -> None:
        with pytest.raises(AttributeError, match="config_file"):
            gd.config.set(config_file="p.toml")

    def test_get_unknown_setting(self) -> None:
        with pytest.raises(AttributeError, match="elipsoid"):
            gd.config.get("elipsoid")

    def test_refresh_rereads_files_and_drops_runtime_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_file = _write(tmp_path / "c.toml", 'scratch_dir = "/s1"\n')
        monkeypatch.setenv("GRID_DOCTOR_CONFIG", str(user_file))
        gd.config.set(ellipsoid="WGS84", weights_store="/w")
        user_file.write_text('scratch_dir = "/s2"\n')
        gd.config.refresh()
        assert gd.config.get() == {
            "ellipsoid": "sphere",
            "weights_store": None,
            "esmf_regrid_weightgen": None,
            "scratch_dir": "/s2",
        }
        assert gd.config.loaded_files[-1] == user_file

    def test_refresh_warns_and_skips_invalid_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        user_file = _write(tmp_path / "c.toml", 'scratch_dir = 1\nweights_store = "/w"\n')
        monkeypatch.setenv("GRID_DOCTOR_CONFIG", str(user_file))
        gd.config.set(scratch_dir="/dropped")
        with pytest.warns(config.ConfigWarning, match="Invalid 'scratch_dir' in .*c.toml"):
            gd.config.refresh()
        assert (gd.config.scratch_dir, gd.config.weights_store) == (None, "/w")

    def test_refresh_with_missing_named_file_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gd.config.set(scratch_dir="/keep")
        monkeypatch.setenv("GRID_DOCTOR_CONFIG", str(tmp_path / "missing.toml"))
        with pytest.raises(FileNotFoundError):
            gd.config.refresh()
        assert gd.config.scratch_dir == "/keep"


class TestLenientLoading:
    """At import and in ``refresh()`` config problems warn instead of failing."""

    @pytest.fixture
    def prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A fake ``sys.prefix`` without a config file."""
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
        return tmp_path / "env"

    def test_unknown_key_is_skipped_with_a_hint(self, prefix: Path, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'weight_store = "/w"\nscratch_dir = "/s"\n')
        with pytest.warns(config.ConfigWarning, match="did you mean 'weights_store'"):
            values, read = config._load({"GRID_DOCTOR_CONFIG": str(path)})
        assert (values, read) == ({"scratch_dir": "/s"}, (path,))

    def test_unknown_key_without_close_match_lists_settings(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'new_key = "hy"\n')
        with pytest.warns(config.ConfigWarning, match=r"'new_key'.*settings are \('ellipsoid'"):
            assert config._read_file(path, strict=False) == {}

    def test_invalid_value_is_skipped(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'ellipsoid = "GRS80"\nweights_store = "/w"\n')
        with pytest.warns(config.ConfigWarning, match="Invalid 'ellipsoid'"):
            assert config._read_file(path, strict=False) == {"weights_store": "/w"}

    def test_invalid_toml_skips_the_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", "ellipsoid = sphere\n")
        with pytest.warns(config.ConfigWarning, match="Invalid config file .*c.toml, ignored"):
            assert config._read_file(path, strict=False) == {}

    def test_invalid_environment_variable_is_skipped(self) -> None:
        env = {"GRID_DOCTOR_ELLIPSOID": "GRS80", "GRID_DOCTOR_SCRATCH_DIR": "/s"}
        with pytest.warns(config.ConfigWarning, match="Invalid GRID_DOCTOR_ELLIPSOID"):
            assert config._read_env(env) == {"scratch_dir": "/s"}
        with pytest.raises(ValueError, match="Invalid GRID_DOCTOR_ELLIPSOID"):
            config._read_env(env, strict=True)

    def test_strict_by_default_for_explicit_files(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'new_key = "hy"\n')
        with pytest.raises(ValueError, match="Unknown setting 'new_key'"):
            config._read_file(path)

    def _import(self, user_file: Path, *flags: str) -> subprocess.CompletedProcess[str]:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_CONFIG": str(user_file), "PYTHONPATH": src}
        script = "import grid_doctor as gd; print(gd.config.scratch_dir)"
        return subprocess.run(
            [sys.executable, *flags, "-c", script], env=env, capture_output=True, text=True
        )

    def test_import_works_with_a_broken_config(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'new_key = "hy"\nscratch_dir = "/s"\n')
        result = self._import(path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "/s"
        assert "ConfigWarning: Unknown setting 'new_key'" in result.stderr

    def test_warning_can_be_made_an_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "c.toml", 'new_key = "hy"\n')
        result = self._import(path, "-W", "error:::grid_doctor.config")
        assert result.returncode != 0
        assert "Unknown setting 'new_key'" in result.stderr


class TestConfigFiles:
    """The environment's and the user's config file, below environment variables."""

    @pytest.fixture
    def prefix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A fake ``sys.prefix``, as in a conda environment."""
        monkeypatch.setattr(sys, "prefix", str(tmp_path / "env"))
        return tmp_path / "env"

    def test_locations(
        self, prefix: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = prefix / "etc" / "grid-doctor" / "config.toml"
        assert config.environment_config_file() == env_file
        # The user file comes from platformdirs, which follows XDG_CONFIG_HOME on Linux.
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        user_file = config.user_config_file()
        if sys.platform.startswith("linux"):
            assert user_file == tmp_path / "xdg" / "grid-doctor" / "config.toml"
        assert (
            user_file.name == "config.toml" and user_file.parent.name == "grid-doctor"
        )
        assert config.config_files({}) == (env_file, user_file)
        assert config.config_files({"GRID_DOCTOR_CONFIG": " /etc/gd.toml "}) == (
            env_file,
            Path("/etc/gd.toml"),
        )
        assert config.config_files({"GRID_DOCTOR_CONFIG": ""}) == (env_file,)

    def test_precedence(self, prefix: Path, tmp_path: Path) -> None:
        env_file = _write(
            prefix / "etc" / "grid-doctor" / "config.toml",
            'ellipsoid = "WGS84"\nweights_store = "/shared/weights"\nscratch_dir = "/scratch"\n'
            'esmf_regrid_weightgen = "/opt/esmf/bin/ESMF_RegridWeightGen"\n',
        )
        user_file = _write(
            tmp_path / "user.toml", 'weights_store = "/mine"\nscratch_dir = ""\n'
        )
        values, read = config._load(
            {"GRID_DOCTOR_CONFIG": str(user_file), "GRID_DOCTOR_ELLIPSOID": "sphere"}
        )
        assert read == (env_file, user_file)
        assert values == {
            "ellipsoid": "sphere",  # environment variable over the environment's file
            "weights_store": "/mine",  # user's file over the environment's file
            "scratch_dir": None,  # an empty string unsets
            "esmf_regrid_weightgen": "/opt/esmf/bin/ESMF_RegridWeightGen",
        }

    def test_missing_files_are_skipped(
        self, prefix: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
        assert config._load({}) == ({}, ())

    def test_named_file_must_exist(self, prefix: Path, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="GRID_DOCTOR_CONFIG.*missing.toml"):
            config._load({"GRID_DOCTOR_CONFIG": str(tmp_path / "missing.toml")})

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (
                'weight_store = "/w"\n',
                r"Unknown setting 'weight_store' in .*bad.toml; did you mean 'weights_store'\?",
            ),
            (
                'ellipsoid = "GRS80"\n',
                r"Invalid 'ellipsoid' in .*bad.toml: ellipsoid must be one of",
            ),
            (
                "scratch_dir = 1\n",
                r"Invalid 'scratch_dir' in .*bad.toml: expected a string",
            ),
            ("[weights]\nstore = '/w'\n", r"Unknown setting 'weights'"),
            ("ellipsoid = sphere\n", r"Invalid config file .*bad.toml"),
        ],
        ids=["unknown-key", "bad-choice", "not-a-string", "table", "not-toml"],
    )
    def test_invalid_file_is_named_in_error(
        self, tmp_path: Path, text: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            config._read_file(_write(tmp_path / "bad.toml", text))

    def test_home_is_expanded_in_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        values = config._read_file(
            _write(
                tmp_path / "c.toml", 'ellipsoid = "sphere"\nscratch_dir = "~/scratch"\n'
            )
        )
        assert values == {
            "ellipsoid": "sphere",
            "scratch_dir": str(tmp_path / "scratch"),
        }

    def test_file_is_read_at_import(self, tmp_path: Path) -> None:
        user_file = _write(tmp_path / "c.toml", 'ellipsoid = "WGS84"\n')
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_CONFIG": str(user_file), "PYTHONPATH": src}
        env.pop("GRID_DOCTOR_ELLIPSOID", None)
        script = "import grid_doctor as gd; print(gd.config.ellipsoid, gd.config.loaded_files[-1])"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.split() == ["WGS84", str(user_file)]


class TestTemplate:
    """``python -m grid_doctor.config.template``."""

    def test_every_setting_is_described(self) -> None:
        assert list(template._HELP) == list(config.SETTINGS)

    def test_template_changes_nothing(self, tmp_path: Path) -> None:
        text = template.render()
        assert all(f"# {name} = " in text for name in config.SETTINGS)
        assert config._read_file(_write(tmp_path / "c.toml", text)) == {}

    def test_uncommented_examples_are_valid(self, tmp_path: Path) -> None:
        lines = [
            line.removeprefix("# ")
            if line.startswith(tuple(f"# {name} = " for name in config.SETTINGS))
            else line
            for line in template.render().splitlines()
        ]
        values = config._read_file(_write(tmp_path / "c.toml", "\n".join(lines)))
        assert sorted(values) == sorted(config.SETTINGS)

    def test_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert template.main(["--stdout"]) == 0
        assert capsys.readouterr().out == template.render()

    def test_output_directory_is_created(self, tmp_path: Path) -> None:
        assert template.main(["-o", str(tmp_path / "a" / "grid-doctor")]) == 0
        assert (
            tmp_path / "a" / "grid-doctor" / "config.toml"
        ).read_text() == template.render()

    def test_output_file(self, tmp_path: Path) -> None:
        assert template.main(["-o", str(tmp_path / "site.toml")]) == 0
        assert (tmp_path / "site.toml").is_file()

    def test_existing_file_needs_force(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = _write(tmp_path / "config.toml", 'ellipsoid = "WGS84"\n')
        assert template.main(["-o", str(tmp_path)]) == 1
        assert "--force" in capsys.readouterr().err
        assert target.read_text() == 'ellipsoid = "WGS84"\n'
        assert template.main(["-o", str(tmp_path), "--force"]) == 0
        assert target.read_text() == template.render()

    def test_stdout_and_output_exclude_each_other(self) -> None:
        with pytest.raises(SystemExit) as error:
            template.main(["--stdout", "-o", "x"])
        assert error.value.code == 2

    def test_default_is_user_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "user" / "config.toml"
        monkeypatch.setattr(template, "user_config_file", lambda: target)
        assert template.main([]) == 0
        assert target.read_text() == template.render()

    def test_set_writes_a_new_file(self, tmp_path: Path) -> None:
        argv = [
            "-o",
            str(tmp_path),
            "--set",
            "weights_store=/work/w",
            "--set",
            "ellipsoid=WGS84",
        ]
        assert template.main(argv) == 0
        values = config._read_file(tmp_path / "config.toml")
        assert values == {"weights_store": "/work/w", "ellipsoid": "WGS84"}

    def test_set_updates_an_existing_file(self, tmp_path: Path) -> None:
        target = _write(
            tmp_path / "config.toml",
            '# my notes\nellipsoid = "WGS84"\n# scratch_dir = "/path"\n',
        )
        argv = [
            "-o",
            str(target),
            "--set",
            "scratch_dir=/scratch/x",
            "--set",
            "weights_store=/w",
        ]
        assert template.main(argv) == 0
        assert target.read_text() == (
            '# my notes\nellipsoid = "WGS84"\n# scratch_dir = "/path"\nscratch_dir = "/scratch/x"\nweights_store = "/w"\n'
        )

    def test_set_keeps_unknown_keys_and_warns(self, tmp_path: Path) -> None:
        """An unknown key, e.g. from a newer version, does not block an update."""
        target = _write(tmp_path / "config.toml", 'new_key = "hy"\n')
        with pytest.warns(config.ConfigWarning, match="Unknown setting 'new_key'"):
            assert template.main(["-o", str(target), "--set", "scratch_dir=/new"]) == 0
        assert target.read_text() == 'new_key = "hy"\nscratch_dir = "/new"\n'

    def test_set_leaves_invalid_toml_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        text = "scratch_dir = /not-quoted\n"
        target = _write(tmp_path / "config.toml", text)
        assert template.main(["-o", str(target), "--set", "ellipsoid=WGS84"]) == 1
        assert "not valid TOML, left unchanged" in capsys.readouterr().err
        assert target.read_text() == text

    def test_set_quotes_values(self, tmp_path: Path) -> None:
        assert (
            template.main(["-o", str(tmp_path), "--set", 'scratch_dir=/a "b"\\c']) == 0
        )
        assert config._read_file(tmp_path / "config.toml") == {
            "scratch_dir": '/a "b"\\c'
        }

    @pytest.mark.parametrize(
        "item",
        ["weight_store=/w", "ellipsoid=GRS80", "scratch_dir"],
        ids=["unknown", "invalid", "no-equals"],
    )
    def test_invalid_set_writes_nothing(self, tmp_path: Path, item: str) -> None:
        with pytest.raises(SystemExit) as error:
            template.main(["-o", str(tmp_path), "--set", item])
        assert error.value.code == 2
        assert not (tmp_path / "config.toml").exists()

    def test_runs_as_module(self) -> None:
        src = str(Path(gd.__file__).parents[1])
        env = {**os.environ, "GRID_DOCTOR_CONFIG": "", "PYTHONPATH": src}
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                "-m",
                "grid_doctor.config.template",
                "--stdout",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.startswith("# grid-doctor configuration")


@pytest.mark.usefixtures("restore_config")
class TestEsmfSettings:
    """``esmf_regrid_weightgen`` and ``scratch_dir`` reach offline weight generation."""

    def test_executable(self) -> None:
        assert esmf_module.resolve_esmf_regrid_weightgen(None) == "ESMF_RegridWeightGen"
        with gd.config.set(esmf_regrid_weightgen="/opt/esmf/ESMF_RegridWeightGen"):
            assert (
                esmf_module.resolve_esmf_regrid_weightgen(None)
                == "/opt/esmf/ESMF_RegridWeightGen"
            )
            assert esmf_module.resolve_esmf_regrid_weightgen("mine") == "mine"

    def test_compute_weights_uses_configured_executable(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset
    ) -> None:
        passed: list[str] = []
        monkeypatch.setattr(
            regrid_module,
            "compute_healpix_weights_backend",
            lambda *args, offline, **kwargs: (
                passed.append(offline.esmf_regrid_weightgen) or Path("w.nc")
            ),
        )
        with gd.config.set(esmf_regrid_weightgen="/opt/esmf/ESMF_RegridWeightGen"):
            gd.compute_healpix_weights(regular_ds, 1)
            gd.compute_healpix_weights(regular_ds, 1, esmf_regrid_weightgen="mine")
        assert passed == ["/opt/esmf/ESMF_RegridWeightGen", "mine"]

    def test_scratch_dir_holds_temporary_workdir(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        description = describe_source(regular_ds, workdir=tmp_path)
        written: list[Path] = []
        monkeypatch.setattr(esmf_module.shutil, "which", lambda executable: "/bin/tool")
        monkeypatch.setattr(
            esmf_module,
            "write_ugrid_mesh_file",
            lambda mesh, path, **kwargs: written.append(Path(path)) or Path(path),
        )
        monkeypatch.setattr(
            esmf_module,
            "run_esmf_regrid_weightgen",
            lambda **kwargs: kwargs["weights_path"],
        )
        scratch = tmp_path / "scratch" / "not-yet"
        with gd.config.set(scratch_dir=str(scratch)):
            esmf_module._run_offline_esmf(
                source_desc=description,
                target_mesh=None,  # type: ignore[arg-type]  # only passed to the fake writer
                target_path=tmp_path / "w.nc",
                level=1,
                nest=True,
                method="nearest",
                config=esmf_module.OfflineWeightConfig(),
                ignore_unmapped=True,
            )
        assert written[0].parent.parent == scratch
        assert written[0].parent.name.startswith("grid_doctor_esmf_")
        assert (
            scratch.is_dir() and not written[0].parent.exists()
        )  # the run's directory is removed


_POINTS = xr.Dataset(
    {"t": ("obs", np.arange(4.0))},
    coords={
        "lat": ("obs", [10.0, 40.0, -30.0, 60.0]),
        "lon": ("obs", [0.0, 90.0, 180.0, 270.0]),
    },
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

        def fake_lonlat_to_healpix(
            lon: np.ndarray, lat: np.ndarray, **kwargs: object
        ) -> np.ndarray:
            return np.zeros(lon.size, dtype=np.int64)  # every point in cell 0

        def fake_geo_module(
            grid: HealpixGrid,
        ) -> tuple[SimpleNamespace, dict[str, str]]:
            passed["cells"] = grid.ellipsoid
            return SimpleNamespace(lonlat_to_healpix=fake_lonlat_to_healpix), {}

        def fake_coords(
            ds: xr.Dataset, cell_ids: np.ndarray, grid: HealpixGrid, **kwargs: object
        ) -> xr.Dataset:
            passed["coords"] = grid.ellipsoid
            return ds

        monkeypatch.setattr(swath, "_require_healpix_geo_module", fake_geo_module)
        monkeypatch.setattr(swath, "_scatter_to_dense", fake_coords)
        monkeypatch.setattr(swath, "_attach_sparse_coords", fake_coords)
        return passed

    @pytest.fixture
    def regrid_passed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> dict[str, Ellipsoid | None]:
        """Record the ``ellipsoid`` that regridding's weight generation and application receive.

        Application gets ``None`` unless an ellipsoid was given: it then
        takes the one the weight file was generated on.
        """
        passed: dict[str, Ellipsoid | None] = {}

        def fake_weights(*args: object, ellipsoid: Ellipsoid, **kwargs: object) -> Path:
            passed["weights"] = ellipsoid
            return tmp_path / "weights.nc"

        def fake_apply(
            ds: xr.Dataset, *args: object, ellipsoid: Ellipsoid | None, **kwargs: object
        ) -> xr.Dataset:
            passed["apply"] = ellipsoid
            return ds

        monkeypatch.setattr(
            regrid_module, "compute_healpix_weights_backend", fake_weights
        )
        monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)
        return passed

    def test_binning_default_is_sphere(
        self, binning_passed: dict[str, Ellipsoid], dense: bool
    ) -> None:
        # Pin the built-in default, whatever the environment says.
        with gd.config.set(ellipsoid="sphere"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_binning_none_uses_config(
        self, binning_passed: dict[str, Ellipsoid], dense: bool
    ) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense)
        assert binning_passed == {"cells": "WGS84", "coords": "WGS84"}

    def test_binning_argument_overrides_config(
        self, binning_passed: dict[str, Ellipsoid], dense: bool
    ) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.bin_to_healpix(_POINTS, 1, dense=dense, ellipsoid="sphere")
        assert binning_passed == {"cells": "sphere", "coords": "sphere"}

    def test_regrid_default_is_sphere(
        self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset
    ) -> None:
        # Pin the built-in default, whatever the environment says.
        with gd.config.set(ellipsoid="sphere"):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "sphere", "apply": None}

    def test_regrid_none_uses_config(
        self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset
    ) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1)
        assert regrid_passed == {"weights": "WGS84", "apply": None}

    def test_regrid_argument_overrides_config(
        self, regrid_passed: dict[str, Ellipsoid | None], regular_ds: xr.Dataset
    ) -> None:
        with gd.config.set(ellipsoid="WGS84"):
            gd.regrid_to_healpix(regular_ds, 1, ellipsoid="sphere")
        assert regrid_passed == {"weights": "sphere", "apply": "sphere"}
