"""Fan-out error paths and edge cases: damaged stores, odd inputs, bad manifests.

The happy paths live in ``test_fanout.py``; this module covers what the
fan-out does when something is not as planned.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr

import grid_doctor as gd
from grid_doctor.io import (
    ArraySpec,
    LevelSpec,
    ManifestError,
    PyramidManifest,
    StoreExistsError,
    StoreReport,
)
from grid_doctor.io import _fanout as fanout
from grid_doctor.io._manifest import plan_tasks

from . import test_fanout as base
from .test_chunked_pyramid import _weight_file

STATE = "_grid_doctor"


@pytest.fixture(autouse=True, scope="module")
def _weights(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Synthetic weights, shared with the helpers of ``test_fanout``."""
    base._WEIGHTS[:] = [_weight_file(tmp_path_factory.mktemp("weights") / "w.nc")]


def _init(tmp_path: Path, **kwargs: Any) -> tuple[dict[int, xr.Dataset], Any]:
    pyramid = base._pyramid()
    return pyramid, gd.init_pyramid_store(
        pyramid, str(tmp_path), split={"time": 4}, **kwargs
    )


def _reconsolidate(store: Path) -> None:
    """Make zarr see changes made to a store behind its back."""
    zarr.consolidate_metadata(str(store), zarr_format=2)


def _replace_with_group(store: Path, name: str) -> None:
    shutil.rmtree(store / name)
    zarr.open_group(str(store), mode="r+", zarr_format=2).create_group(name)
    _reconsolidate(store)


def _timed_task(manifest: PyramidManifest) -> Any:
    return next(t for t in manifest.tasks if t.region and "ta" in t.variables)


# ----------------------------------------------------------------------
# Corrupt state files
# ----------------------------------------------------------------------


class TestCorruptStateFiles:
    def test_unreadable_manifest_is_not_current(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        (tmp_path / STATE / "manifest.json").write_text("{not json")
        assert not gd.is_current(manifest)
        assert not gd.is_task_done(manifest.tasks[0])

    def test_unreadable_marker_is_not_done(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        task = manifest.tasks[0]
        done = tmp_path / STATE / "done"
        done.mkdir(parents=True)
        (done / f"{task.task_id}.json").write_text("{not json")
        assert not gd.is_task_done(task)

    def test_unreadable_manifest_blocks_init(self, tmp_path: Path) -> None:
        _init(tmp_path)
        (tmp_path / STATE / "manifest.json").write_text("{not json")
        with pytest.raises(StoreExistsError, match="an unreadable manifest"):
            _init(tmp_path)

    def test_killed_init_is_reported(self, tmp_path: Path) -> None:
        """No manifest, and an ``init.json`` that is not even valid JSON."""
        _, manifest = _init(tmp_path)
        (tmp_path / STATE / "manifest.json").unlink()
        (tmp_path / STATE / "init.json").write_text("{truncated")
        report = gd.check_pyramid_store(manifest)
        assert not report.complete
        (problem,) = [p for p in report.problems if "no manifest" in p]
        assert "an init started ? on host ?" in problem

    def test_foreign_files_in_done_are_ignored(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        done = tmp_path / STATE / "done"
        done.mkdir(parents=True)
        (done / "README").write_text("not a marker")
        (done / ".task-000000.json.1234.tmp").write_text("{half")
        (done / ".hidden.json").write_text("{half")
        assert gd.check_pyramid_store(manifest).stale_markers == 0

    def test_summary_mentions_stale_markers(self) -> None:
        report = StoreReport("m", 2, ("a",), ("b",), (), 3, ())
        assert "Ignored 3 stale marker(s)." in report.summary()


# ----------------------------------------------------------------------
# Damaged stores
# ----------------------------------------------------------------------


class TestDamagedStore:
    def test_missing_coordinate_and_array(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        store = tmp_path / f"level_{base.MAX_LEVEL}.zarr"
        shutil.rmtree(store / "time")
        shutil.rmtree(store / "pr")
        _reconsolidate(store)
        problems = gd.check_pyramid_store(manifest).problems
        assert f"level {base.MAX_LEVEL}: coordinate 'time' missing" in problems
        assert f"level {base.MAX_LEVEL}: array 'pr' missing" in problems

    def test_array_replaced_by_group_is_missing(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        _replace_with_group(tmp_path / "level_0.zarr", "ta")
        assert "level 0: array 'ta' missing" in gd.check_pyramid_store(manifest).problems

    def test_changed_layout_is_reported(self, tmp_path: Path) -> None:
        _, manifest = _init(tmp_path)
        store = tmp_path / "level_0.zarr"
        spec = manifest.level(0).array("lsm")
        group = zarr.open_group(str(store), mode="r+", zarr_format=2)
        group.create_array(
            "lsm",
            shape=(spec.shape[0] + 1,),
            chunks=(5,),
            dtype="int16",
            attributes={"_ARRAY_DIMENSIONS": ["cell"]},
            overwrite=True,
        )
        _reconsolidate(store)
        problems = " | ".join(gd.check_pyramid_store(manifest).problems)
        assert f"'lsm' has shape ({spec.shape[0] + 1},)" in problems
        assert "'lsm' has write unit (5,)" in problems
        assert f"'lsm' has dtype int16, manifest says {spec.dtype}" in problems

    def test_inspect_chunks_counts_a_deleted_array(self, tmp_path: Path) -> None:
        """Chunk files gone, metadata still there: all chunks count as absent."""
        pyramid, manifest = _init(tmp_path)
        base._run_all(pyramid, manifest.tasks)
        shutil.rmtree(tmp_path / "level_0.zarr" / "pr")
        report = gd.check_pyramid_store(manifest, inspect_chunks=True)
        assert report.complete, "markers decide completeness, not chunks"
        writes_pr = {t.task_id for t in manifest.tasks if "pr" in t.variables}
        assert writes_pr <= {task_id for task_id, _, _ in report.absent_chunks}

    def test_inspect_chunks_skips_unopenable_levels(self, tmp_path: Path) -> None:
        pyramid, manifest = _init(tmp_path)
        base._run_all(pyramid, manifest.tasks)
        shutil.rmtree(tmp_path / "level_1.zarr")
        report = gd.check_pyramid_store(manifest, inspect_chunks=True)
        assert any(p.startswith("level 1: cannot open") for p in report.problems)
        assert report.missing_tasks == ()


# ----------------------------------------------------------------------
# Resume against a different pyramid
# ----------------------------------------------------------------------


class TestResumeDifferences:
    def _resume(self, tmp_path: Path, pyramid: dict[int, xr.Dataset]) -> Any:
        return gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, if_exists="resume"
        )

    def test_other_levels(self, tmp_path: Path) -> None:
        pyramid, _ = _init(tmp_path)
        fewer = {k: v for k, v in pyramid.items() if k != 0}
        with pytest.raises(StoreExistsError, match=r"levels \[0, 1, 2, 3\] != \[1, 2, 3\]"):
            self._resume(tmp_path, fewer)

    def test_array_replaced_by_group(self, tmp_path: Path) -> None:
        pyramid, _ = _init(tmp_path)
        _replace_with_group(tmp_path / "level_2.zarr", "ta")
        with pytest.raises(StoreExistsError, match="level 2: .*is not a zarr array"):
            self._resume(tmp_path, pyramid)

    def test_other_coordinates(self, tmp_path: Path) -> None:
        pyramid, _ = _init(tmp_path)
        extra = {k: v.assign_coords(member=0) for k, v in pyramid.items()}
        with pytest.raises(StoreExistsError, match="layout coords"):
            self._resume(tmp_path, extra)

    def test_other_variables(self, tmp_path: Path) -> None:
        pyramid, _ = _init(tmp_path)
        fewer = {k: v.drop_vars("pr") for k, v in pyramid.items()}
        with pytest.raises(StoreExistsError, match=r"variables \['lsm', 'pr', 'ta'\] != \['lsm', 'ta'\]"):
            self._resume(tmp_path, fewer)

    def test_in_memory_variables_resume(self, tmp_path: Path) -> None:
        """Computed (numpy) variables have no chunks to compare; fine."""
        pyramid, manifest = _init(tmp_path)
        loaded = {k: v.assign(lsm=v["lsm"].compute()) for k, v in pyramid.items()}
        assert self._resume(tmp_path, loaded) == manifest


# ----------------------------------------------------------------------
# Coordinate comparison
# ----------------------------------------------------------------------


class TestCoordinateDifferences:
    @pytest.fixture
    def store(self, tmp_path: Path) -> tuple[str, xr.Dataset]:
        ds = xr.Dataset(
            coords={
                "time": ("time", np.arange(3)),
                "bounds": (("time", "nv"), np.arange(6).reshape(3, 2)),
                "latitude": ("cell", np.linspace(-45.0, 45.0, 4)),
            }
        )
        path = str(tmp_path / "coords.zarr")
        ds.to_zarr(path, zarr_format=2, consolidated=True)
        return path, ds

    def _diff(self, store: str, ds: xr.Dataset) -> list[str]:
        # "absent" is in neither the dataset nor the store: skipped.
        return fanout.coordinate_differences(
            ds, store, ["time", "bounds", "latitude", "absent"], zarr_format=2
        )

    def test_transposed_coordinate_matches(self, store: tuple[str, xr.Dataset]) -> None:
        path, ds = store
        assert ds["bounds"].dims == ("time", "nv")
        assert self._diff(path, ds.transpose("nv", "time", "cell")) == []

    def test_spatial_coordinates_are_not_compared(
        self, store: tuple[str, xr.Dataset]
    ) -> None:
        path, ds = store
        moved = ds.assign_coords(latitude=("cell", np.zeros(4)))
        assert self._diff(path, moved) == []

    def test_other_dims_are_a_difference(self, store: tuple[str, xr.Dataset]) -> None:
        path, ds = store
        other = ds.drop_vars("bounds").assign_coords(
            bounds=(("time", "edge"), np.arange(6).reshape(3, 2))
        )
        (problem,) = self._diff(path, other)
        assert "coordinate 'bounds': dims ('time', 'edge') != stored" in problem


# ----------------------------------------------------------------------
# Worker input
# ----------------------------------------------------------------------


class TestWorkerInput:
    def test_transposed_pyramid_is_written_correctly(self, tmp_path: Path) -> None:
        pyramid, manifest = _init(tmp_path)
        flipped = {k: v.transpose("cell", "time") for k, v in pyramid.items()}
        base._run_all(flipped, manifest.tasks)
        assert gd.check_pyramid_store(manifest).complete
        base._assert_store_equals_pyramid(tmp_path, pyramid)

    def test_missing_variable(self, tmp_path: Path) -> None:
        pyramid, manifest = _init(tmp_path)
        without = {k: v.drop_vars("ta") for k, v in pyramid.items()}
        with pytest.raises(ManifestError, match="variable 'ta' missing on level"):
            gd.write_pyramid_task(without, _timed_task(manifest))

    def test_other_dimensions(self, tmp_path: Path) -> None:
        pyramid, manifest = _init(tmp_path)
        renamed = {k: v.rename({"time": "step"}) for k, v in pyramid.items()}
        with pytest.raises(ManifestError, match="has dims .*the store expects"):
            gd.write_pyramid_task(renamed, _timed_task(manifest))

    def test_wrong_length_outside_the_region(self, tmp_path: Path) -> None:
        pyramid, manifest = _init(tmp_path)
        cut = {k: v.isel(cell=slice(0, 10)) for k, v in pyramid.items()}
        with pytest.raises(ManifestError, match="10 elements along 'cell', the store"):
            gd.write_pyramid_task(cut, _timed_task(manifest))


# ----------------------------------------------------------------------
# init_pyramid_store arguments
# ----------------------------------------------------------------------


class TestInitArguments:
    def test_invalid_structure(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="structure must be 'flat' or 'multiscales'"):
            gd.init_pyramid_store(base._pyramid(), str(tmp_path), structure="tree")  # type: ignore[call-overload]

    def test_unknown_materialise_name_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="grid_doctor"):
            manifest = gd.init_pyramid_store(
                base._pyramid(), str(tmp_path), materialise=["lsm", "nope"]
            )
        assert "materialise names unknown variables: nope" in caplog.text
        assert manifest.materialised == ("lsm",)

    def test_variables_per_task_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="variables_per_task must be a positive"):
            gd.init_pyramid_store(base._pyramid(), str(tmp_path), variables_per_task=0)


# ----------------------------------------------------------------------
# Manifest validation
# ----------------------------------------------------------------------


@pytest.fixture
def manifest_dict(tmp_path: Path) -> dict[str, Any]:
    _, manifest = _init(tmp_path)
    data: dict[str, Any] = json.loads(manifest.to_json())
    return data


def _untimed(data: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in data["tasks"] if not t["region"])


class TestManifestValidation:
    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda d: d["levels"][0]["data_vars"].append(
                    dict(d["levels"][0]["data_vars"][0])
                ),
                "duplicate data variables",
            ),
            (
                lambda d: d["levels"][0]["coords"].append(
                    d["levels"][0]["data_vars"][0]["name"]
                ),
                "listed as coordinate and data variable",
            ),
            (lambda d: d["levels"].append(dict(d["levels"][0])), "is listed twice"),
            (lambda d: d["tasks"][0].update(task_id=""), "must be a non-empty string"),
            (lambda d: d["tasks"][0].update(levels=[]), "levels must not be empty"),
            (
                lambda d: d["tasks"][0].update(region=[["time", 0]]),
                r"region entries must be \(dim, start, stop\)",
            ),
            (
                lambda d: _untimed(d).update(region={"time": [0, 4]}),
                "has no dimension 'time'",
            ),
            (
                lambda d: next(t for t in d["tasks"] if t["region"]).update(region={}),
                "split along different dimensions",
            ),
        ],
    )
    def test_bad_manifest(
        self,
        manifest_dict: dict[str, Any],
        mutate: Callable[[dict[str, Any]], None],
        message: str,
    ) -> None:
        mutate(manifest_dict)
        with pytest.raises(ManifestError, match=message):
            PyramidManifest.from_dict(manifest_dict)

    def test_objects_of_the_wrong_type(self, manifest_dict: dict[str, Any]) -> None:
        manifest = PyramidManifest.from_dict(manifest_dict)
        with pytest.raises(ManifestError, match="levels must contain LevelSpec"):
            dataclasses.replace(manifest, levels=("level_0",))  # type: ignore[arg-type]
        with pytest.raises(ManifestError, match="data_vars must contain ArraySpec"):
            dataclasses.replace(manifest.levels[0], data_vars=("ta",))  # type: ignore[arg-type]
        with pytest.raises(ManifestError, match=r"split entries must be \(dim, size\)"):
            dataclasses.replace(manifest, split=(("time",),))  # type: ignore[arg-type]

    def test_lookups_of_unknown_names(self, manifest_dict: dict[str, Any]) -> None:
        manifest = PyramidManifest.from_dict(manifest_dict)
        with pytest.raises(KeyError, match="manifest has no level 17"):
            manifest.level(17)
        with pytest.raises(KeyError, match=r"unknown task ids: \['nope'\]"):
            manifest.select_tasks(["nope"])

    def test_split_dims_must_agree_across_levels(self) -> None:
        def level(n: int, dims: tuple[str, ...], shape: tuple[int, ...]) -> LevelSpec:
            spec = ArraySpec("ta", dims, shape, shape, "float32")
            return LevelSpec(n, f"level_{n}.zarr", (), (spec,))

        levels = [level(1, ("time", "cell"), (4, 48)), level(0, ("cell",), (12,))]
        with pytest.raises(ManifestError, match="has split dimensions"):
            plan_tasks(levels, split={"time": 2})
