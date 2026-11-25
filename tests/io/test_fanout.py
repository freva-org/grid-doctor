"""Manifest-driven fan-out writes: init -> tasks -> workers -> check (``grid_doctor.io._fanout``)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import s3fs
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem

import grid_doctor as gd
from grid_doctor.io import (
    ManifestError,
    StaleTaskError,
    StoreExistsError,
    StoreReport,
)
from grid_doctor.io import _fanout as fanout
from grid_doctor.io._fanout import IncompleteStoreError
from grid_doctor.io._save import save_pyramid

from ..helpers import _weight_file
from ..io.fanout_helpers import _WEIGHTS, MAX_LEVEL, _assert_store_equals_pyramid, _pyramid, _run_all, _source


@pytest.fixture(autouse=True, scope="module")
def _weights(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Synthetic weights, so the tests do not need ESMPy."""
    _WEIGHTS[:] = [_weight_file(tmp_path_factory.mktemp("weights") / "w.nc")]


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------


class TestFanOut:
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_parallel_tasks_match_serial_save(
        self, tmp_path: Path, zarr_format: int
    ) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(
            pyramid,
            str(tmp_path / "fanout"),
            split={"time": 2},
            zarr_format=zarr_format,  # type: ignore[arg-type]
        )
        assert gd.check_pyramid_store(manifest).missing_tasks == tuple(
            t.task_id for t in manifest.tasks
        )
        assert _run_all(pyramid, manifest.tasks) == list(manifest.tasks)
        report = gd.check_pyramid_store(manifest)
        assert report.complete, report.summary()
        assert bool(report)

        save_pyramid(pyramid, str(tmp_path / "serial"), mode="w")
        for level in pyramid:
            serial = xr.open_zarr(tmp_path / "serial" / f"level_{level}.zarr")
            fanout = xr.open_zarr(tmp_path / "fanout" / f"level_{level}.zarr")
            xr.testing.assert_identical(fanout.compute(), serial.compute())

    def test_manifest_is_persisted(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(_pyramid(), f"{tmp_path}/", split={"time": 4})
        assert manifest.path == str(tmp_path)
        assert gd.load_manifest(str(tmp_path)) == manifest
        assert (tmp_path / "_grid_doctor" / "manifest.json").is_file()

    def test_levels_are_finest_first(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path))
        assert [s.level for s in manifest.levels] == list(range(MAX_LEVEL, -1, -1))

    def test_worker_may_pass_presubset_pyramid(self, tmp_path: Path) -> None:
        """Workers can regrid only their slice of the source."""
        source = _source()
        manifest = gd.init_pyramid_store(
            _pyramid(source.copy()), str(tmp_path), split={"time": 4}
        )
        for task in manifest.tasks:
            sub_source = source.isel(task.isel())
            gd.write_pyramid_task(_pyramid(sub_source), task)
        assert gd.check_pyramid_store(str(tmp_path)).complete
        _assert_store_equals_pyramid(tmp_path, _pyramid(source))

    def test_misaligned_worker_chunks_are_rechunked(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A worker chunked differently from the init must not split chunks."""
        manifest = gd.init_pyramid_store(
            _pyramid(_source(time_chunk=2)), str(tmp_path), split={"time": 4}
        )
        odd = _pyramid(_source(time_chunk=3))
        with caplog.at_level(logging.WARNING):
            _run_all(odd, manifest.tasks)
        assert "Rechunking" in caplog.text
        _assert_store_equals_pyramid(tmp_path, odd)

    def test_numpy_backed_pyramid(self, tmp_path: Path) -> None:
        pyramid = {lvl: ds.compute() for lvl, ds in _pyramid().items()}
        chunked = _pyramid()
        manifest = gd.init_pyramid_store(chunked, str(tmp_path), split={"time": 4})
        _run_all(pyramid, manifest.tasks)
        assert gd.check_pyramid_store(manifest).complete
        _assert_store_equals_pyramid(tmp_path, pyramid)

    def test_materialised_variables_are_not_tasks(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, materialise=["lsm"]
        )
        assert manifest.materialised == ("lsm",)
        assert all("lsm" not in t.variables for t in manifest.tasks)
        assert "lsm" in manifest.levels[0].coords
        stored = xr.open_zarr(tmp_path / f"level_{MAX_LEVEL}.zarr")["lsm"]
        np.testing.assert_allclose(
            stored.values, pyramid[MAX_LEVEL]["lsm"].values, equal_nan=True
        )

    def test_zero_dimensional_variable(self, tmp_path: Path) -> None:
        pyramid = {
            lvl: ds.assign(scale=xr.DataArray(np.float32(2.5)))
            for lvl, ds in _pyramid().items()
        }
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        _run_all(pyramid, manifest.tasks)
        assert gd.check_pyramid_store(manifest).complete
        stored = xr.open_zarr(tmp_path / "level_0.zarr")
        assert float(stored["scale"]) == 2.5

    def test_sharded_store_splits_on_shards(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        ncell = {lvl: ds.sizes["cell"] for lvl, ds in pyramid.items()}
        encoding = {
            lvl: {name: {"shards": (4, ncell[lvl])} for name in ("ta", "pr")}
            for lvl in pyramid
        }
        manifest = gd.init_pyramid_store(
            pyramid,
            str(tmp_path),
            zarr_format=3,
            encoding=encoding,
            split={"time": 2},
        )
        assert manifest.split == (("time", 4),)
        assert manifest.levels[0].array("ta").chunks[0] == 4
        _run_all(pyramid, manifest.tasks)
        report = gd.check_pyramid_store(manifest, inspect_chunks=True)
        assert report.complete
        assert report.absent_chunks == ()
        resumed = gd.init_pyramid_store(
            pyramid,
            str(tmp_path),
            zarr_format=3,
            encoding=encoding,
            split={"time": 2},
            if_exists="resume",
        )
        assert resumed == manifest
        _assert_store_equals_pyramid(tmp_path, pyramid)
# ----------------------------------------------------------------------
# Failure handling and the completeness check
# ----------------------------------------------------------------------


class TestCheckAndFailures:
    def test_missing_tasks_are_reported_and_retryable(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 2})
        first, *rest = manifest.tasks
        _run_all(pyramid, rest[1:])

        report = gd.check_pyramid_store(manifest)
        assert not report.complete
        assert report.missing_tasks == (first.task_id, rest[0].task_id)
        assert "INCOMPLETE" in report.summary()
        with pytest.raises(IncompleteStoreError, match=first.task_id):
            report.raise_if_incomplete()

        _run_all(pyramid, manifest.select_tasks(report.missing_tasks))
        assert gd.check_pyramid_store(manifest).complete
        _assert_store_equals_pyramid(tmp_path, pyramid)

    def test_rerunning_a_task_is_harmless(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        _run_all(pyramid, manifest.tasks)
        _run_all(pyramid, manifest.tasks[:1])
        assert gd.check_pyramid_store(manifest).complete
        _assert_store_equals_pyramid(tmp_path, pyramid)

    def test_reinit_invalidates_old_tasks_and_markers(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        old = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        _run_all(pyramid, old.tasks)
        new = gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, if_exists="overwrite"
        )
        assert new.manifest_id != old.manifest_id

        with pytest.raises(StaleTaskError, match="re-initialised"):
            gd.write_pyramid_task(pyramid, old.tasks[0])
        report = gd.check_pyramid_store(new)
        assert report.done_tasks == ()
        stale = gd.check_pyramid_store(old)
        assert any("re-initialised" in p for p in stale.problems)
        assert not stale.complete

    def test_stale_and_corrupt_markers_are_ignored(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        done = tmp_path / "_grid_doctor" / "done"
        done.mkdir(parents=True)
        task_id = manifest.tasks[0].task_id
        (done / f"{task_id}.json").write_text(
            json.dumps({"manifest_id": "other", "task_id": task_id})
        )
        (done / "garbage.json").write_text("{not json")
        report = gd.check_pyramid_store(manifest)
        assert report.stale_markers == 2
        assert task_id in report.missing_tasks

    def test_structural_damage_is_a_problem(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        _run_all(pyramid, manifest.tasks)
        shutil.rmtree(tmp_path / "level_1.zarr")
        report = gd.check_pyramid_store(manifest)
        assert not report.complete
        assert any("level 1" in p for p in report.problems)
        assert report.missing_tasks == ()

    def test_inspect_chunks_reports_fill_value_chunks(self, tmp_path: Path) -> None:
        source = _source()
        source["ta"][:4] = np.nan  # first task region of 'ta' is all fill value
        pyramid = _pyramid(source)
        manifest = gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, variables_per_task=1
        )
        _run_all(pyramid, manifest.tasks)
        report = gd.check_pyramid_store(manifest, inspect_chunks=True)
        assert report.complete, "absent chunks must not make a store incomplete"
        ta_first = next(
            t.task_id
            for t in manifest.tasks
            if t.variables == ("ta",) and t.region[0][1] == 0
        )
        absent = {task_id: (a, e) for task_id, a, e in report.absent_chunks}
        assert ta_first in absent
        assert absent[ta_first][0] == absent[ta_first][1]
        assert "chunk(s) not stored" in report.summary()

    def test_wrong_pyramid_is_rejected(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        task = next(t for t in manifest.tasks if t.region)
        shorter = _pyramid(_source(ntime=7))
        with pytest.raises(ManifestError, match="expected the full length 10"):
            gd.write_pyramid_task(shorter, task)
        missing_level = {k: v for k, v in _pyramid().items() if k != 0}
        with pytest.raises(ManifestError, match="level 0"):
            gd.write_pyramid_task(missing_level, task)

    def test_cache_checks(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        task = manifest.tasks[0]
        assert gd.is_current(manifest) and gd.is_current(task)
        assert not gd.is_task_done(task)
        gd.write_pyramid_task(pyramid, task)
        assert gd.is_task_done(task)
        gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, if_exists="overwrite"
        )
        assert not gd.is_current(manifest)
        assert not gd.is_task_done(task)
        shutil.rmtree(tmp_path)
        assert not gd.is_current(manifest)

    def test_no_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="run init_pyramid_store first"):
            gd.load_manifest(str(tmp_path))

    def test_report_is_serialisable_for_reflow(self) -> None:
        reflow = pytest.importorskip("reflow")
        report = StoreReport("m", 2, ("a",), ("b",), ("p",), 1, (("a", 1, 2),))
        codec = reflow.ManifestCodec()
        assert codec.loads(codec.dumps(report)) == report


class _FakeS3(MemoryFileSystem):  # type: ignore[misc]
    """In-memory stand-in for :class:`s3fs.S3FileSystem`."""

    protocol = ("s3", "memory")
    root_marker = ""
    cacheable = False

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        path = str(path)
        for prefix in ("s3://", "memory://"):
            path = path.removeprefix(prefix)
        return path.rstrip("/")


def test_s3_target(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = _FakeS3()
    monkeypatch.setattr(s3fs, "S3FileSystem", lambda **_: fs)
    pyramid = _pyramid()
    path = "s3://bucket/pyramid"
    manifest = gd.init_pyramid_store(pyramid, path, {}, split={"time": 4})
    assert fs.exists("bucket/pyramid/_grid_doctor/manifest.json")
    for task in manifest.tasks[1:]:
        gd.write_pyramid_task(pyramid, task, {})
    assert gd.check_pyramid_store(path, {}).missing_tasks == (
        manifest.tasks[0].task_id,
    )
    gd.write_pyramid_task(pyramid, manifest.tasks[0], {})
    report = gd.check_pyramid_store(path, {}, inspect_chunks=True)
    assert report.complete, report.summary()
    for level, expected in pyramid.items():
        stored = xr.open_zarr(s3fs.S3Map(root=f"{path}/level_{level}.zarr", s3=fs))
        np.testing.assert_allclose(
            stored["ta"].values, expected["ta"].values, equal_nan=True
        )
    assert (
        gd.init_pyramid_store(pyramid, path, {}, split={"time": 4}, if_exists="resume")
        == manifest
    )
    gd.init_pyramid_store(pyramid, path, {}, split={"time": 4}, if_exists="overwrite")
    assert not fs.exists("bucket/pyramid/_grid_doctor/done")
    assert not fs.exists("bucket/pyramid/_grid_doctor/init.json")


# ----------------------------------------------------------------------
# Existing targets: error / resume / overwrite, interrupted inits
# ----------------------------------------------------------------------


class TestIfExists:
    def test_default_refuses_existing_pyramid(self, tmp_path: Path) -> None:
        gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        with pytest.raises(StoreExistsError, match="already holds a pyramid"):
            gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})

    def test_refuses_foreign_store(self, tmp_path: Path) -> None:
        save_pyramid(_pyramid(), str(tmp_path), mode="w")
        with pytest.raises(StoreExistsError, match="level stores level_3.zarr"):
            gd.init_pyramid_store(_pyramid(), str(tmp_path))
        with pytest.raises(StoreExistsError, match="no readable manifest"):
            gd.init_pyramid_store(_pyramid(), str(tmp_path), if_exists="resume")

    def test_invalid_if_exists(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="if_exists must be"):
            gd.init_pyramid_store(_pyramid(), str(tmp_path), if_exists="w")  # type: ignore[arg-type]

    def test_resume_on_empty_target_initialises(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(
            _pyramid(), str(tmp_path), split={"time": 4}, if_exists="resume"
        )
        assert gd.load_manifest(str(tmp_path)) == manifest

    def test_resume_keeps_manifest_and_progress(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        half = manifest.tasks[: len(manifest.tasks) // 2]
        _run_all(pyramid, half)
        meta = tmp_path / f"level_{MAX_LEVEL}.zarr" / ".zmetadata"
        before = meta.stat().st_mtime_ns

        # A re-submitted job: same build, same parameters.
        resumed = gd.init_pyramid_store(
            _pyramid(), str(tmp_path), split={"time": 4}, if_exists="resume"
        )
        assert resumed == manifest
        assert meta.stat().st_mtime_ns == before, "resume must not write"
        report = gd.check_pyramid_store(resumed)
        assert report.done_tasks == tuple(t.task_id for t in half)

        _run_all(pyramid, resumed.select_tasks(report.missing_tasks))
        assert gd.check_pyramid_store(resumed).complete
        _assert_store_equals_pyramid(tmp_path, pyramid)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"split": {"time": 2}}, "split"),
            (
                {"split": {"time": 4}, "variables_per_task": 1},
                "tasks: 4 stored, 7 planned",
            ),
            ({"split": {"time": 4}, "materialise": ["lsm"]}, "materialise"),
            ({"split": {"time": 4}, "zarr_format": 3}, "zarr_format 2 != 3"),
        ],
    )
    def test_resume_refuses_different_plan(
        self, tmp_path: Path, kwargs: dict[str, Any], message: str
    ) -> None:
        gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        with pytest.raises(StoreExistsError, match=message):
            gd.init_pyramid_store(
                _pyramid(), str(tmp_path), if_exists="resume", **kwargs
            )

    def test_resume_refuses_different_data(self, tmp_path: Path) -> None:
        gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        with pytest.raises(StoreExistsError, match="layout"):
            gd.init_pyramid_store(
                _pyramid(_source(ntime=12)),
                str(tmp_path),
                split={"time": 4},
                if_exists="resume",
            )
        with pytest.raises(StoreExistsError, match="chunks"):
            gd.init_pyramid_store(
                _pyramid(_source(time_chunk=1)),
                str(tmp_path),
                split={"time": 4},
                if_exists="resume",
            )

    def test_resume_refuses_moved_store(self, tmp_path: Path) -> None:
        gd.init_pyramid_store(_pyramid(), str(tmp_path / "a"), split={"time": 4})
        shutil.copytree(tmp_path / "a", tmp_path / "b")
        with pytest.raises(StoreExistsError, match="path"):
            gd.init_pyramid_store(
                _pyramid(), str(tmp_path / "b"), split={"time": 4}, if_exists="resume"
            )

    def test_overwrite_discards(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        old = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        _run_all(pyramid, old.tasks)
        new = gd.init_pyramid_store(
            pyramid, str(tmp_path), split={"time": 4}, if_exists="overwrite"
        )
        assert new.manifest_id != old.manifest_id
        assert gd.check_pyramid_store(new).done_tasks == ()
        stored = xr.open_zarr(tmp_path / f"level_{MAX_LEVEL}.zarr")["ta"].values
        assert np.isnan(stored).all()

    def test_interrupted_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An init killed by its time limit leaves stores but no manifest."""
        monkeypatch.setenv("SLURM_JOB_ID", "4711")

        def killed(*args: Any, **kwargs: Any) -> None:
            raise KeyboardInterrupt  # stands in for SIGTERM/SIGKILL

        with monkeypatch.context() as patch:
            patch.setattr("grid_doctor.io._save.finish_init", killed)
            with pytest.raises(KeyboardInterrupt):
                gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        assert (tmp_path / "_grid_doctor" / "init.json").is_file()
        assert not (tmp_path / "_grid_doctor" / "manifest.json").exists()

        with pytest.raises(ManifestError, match="Slurm job 4711.*not finished"):
            gd.load_manifest(str(tmp_path))
        with pytest.raises(StoreExistsError, match="has not finished"):
            gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        with pytest.raises(StoreExistsError, match="no readable manifest"):
            gd.init_pyramid_store(
                _pyramid(), str(tmp_path), split={"time": 4}, if_exists="resume"
            )

        manifest = gd.init_pyramid_store(
            _pyramid(), str(tmp_path), split={"time": 4}, if_exists="overwrite"
        )
        assert not (tmp_path / "_grid_doctor" / "init.json").exists()
        assert gd.load_manifest(str(tmp_path)) == manifest

    def test_reinit_during_write_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        original = fanout.store_all

        def store_all_while_reinit(writes: Any) -> None:
            # Someone forces a re-init after the worker checked the manifest.
            gd.init_pyramid_store(
                pyramid, str(tmp_path), split={"time": 4}, if_exists="overwrite"
            )
            original(writes)

        monkeypatch.setattr(fanout, "store_all", store_all_while_reinit)
        with pytest.raises(StaleTaskError, match="re-initialised while"):
            gd.write_pyramid_task(pyramid, manifest.tasks[0])
        assert gd.check_pyramid_store(str(tmp_path)).done_tasks == ()
# ----------------------------------------------------------------------
# Coordinate values: resume and workers must not relabel data
# ----------------------------------------------------------------------


def _labelled(
    start: int, *, ntime: int = 4, time_chunk: int = 2, **coords: Any
) -> dict[int, xr.Dataset]:
    ds = xr.Dataset(
        {"t": (("time", "cell"), np.full((ntime, 12), float(start)))},
        coords={"time": np.arange(start, start + ntime), **coords},
    )
    return {0: ds.chunk(time=time_chunk, cell=12)}


def _hourly(start: str, ntime: int = 6) -> dict[int, xr.Dataset]:
    time = pd.date_range(start, periods=ntime, freq="h")
    ds = xr.Dataset(
        {"t": (("time", "cell"), np.arange(ntime * 12, dtype="f4").reshape(ntime, 12))},
        coords={"time": time},
    )
    ds["time"].encoding = {"units": "hours since 2000-01-01", "dtype": "int32"}
    return {0: ds.chunk(time=2, cell=12)}


class TestCoordinateValues:
    def test_review_case_resume_refuses_other_time_range(self, tmp_path: Path) -> None:
        old, new = _labelled(0), _labelled(100)
        manifest = gd.init_pyramid_store(old, str(tmp_path), split={"time": 2})
        gd.write_pyramid_task(old, manifest.tasks[0])
        with pytest.raises(
            StoreExistsError, match="coordinate 'time' differs at time=0"
        ):
            gd.init_pyramid_store(
                new, str(tmp_path), split={"time": 2}, if_exists="resume"
            )

    def test_review_case_worker_refuses_other_time_range(self, tmp_path: Path) -> None:
        old, new = _labelled(0), _labelled(100)
        manifest = gd.init_pyramid_store(old, str(tmp_path), split={"time": 2})
        gd.write_pyramid_task(old, manifest.tasks[0])
        with pytest.raises(ManifestError, match="pyramid has 102, store has 2"):
            gd.write_pyramid_task(new, manifest.tasks[1])
        stored = xr.open_zarr(tmp_path / "level_0.zarr")["t"][:, 0].values
        np.testing.assert_array_equal(stored, [0.0, 0.0, np.nan, np.nan])
        assert not gd.check_pyramid_store(manifest).complete

    def test_non_split_coordinate_is_checked(self, tmp_path: Path) -> None:
        lev = xr.DataArray([1000.0, 850.0], dims="lev")
        pyramid = {
            0: xr.Dataset(
                {"t": (("time", "lev", "cell"), np.zeros((4, 2, 12)))},
                coords={"time": np.arange(4), "lev": lev},
            ).chunk(time=2, lev=2, cell=12)
        }
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 2})
        other = {0: pyramid[0].assign_coords(lev=[925.0, 850.0])}
        with pytest.raises(ManifestError, match="coordinate 'lev' differs at lev=0"):
            gd.write_pyramid_task(other, manifest.tasks[0])
        with pytest.raises(StoreExistsError, match="coordinate 'lev'"):
            gd.init_pyramid_store(
                other, str(tmp_path), split={"time": 2}, if_exists="resume"
            )

    def test_encoded_times_resume_and_presubset_workers(self, tmp_path: Path) -> None:
        """CF-encoded int32 hours: same dates pass, shifted dates fail."""
        pyramid = _hourly("2020-01-01")
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 2})
        source = pyramid[0]
        for task in manifest.tasks:
            gd.write_pyramid_task({0: source.isel(task.isel())}, task)
        resumed = gd.init_pyramid_store(
            _hourly("2020-01-01"), str(tmp_path), split={"time": 2}, if_exists="resume"
        )
        assert resumed == manifest and gd.check_pyramid_store(resumed).complete

        shifted = _hourly("2020-01-01T01")
        with pytest.raises(ManifestError, match="coordinate 'time' differs at time=2"):
            gd.write_pyramid_task(
                {0: shifted[0].isel(manifest.tasks[1].isel())}, manifest.tasks[1]
            )
        with pytest.raises(StoreExistsError, match="2020-01-01T01"):
            gd.init_pyramid_store(
                shifted, str(tmp_path), split={"time": 2}, if_exists="resume"
            )

    def test_downcast_and_nan_coordinates_compare_as_stored(
        self, tmp_path: Path
    ) -> None:
        """float64 labels stored as float32, with a NaN label, still match."""
        depth = xr.DataArray([0.1, np.nan, 1 / 3], dims="depth")
        ds = xr.Dataset(
            {"t": (("time", "depth", "cell"), np.zeros((4, 3, 12)))},
            coords={"time": np.arange(4), "depth": depth},
        ).chunk(time=2, depth=3, cell=12)
        ds["depth"].encoding = {"dtype": "float32"}
        pyramid = {0: ds}
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 2})
        _run_all(pyramid, manifest.tasks)
        assert (
            gd.init_pyramid_store(
                pyramid, str(tmp_path), split={"time": 2}, if_exists="resume"
            )
            == manifest
        )
        assert gd.check_pyramid_store(manifest).complete


class TestMultiscalesStructure:
    """``structure="multiscales"`` writes a datatree template, no tasks yet."""

    def test_returns_none_and_writes_one_store(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        result = gd.init_pyramid_store(
            pyramid, str(tmp_path / "tree.zarr"), structure="multiscales"
        )
        assert result is None
        tree = xr.open_datatree(tmp_path / "tree.zarr", engine="zarr")
        assert {f"level_{level}" for level in pyramid} <= set(tree.children)
        assert not (tmp_path / "tree.zarr" / "_grid_doctor").exists()

    @pytest.mark.parametrize(
        "kwargs",
        [{"split": {"time": 2}}, {"variables_per_task": 1}, {"if_exists": "resume"}],
    )
    def test_fan_out_options_are_rejected(
        self, tmp_path: Path, kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(NotImplementedError, match="multiscales"):
            gd.init_pyramid_store(
                _pyramid(), str(tmp_path / "tree.zarr"), structure="multiscales", **kwargs
            )
        assert not (tmp_path / "tree.zarr").exists()
