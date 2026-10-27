"""Manifest-driven fan-out writes: init -> tasks -> workers -> check."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import s3fs
import xarray as xr
from fsspec.implementations.memory import MemoryFileSystem

import grid_doctor as gd
from grid_doctor.helpers import create_healpix_pyramid, save_pyramid
from grid_doctor.io import (
    ArraySpec,
    LevelSpec,
    ManifestError,
    PyramidManifest,
    StaleTaskError,
    StoreExistsError,
    StoreReport,
    WriteTask,
)
from grid_doctor.io import _fanout as fanout
from grid_doctor.io._fanout import IncompleteStoreError
from grid_doctor.io._manifest import plan_tasks

from .test_chunked_pyramid import _weight_file

MAX_LEVEL = 3
NTIME = 10
NLAT, NLON = 8, 16


def _source(ntime: int = NTIME, *, time_chunk: int = 2) -> xr.Dataset:
    rng = np.random.default_rng(3)
    ta = rng.normal(280, 5, size=(ntime, NLAT, NLON)).astype("float32")
    ta[:, :2, :] = np.nan  # some missing data, like a regional source
    ds = xr.Dataset(
        {
            "ta": (("time", "lat", "lon"), ta),
            "pr": (("time", "lat", "lon"), rng.random((ntime, NLAT, NLON))),
            "lsm": (("lat", "lon"), (rng.random((NLAT, NLON)) > 0.5).astype("f4")),
        },
        coords={
            "time": np.arange(ntime),
            "lat": np.linspace(-80, 80, NLAT),
            "lon": np.linspace(0, 337.5, NLON),
        },
    )
    return ds.chunk(time=time_chunk)


_WEIGHTS: list[Path] = []


@pytest.fixture(autouse=True, scope="module")
def _weights(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Synthetic weights, so the tests do not need ESMPy."""
    _WEIGHTS[:] = [_weight_file(tmp_path_factory.mktemp("weights") / "w.nc")]


def _pyramid(source: xr.Dataset | None = None) -> dict[int, xr.Dataset]:
    return create_healpix_pyramid(
        _source() if source is None else source,
        max_level=MAX_LEVEL,
        min_level=0,
        weights_path=_WEIGHTS[0],
        cell_chunks=192,
    )


def _run_all(
    pyramid: dict[int, xr.Dataset], tasks: list[WriteTask] | tuple[WriteTask, ...]
) -> list[WriteTask]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda t: gd.write_pyramid_task(pyramid, t), tasks))


def _assert_store_equals_pyramid(path: Path, pyramid: dict[int, xr.Dataset]) -> None:
    for level, expected in pyramid.items():
        stored = xr.open_zarr(path / f"level_{level}.zarr")
        for name in expected.data_vars:
            np.testing.assert_allclose(
                stored[name].values, expected[name].values, equal_nan=True
            )


def _spec(
    name: str, dims: tuple[str, ...], shape: tuple[int, ...], chunks: tuple[int, ...]
) -> ArraySpec:
    return ArraySpec(name=name, dims=dims, shape=shape, chunks=chunks, dtype="float32")


# ----------------------------------------------------------------------
# Planning (pure)
# ----------------------------------------------------------------------


class TestPlanTasks:
    def _levels(self) -> list[LevelSpec]:
        return [
            LevelSpec(
                level=2,
                store="level_2.zarr",
                coords=("time", "cell"),
                data_vars=(
                    _spec("ta", ("time", "cell"), (10, 192), (2, 192)),
                    _spec("pr", ("time", "cell"), (10, 192), (3, 192)),
                    _spec("lsm", ("cell",), (192,), (192,)),
                ),
            ),
            LevelSpec(
                level=1,
                store="level_1.zarr",
                coords=("time", "cell"),
                data_vars=(
                    _spec("ta", ("time", "cell"), (10, 48), (2, 48)),
                    _spec("pr", ("time", "cell"), (10, 48), (3, 48)),
                    _spec("lsm", ("cell",), (48,), (48,)),
                ),
            ),
        ]

    def test_split_is_rounded_to_common_write_unit(self) -> None:
        # chunks 2 and 3 along time -> boundaries must be multiples of 6.
        split, tasks = plan_tasks(self._levels(), split={"time": 4})
        assert split == (("time", 6),)
        regions = sorted({t.region for t in tasks if t.region})
        assert regions == [(("time", 0, 6),), (("time", 6, 10),)]

    def test_variables_sharing_split_dims_are_grouped(self) -> None:
        _, tasks = plan_tasks(self._levels(), split={"time": 6})
        timed = [t for t in tasks if t.region]
        assert all(t.variables == ("ta", "pr") for t in timed)
        assert all(t.levels == (2, 1) for t in tasks)

    def test_static_variable_gets_one_unsplit_task(self) -> None:
        _, tasks = plan_tasks(self._levels(), split={"time": 6})
        static = [t for t in tasks if not t.region]
        assert [t.variables for t in static] == [("lsm",)]

    def test_variables_per_task(self) -> None:
        _, tasks = plan_tasks(self._levels(), split={"time": 6}, variables_per_task=1)
        timed = [t for t in tasks if t.region]
        assert len(timed) == 4
        assert {t.variables for t in timed} == {("ta",), ("pr",)}

    def test_no_split_gives_single_task(self) -> None:
        split, tasks = plan_tasks(self._levels())
        assert split == ()
        assert len(tasks) == 1
        assert set(tasks[0].variables) == {"ta", "pr", "lsm"}

    def test_two_split_dims_give_product(self) -> None:
        levels = [
            LevelSpec(
                level=0,
                store="level_0.zarr",
                coords=(),
                data_vars=(
                    _spec("t", ("time", "lev", "cell"), (4, 6, 12), (1, 2, 12)),
                ),
            )
        ]
        split, tasks = plan_tasks(levels, split={"time": 2, "lev": 3})
        assert split == (("time", 2), ("lev", 4))
        assert len(tasks) == 2 * 2
        assert (("time", 2, 4), ("lev", 4, 6)) in {t.region for t in tasks}

    def test_regions_cover_dimension_exactly_once(self) -> None:
        _, tasks = plan_tasks(self._levels(), split={"time": 1})
        covered = sorted(
            (start, stop) for t in tasks if t.region for _, start, stop in t.region
        )
        flat = [i for start, stop in covered for i in range(start, stop)]
        assert flat == list(range(10))

    @pytest.mark.parametrize(
        ("split", "message"),
        [
            ({"cell": 48}, "cannot split along 'cell'"),
            ({"depth": 2}, "not a dimension of any data variable"),
            ({"time": 0}, "positive int"),
        ],
    )
    def test_invalid_split(self, split: dict[str, int], message: str) -> None:
        with pytest.raises(ManifestError, match=message):
            plan_tasks(self._levels(), split=split)

    def test_inconsistent_length_is_rejected(self) -> None:
        levels = self._levels()
        bad = LevelSpec(
            level=0,
            store="level_0.zarr",
            coords=(),
            data_vars=(_spec("ta", ("time", "cell"), (9, 12), (1, 12)),),
        )
        with pytest.raises(ManifestError, match="different lengths"):
            plan_tasks([*levels, bad], split={"time": 2})


# ----------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------


class TestSerialisation:
    def test_manifest_json_round_trip(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        again = PyramidManifest.from_json(manifest.to_json())
        assert again == manifest
        # Shared fields are not repeated per task in the file.
        data = json.loads(manifest.to_json())
        assert "manifest_id" not in data["tasks"][0]
        assert data["tasks"][0]["region"] in ({"time": [0, 4]}, {})

    def test_task_json_round_trip(self, tmp_path: Path) -> None:
        manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        for task in manifest.tasks:
            assert WriteTask.from_json(task.to_json()) == task

    def test_reflow_codec_round_trip(self, tmp_path: Path) -> None:
        reflow = pytest.importorskip("reflow")
        codec = reflow.ManifestCodec()
        manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        assert codec.loads(codec.dumps(manifest)) == manifest
        tasks = list(manifest.tasks)
        assert codec.loads(codec.dumps(tasks)) == tasks

    def test_unknown_schema_version(self) -> None:
        with pytest.raises(ManifestError, match="schema version 99"):
            PyramidManifest.from_dict({"schema_version": 99})

    def test_slices(self) -> None:
        task = WriteTask("t", "m", "/x", 2, ("a",), (("time", 4, 8),), (1,))
        assert task.slices == {"time": slice(4, 8)}
        assert task.isel() == task.slices


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
            _pyramid(source), str(tmp_path), split={"time": 4}
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
    cachable = False

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
# Validation of manifests and tasks
# ----------------------------------------------------------------------


@pytest.fixture
def manifest_dict(tmp_path: Path) -> dict[str, Any]:
    """JSON form of a real manifest: time chunk 2, tasks of 4 steps."""
    manifest = gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
    data: dict[str, Any] = json.loads(manifest.to_json())
    return data


def _timed(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in data["tasks"] if t["region"]]


class TestValidation:
    def test_valid_manifest_round_trips(self, manifest_dict: dict[str, Any]) -> None:
        manifest = PyramidManifest.from_dict(manifest_dict)
        assert manifest.to_dict() == manifest_dict

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            (["levels"], r"manifest: missing field 'levels'"),
            (["tasks", 0, "variables"], r"tasks\[0\]: missing field 'variables'"),
            (
                ["levels", 0, "data_vars", 0, "chunks"],
                r"levels\[0\]\.data_vars\[0\]: missing field 'chunks'",
            ),
        ],
    )
    def test_missing_field_is_named(
        self, manifest_dict: dict[str, Any], path: list[Any], message: str
    ) -> None:
        parent: Any = manifest_dict
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
        with pytest.raises(ManifestError, match=message):
            PyramidManifest.from_dict(manifest_dict)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda d: d.update(zarr_format=4), "zarr_format must be 2 or 3"),
            (lambda d: d.update(zarr_format=True), "zarr_format must be 2 or 3"),
            (lambda d: d.update(split={"time": 0}), r"split\['time'\] must be >= 1"),
            (lambda d: d.update(split={"cell": 4}), "spatial dimension 'cell'"),
            (lambda d: d.update(split=[["time", 4]]), "split must be an object"),
            (
                lambda d: d["levels"][0]["data_vars"][0].update(dims="time"),
                "dims must be a list, got str",
            ),
            (
                lambda d: d["levels"][0]["data_vars"][0].update(shape=[True, 3]),
                "shape entry must be an integer",
            ),
            (
                lambda d: d["levels"][0]["data_vars"][0].update(chunks=[2]),
                "must have the same length",
            ),
            (
                lambda d: d["levels"][0]["data_vars"][0].update(chunks=[0, 192]),
                "chunks entry must be >= 1",
            ),
            (lambda d: d["levels"][0].update(store="../x.zarr"), "store '../x.zarr'"),
            (lambda d: d["tasks"][0].update(task_id="../evil"), "task_id '../evil'"),
            (lambda d: d["tasks"][0].update(variables=[]), "must not be empty"),
            (lambda d: d["tasks"][0].update(variables=["a/b"]), "not a valid array"),
            (lambda d: d["tasks"][0].update(region={"time": [4, 0]}), "stop < start"),
            (lambda d: d["tasks"][0].update(region={"time": [0]}), r"\[start, stop\]"),
            (lambda d: d.update(schema_version="1"), "schema version '1'"),
        ],
    )
    def test_bad_field(
        self,
        manifest_dict: dict[str, Any],
        mutate: Callable[[dict[str, Any]], None],
        message: str,
    ) -> None:
        mutate(manifest_dict)
        with pytest.raises(ManifestError, match=message):
            PyramidManifest.from_dict(manifest_dict)

    def test_misaligned_region(self, manifest_dict: dict[str, Any]) -> None:
        _timed(manifest_dict)[1]["region"] = {"time": [3, 8]}
        with pytest.raises(ManifestError, match="start 3 is not a multiple of"):
            PyramidManifest.from_dict(manifest_dict)

    def test_misaligned_region_end(self, manifest_dict: dict[str, Any]) -> None:
        _timed(manifest_dict)[0]["region"] = {"time": [0, 3]}
        with pytest.raises(ManifestError, match="stop 3 is neither a multiple"):
            PyramidManifest.from_dict(manifest_dict)

    def test_region_outside_array(self, manifest_dict: dict[str, Any]) -> None:
        _timed(manifest_dict)[-1]["region"] = {"time": [8, 12]}
        with pytest.raises(ManifestError, match="exceeds length 10"):
            PyramidManifest.from_dict(manifest_dict)

    def test_overlapping_regions(self, manifest_dict: dict[str, Any]) -> None:
        _timed(manifest_dict)[1]["region"] = {"time": [2, 8]}
        with pytest.raises(ManifestError, match="tasks overlap along 'time'"):
            PyramidManifest.from_dict(manifest_dict)

    def test_gap_between_regions(self, manifest_dict: dict[str, Any]) -> None:
        manifest_dict["tasks"].remove(_timed(manifest_dict)[1])
        with pytest.raises(ManifestError, match="tasks gap along 'time' at index 4"):
            PyramidManifest.from_dict(manifest_dict)

    def test_duplicate_region(self, manifest_dict: dict[str, Any]) -> None:
        clone = dict(_timed(manifest_dict)[0], task_id="task-999999")
        manifest_dict["tasks"].append(clone)
        with pytest.raises(ManifestError, match="more than one task"):
            PyramidManifest.from_dict(manifest_dict)

    def test_duplicate_task_id(self, manifest_dict: dict[str, Any]) -> None:
        manifest_dict["tasks"][1]["task_id"] = manifest_dict["tasks"][0]["task_id"]
        with pytest.raises(ManifestError, match="task ids has duplicates"):
            PyramidManifest.from_dict(manifest_dict)

    def test_uncovered_variable(self, manifest_dict: dict[str, Any]) -> None:
        manifest_dict["tasks"] = _timed(manifest_dict)  # drops the 'lsm' task
        with pytest.raises(ManifestError, match="'lsm' on level .* not written"):
            PyramidManifest.from_dict(manifest_dict)

    def test_unknown_variable_and_level(self, manifest_dict: dict[str, Any]) -> None:
        manifest_dict["tasks"][0]["variables"] = ["nope"]
        with pytest.raises(ManifestError, match="no data variable 'nope'"):
            PyramidManifest.from_dict(manifest_dict)
        manifest_dict["tasks"][0]["variables"] = ["ta"]
        manifest_dict["tasks"][0]["levels"] = [17]
        with pytest.raises(ManifestError, match="level 17 is not in the manifest"):
            PyramidManifest.from_dict(manifest_dict)

    def test_region_dim_must_be_split_dim(self, manifest_dict: dict[str, Any]) -> None:
        _timed(manifest_dict)[0]["region"] = {"time": [0, 4], "lat": [0, 1]}
        with pytest.raises(ManifestError, match=r"\['lat'\] are not split"):
            PyramidManifest.from_dict(manifest_dict)

    def test_invalid_json(self) -> None:
        with pytest.raises(ManifestError, match="manifest: invalid JSON"):
            PyramidManifest.from_json("{not json")
        with pytest.raises(ManifestError, match="task: expected an object"):
            WriteTask.from_json("[1, 2]")

    def test_direct_construction_is_validated(
        self, manifest_dict: dict[str, Any]
    ) -> None:
        """reflow rebuilds dataclasses with cls(**fields): no from_dict involved."""
        manifest = PyramidManifest.from_dict(manifest_dict)
        task = next(t for t in manifest.tasks if t.region)
        shifted = dataclasses.replace(task, region=(("time", 1, 4),))
        tasks = tuple(shifted if t == task else t for t in manifest.tasks)
        with pytest.raises(ManifestError, match="not a multiple of"):
            dataclasses.replace(manifest, tasks=tasks)
        with pytest.raises(ManifestError, match="differs from the manifest"):
            dataclasses.replace(
                manifest,
                tasks=(dataclasses.replace(task, manifest_id="other"),)
                + manifest.tasks[1:],
            )
        with pytest.raises(ManifestError, match="schema version 2"):
            dataclasses.replace(manifest, schema_version=2)

    def test_reflow_codec_is_validated(self, manifest_dict: dict[str, Any]) -> None:
        reflow = pytest.importorskip("reflow")
        codec = reflow.ManifestCodec()
        task = PyramidManifest.from_dict(manifest_dict).tasks[0]
        payload = codec.dumps(task)
        assert '"zarr_format":2' in payload
        with pytest.raises(ManifestError, match="zarr_format must be 2 or 3"):
            codec.loads(payload.replace('"zarr_format":2', '"zarr_format":5'))

    def test_lists_are_normalised_to_tuples(self) -> None:
        listed = WriteTask(
            task_id="task-1",
            manifest_id="abc",
            path="/x",
            zarr_format=2,
            variables=["ta"],  # type: ignore[arg-type]
            region=[["time", 0, 4]],  # type: ignore[arg-type]
            levels=[3],  # type: ignore[arg-type]
        )
        tupled = WriteTask("task-1", "abc", "/x", 2, ("ta",), (("time", 0, 4),), (3,))
        assert listed == tupled
        assert hash(listed) == hash(tupled)

    def test_worker_rejects_edited_task(self, tmp_path: Path) -> None:
        pyramid = _pyramid()
        manifest = gd.init_pyramid_store(pyramid, str(tmp_path), split={"time": 4})
        first, second = [t for t in manifest.tasks if t.region][:2]
        # Well-formed on its own, but not the task the manifest planned.
        edited = dataclasses.replace(first, region=second.region)
        with pytest.raises(ManifestError, match="differs from the task stored"):
            gd.write_pyramid_task(pyramid, edited)
        unknown = dataclasses.replace(first, task_id="task-424242")
        with pytest.raises(ManifestError, match="is not part of manifest"):
            gd.write_pyramid_task(pyramid, unknown)
        assert not (tmp_path / "_grid_doctor" / "done").exists()

    def test_corrupt_stored_manifest(self, tmp_path: Path) -> None:
        gd.init_pyramid_store(_pyramid(), str(tmp_path), split={"time": 4})
        stored = tmp_path / "_grid_doctor" / "manifest.json"
        data = json.loads(stored.read_text())
        del data["tasks"][0]["levels"]
        stored.write_text(json.dumps(data))
        with pytest.raises(ManifestError, match=r"tasks\[0\]: missing field 'levels'"):
            gd.load_manifest(str(tmp_path))


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
            ({"split": {"time": 4}, "variables_per_task": 1}, "tasks: 4 stored, 7 planned"),
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
            patch.setattr("grid_doctor.helpers.finish_init", killed)
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
