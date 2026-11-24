"""Planning, serialising and validating fan-out manifests (``grid_doctor.io._manifest``)."""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import grid_doctor as gd
from grid_doctor.io import (
    ArraySpec,
    LevelSpec,
    ManifestError,
    PyramidManifest,
    WriteTask,
)
from grid_doctor.io._manifest import Region, plan_tasks

from ..helpers import _weight_file
from ..io.fanout_helpers import _WEIGHTS, _pyramid


@pytest.fixture(autouse=True, scope="module")
def _weights(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Synthetic weights, so the tests do not need ESMPy."""
    _WEIGHTS[:] = [_weight_file(tmp_path_factory.mktemp("weights") / "w.nc")]


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
        with pytest.raises(
            ManifestError, match="intervals along 'time' overlap at index 2"
        ):
            PyramidManifest.from_dict(manifest_dict)

    def test_gap_between_regions(self, manifest_dict: dict[str, Any]) -> None:
        manifest_dict["tasks"].remove(_timed(manifest_dict)[1])
        with pytest.raises(ManifestError, match="gap along 'time' at index 4"):
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
# Multi-dimensional splits: the partition check in detail
# ----------------------------------------------------------------------

# Manifest order deliberately differs from alphabetical order.
_DIMS = ("time", "lev", "ens")


def _grid_manifest(
    sizes: dict[str, int], cuts: dict[str, list[int]]
) -> tuple[PyramidManifest, list[Region]]:
    """A store-less manifest whose tasks form the grid given by *cuts*."""
    dims = tuple(sizes)
    spec = ArraySpec(
        "v", (*dims, "cell"), (*sizes.values(), 12), (*(1 for _ in dims), 12), "f4"
    )
    level = LevelSpec(0, "level_0.zarr", (), (spec,))
    per_dim = [
        [(a, b) for a, b in zip([0, *cuts[d]], [*cuts[d], sizes[d]])] for d in dims
    ]
    regions = [
        tuple((d, a, b) for d, (a, b) in zip(dims, combo))
        for combo in itertools.product(*per_dim)
    ]
    return _manifest_for(level, dims, regions), regions


def _manifest_for(
    level: LevelSpec, dims: tuple[str, ...], regions: list[Region]
) -> PyramidManifest:
    tasks = tuple(
        WriteTask(f"task-{i:04d}", "m1", "/x", 2, ("v",), r, (0,))
        for i, r in enumerate(regions)
    )
    return PyramidManifest(
        "m1", "/x", 2, "now", "test", tuple((d, 1) for d in dims), (level,), tasks
    )


class TestMultiDimensionalPartition:
    sizes: ClassVar[dict[str, int]] = {"time": 6, "lev": 4, "ens": 3}
    cuts: ClassVar[dict[str, list[int]]] = {
        "time": [2, 4],
        "lev": [1, 3],
        "ens": [1, 2],
    }

    def test_planner_grid_is_accepted(self) -> None:
        manifest, _ = _grid_manifest(self.sizes, self.cuts)
        assert len(manifest.tasks) == 3 * 3 * 3

    def test_entry_order_inside_regions_is_irrelevant(self) -> None:
        manifest, regions = _grid_manifest(self.sizes, self.cuts)
        shuffled = [tuple(reversed(r)) for r in regions]
        _manifest_for(manifest.levels[0], _DIMS, shuffled)

    @pytest.mark.parametrize("dim", _DIMS)
    def test_overlap_in_any_dimension(self, dim: str) -> None:
        manifest, regions = _grid_manifest(self.sizes, self.cuts)
        index = _DIMS.index(dim)
        first = list(regions[-1])
        name, start, stop = first[index]
        first[index] = (name, start - 1, stop)  # reach into the neighbour
        regions[-1] = tuple(first)
        with pytest.raises(ManifestError, match=f"along '{dim}' overlap"):
            _manifest_for(manifest.levels[0], _DIMS, regions)

    @pytest.mark.parametrize("dim", _DIMS)
    def test_gap_in_any_dimension(self, dim: str) -> None:
        manifest, regions = _grid_manifest(self.sizes, self.cuts)
        cut = self.cuts[dim][0]
        kept = [r for r in regions if {d: a for d, a, _ in r}[dim] != cut]
        with pytest.raises(ManifestError, match=f"gap along '{dim}'"):
            _manifest_for(manifest.levels[0], _DIMS, kept)

    def test_missing_combination(self) -> None:
        manifest, regions = _grid_manifest(self.sizes, self.cuts)
        with pytest.raises(ManifestError, match="26 task region.*full grid of 27"):
            _manifest_for(manifest.levels[0], _DIMS, regions[:5] + regions[6:])

    def test_duplicate_region_with_reordered_entries(self) -> None:
        manifest, regions = _grid_manifest(self.sizes, self.cuts)
        regions.append(tuple(reversed(regions[4])))
        with pytest.raises(ManifestError, match="more than one task"):
            _manifest_for(manifest.levels[0], _DIMS, regions)

    def test_never_accepts_an_invalid_tiling(self) -> None:
        """Randomised: every accepted manifest writes each element exactly once."""
        rng = random.Random(54)
        accepted = 0
        for _ in range(3000):
            ndim = rng.randint(1, 3)
            dims = _DIMS[:ndim]
            sizes = {d: rng.randint(1, 5) for d in dims}
            cuts = {
                d: sorted(rng.sample(range(1, n), rng.randint(0, n - 1)))
                for d, n in sizes.items()
            }
            manifest, regions = _grid_manifest(sizes, cuts)
            for _ in range(rng.randint(0, 2)):  # random damage
                i = rng.randrange(len(regions))
                entries = list(regions[i])
                op = rng.choice(["drop", "dup", "widen", "shift", "reorder"])
                if op == "drop" and len(regions) > 1:
                    regions.pop(i)
                elif op == "dup":
                    regions.append(regions[i])
                elif op == "reorder":
                    rng.shuffle(entries)
                    regions[i] = tuple(entries)
                elif op in ("widen", "shift"):
                    k = rng.randrange(ndim)
                    d, a, b = entries[k]
                    a, b = (a - 1, b + 1) if op == "widen" else (a + 1, b + 1)
                    if 0 <= a <= b <= sizes[d]:
                        entries[k] = (d, a, b)
                        regions[i] = tuple(entries)
            count = np.zeros(tuple(sizes.values()), int)
            for region in regions:
                bounds = {d: slice(a, b) for d, a, b in region}
                count[tuple(bounds[d] for d in dims)] += 1
            try:
                _manifest_for(manifest.levels[0], dims, regions)
            except ManifestError:
                continue
            accepted += 1
            assert (count == 1).all(), (sizes, regions)
        assert accepted > 500  # the undamaged grids and harmless reorders
