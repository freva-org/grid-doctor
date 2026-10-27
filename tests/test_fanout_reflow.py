"""The reflow workflow from the fan-out recipe, run end to end with run_local."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
import xarray as xr

import grid_doctor as gd

from .test_chunked_pyramid import _weight_file
from .test_fanout import MAX_LEVEL, _source

reflow = pytest.importorskip("reflow")
Param, Result, Workflow = reflow.Param, reflow.Result, reflow.Workflow

wf = Workflow("healpix-fanout-test")


def build_pyramid(
    source: str, weights: str, region: dict[str, slice] | None = None
) -> dict[int, xr.Dataset]:
    ds = xr.open_zarr(source)
    if region:
        ds = ds.isel(region)
    return gd.create_healpix_pyramid(
        ds, max_level=MAX_LEVEL, weights_path=weights, cell_chunks=192
    )


# Params are not part of reflow's cache identity (2604.4), so a cached init
# could hand out tasks for another target.  Always run it; "resume" makes a
# re-submit continue the existing store instead of wiping it.
@wf.job(cache=False)
def init(
    source: Annotated[str, Param(help="Input file")],
    weights: Annotated[str, Param(help="Weight file")],
    target: Annotated[str, Param(help="Pyramid root")],
    time_per_task: Annotated[int, Param(help="Time steps per task")] = 4,
    overwrite: Annotated[bool, Param(help="Discard an existing store")] = False,
) -> list[gd.WriteTask]:
    manifest = gd.init_pyramid_store(
        build_pyramid(source, weights),
        target,
        split={"time": time_per_task},
        if_exists="overwrite" if overwrite else "resume",
    )
    return list(manifest.tasks)


@wf.job(array=True, verify=gd.is_task_done)
def write(
    task: Annotated[gd.WriteTask, Result(step="init")],
    source: Annotated[str, Param(help="Input file")],
    weights: Annotated[str, Param(help="Weight file")],
) -> gd.WriteTask:
    # Test hooks: record every call, and let one element die like a job
    # that hit its time limit.
    if log := os.environ.get("GD_TEST_LOG"):
        with open(log, "a") as handle:
            handle.write(task.task_id + "\n")
    if os.environ.get("GD_TEST_FAIL") == task.task_id:
        raise RuntimeError("DUE TO TIME LIMIT")
    return gd.write_pyramid_task(build_pyramid(source, weights, task.isel()), task)


@wf.job()
def check(
    done: Annotated[list[gd.WriteTask], Result(step="write")],
    target: Annotated[str, Param(help="Pyramid root")],
) -> gd.StoreReport:
    report = gd.check_pyramid_store(target)
    report.raise_if_incomplete()
    return report


def _failed(run: object) -> list[tuple[str, int | None]]:
    info = run.status(as_dict=True)  # type: ignore[attr-defined]
    return [(i["task_name"], i.get("array_index")) for i in info["failed_instances"]]


@pytest.fixture
def _no_fork() -> Iterator[None]:
    """Forked workers inherit the parent's dask/numba thread state and can
    deadlock; real array jobs are separate processes anyway."""
    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    yield
    multiprocessing.set_start_method(previous, force=True)


@pytest.mark.usefixtures("_no_fork")
@pytest.mark.parametrize("max_workers", [1, 3])
def test_reflow_run_local(tmp_path: Path, max_workers: int) -> None:
    source = tmp_path / "source.zarr"
    _source().to_zarr(source)
    weights = _weight_file(tmp_path / "weights.nc")
    target = tmp_path / "pyramid"

    run = wf.run_local(
        run_dir=tmp_path / "run",
        max_workers=max_workers,
        source=str(source),
        weights=str(weights),
        target=str(target),
        time_per_task=4,
    )
    assert _failed(run) == [], run.status(errors=True)

    report = gd.check_pyramid_store(str(target))
    assert report.complete, report.summary()
    expected = build_pyramid(str(source), str(weights))
    for level, ds in expected.items():
        stored = xr.open_zarr(target / f"level_{level}.zarr")
        for name in ds.data_vars:
            np.testing.assert_allclose(
                stored[name].values, ds[name].values, equal_nan=True
            )


def test_resubmit_after_timeout_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain re-submit continues the store and only redoes the failed task."""
    source = tmp_path / "source.zarr"
    _source().to_zarr(source)
    weights = _weight_file(tmp_path / "weights.nc")
    target = tmp_path / "pyramid"
    log = tmp_path / "calls.txt"
    params = {"source": str(source), "weights": str(weights), "target": str(target)}
    monkeypatch.setenv("GD_TEST_LOG", str(log))

    monkeypatch.setenv("GD_TEST_FAIL", "task-000001")
    first = wf.run_local(run_dir=tmp_path / "run1", on_error="continue", **params)
    assert _failed(first) == [("write", 1)]
    manifest = gd.load_manifest(str(target))
    assert gd.check_pyramid_store(manifest).missing_tasks == ("task-000001",)

    monkeypatch.delenv("GD_TEST_FAIL")
    log.write_text("")
    second = wf.run_local(run_dir=tmp_path / "run2", **params)
    assert _failed(second) == [], second.status(errors=True)
    assert log.read_text().split() == ["task-000001"], "only the failed task reruns"
    assert gd.load_manifest(str(target)) == manifest, (
        "same manifest, not re-initialised"
    )
    assert gd.check_pyramid_store(str(target)).complete

    third = wf.run_local(run_dir=tmp_path / "run3", overwrite=True, **params)
    assert _failed(third) == [], third.status(errors=True)
    assert gd.load_manifest(str(target)).manifest_id != manifest.manifest_id
