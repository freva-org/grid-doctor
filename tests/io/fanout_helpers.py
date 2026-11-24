"""Builders shared by the fan-out tests (``test_fanout*.py``)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr

import grid_doctor as gd
from grid_doctor.io import WriteTask
from grid_doctor.pyramid._build import create_healpix_pyramid

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
