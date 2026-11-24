"""Dataset preparation for writing (``grid_doctor.io._xarray``)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from grid_doctor.metadata import (
    HealpixGrid,
    HealpixMetadata,
)

LEVEL = 2


def _dense(level: int = LEVEL, *, coords: bool = True) -> xr.Dataset:
    npix = 12 * 4**level
    ds = xr.Dataset(
        {
            "t": (("time", "cell"), np.zeros((2, npix))),
            "static": ("time", np.arange(2.0)),
        },
        coords={"time": [0, 1]},
        attrs={"title": "test"},
    )
    if coords:
        ds = ds.assign_coords(
            cell=np.arange(npix, dtype=np.int64),
            latitude=("cell", np.zeros(npix)),
            longitude=("cell", np.zeros(npix)),
        )
    return ds


def _sparse(cells: list[int]) -> xr.Dataset:
    return xr.Dataset(
        {"t": ("cell", np.zeros(len(cells)))},
        coords={"cell": np.asarray(cells, dtype=np.int64)},
    )


class TestMaybeDropCoordinates:
    def test_dense_store_drops_coords_and_is_marked(self) -> None:
        from grid_doctor.io._xarray import maybe_drop_coordinates

        ds = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        out = maybe_drop_coordinates(ds, LEVEL, write_coords=False)
        assert not {"cell", "latitude", "longitude"} & set(out.variables)
        assert HealpixMetadata.from_dataset(out).implicit_coords

    def test_sparse_subset_keeps_cell_ids(self) -> None:
        from grid_doctor.io._xarray import maybe_drop_coordinates

        ds = HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True).apply(_sparse([5, 9]))
        out = maybe_drop_coordinates(ds, LEVEL, write_coords=False)
        np.testing.assert_array_equal(out["cell"].values, [5, 9])
        assert not HealpixMetadata.from_dataset(out).implicit_coords
