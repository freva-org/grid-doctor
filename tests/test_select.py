"""Tests for `grid_doctor.select` and `save_pyramid(write_coords=...)`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from grid_doctor import (
    save_pyramid,
    select_bbox,
    select_cells,
    select_cone,
)
from grid_doctor.helpers import WRITE_COORDS_MAX_LEVEL
from grid_doctor.select import _contiguous_runs, _parents_to_ranges
from grid_doctor.types import HEALPIX_INDEX

LEVEL = 8
DELTA = 4
NPIX = 12 * 4**LEVEL
BBOX_LON = (13.1, 13.8)
BBOX_LAT = (52.3, 52.7)


def _centres(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from healpix_geo import nested

    lon, lat = nested.healpix_to_lonlat(
        np.asarray(cells, dtype=np.int64), LEVEL, ellipsoid="sphere"
    )
    return np.asarray(lon), np.asarray(lat)


@pytest.fixture
def dense_ds() -> xr.Dataset:
    """Dense global dataset whose values encode the healpix index."""
    data = (np.arange(NPIX) % 97).astype(np.float32)
    return xr.Dataset(
        {"t2m": (HEALPIX_INDEX, data)},
        attrs={
            "healpix_level": LEVEL,
            "healpix_nside": 2**LEVEL,
            "healpix_order": "nested",
        },
    )


class TestRangeMachinery:
    def test_contiguous_runs(self) -> None:
        ids = np.array([5, 1, 2, 3, 9, 10], dtype=np.int64)
        assert _contiguous_runs(ids) == [(1, 4), (5, 6), (9, 11)]

    def test_single_run(self) -> None:
        assert _contiguous_runs(np.arange(4, dtype=np.int64)) == [(0, 4)]

    def test_empty(self) -> None:
        assert _contiguous_runs(np.array([], dtype=np.int64)) == []

    def test_parents_to_ranges(self) -> None:
        parents = np.array([2, 3, 7], dtype=np.int64)
        assert _parents_to_ranges(parents, 2) == [(32, 64), (112, 128)]


class TestSelectCells:
    def test_positional_dense_selection(self, dense_ds: xr.Dataset) -> None:
        cells = np.array([0, 1, 2, 100, 5000], dtype=np.int64)
        result = select_cells(dense_ds, cells)
        assert (result[HEALPIX_INDEX].values == cells).all()
        assert np.allclose(result["t2m"].values, cells % 97)

    def test_coords_are_reconstructed(self, dense_ds: xr.Dataset) -> None:
        cells = np.array([42, 43], dtype=np.int64)
        result = select_cells(dense_ds, cells)
        lon, lat = _centres(cells)
        assert np.allclose(result["longitude"].values, lon)
        assert np.allclose(result["latitude"].values, lat)
        assert "crs" in result.coords
        assert result["t2m"].attrs["grid_mapping"] == "crs"
        assert int(result.attrs["grid_doctor_sparse"]) == 1
        assert result.attrs["healpix_level"] == LEVEL

    def test_selection_from_compact_subset(self, dense_ds: xr.Dataset) -> None:
        first = select_cells(dense_ds, np.arange(100, 200))
        second = select_cells(first, np.array([150, 160]))
        assert np.allclose(second["t2m"].values, np.array([150, 160]) % 97)

    def test_missing_cell_in_subset_raises(self, dense_ds: xr.Dataset) -> None:
        subset = select_cells(dense_ds, np.arange(100, 200))
        with pytest.raises(KeyError, match="not present"):
            select_cells(subset, np.array([5]))

    def test_out_of_range_raises(self, dense_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="within"):
            select_cells(dense_ds, np.array([NPIX]))

    def test_empty_selection_raises(self, dense_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="No cells"):
            select_cells(dense_ds, np.array([], dtype=np.int64))

    def test_missing_level_attribute_raises(self) -> None:
        ds = xr.Dataset({"t2m": (HEALPIX_INDEX, np.zeros(12))})
        with pytest.raises(ValueError, match="healpix_level"):
            select_cells(ds, np.array([0]))
        result = select_cells(ds, np.array([0]), level=0)
        assert result.sizes[HEALPIX_INDEX] == 1

    def test_ring_ordering_rejected(self, dense_ds: xr.Dataset) -> None:
        ds = dense_ds.copy()
        ds.attrs["healpix_order"] = "ring"
        with pytest.raises(ValueError, match="nested"):
            select_cells(ds, np.array([0]))

    def test_batch_dimensions_preserved(self, dense_ds: xr.Dataset) -> None:
        ds = dense_ds.copy()
        ds["t2m"] = ds["t2m"].expand_dims(time=[0, 1])
        result = select_cells(ds, np.array([10, 11]))
        assert result["t2m"].dims == ("time", HEALPIX_INDEX)
        assert result.sizes == {"time": 2, HEALPIX_INDEX: 2}


class TestSelectBbox:
    def test_covers_the_box(self, dense_ds: xr.Dataset) -> None:
        result = select_bbox(
            dense_ds, lon=BBOX_LON, lat=BBOX_LAT, query_delta=DELTA
        )
        # Every cell whose centre lies inside the box must be included.
        lon, lat = _centres(np.arange(NPIX))
        inside = (
            (lon > BBOX_LON[0])
            & (lon < BBOX_LON[1])
            & (lat > BBOX_LAT[0])
            & (lat < BBOX_LAT[1])
        )
        assert np.isin(np.nonzero(inside)[0], result[HEALPIX_INDEX].values).all()
        assert np.allclose(result["t2m"].values, result[HEALPIX_INDEX].values % 97)

    def test_selection_is_local(self, dense_ds: xr.Dataset) -> None:
        """The covering superset stays within one parent cell of the box."""
        result = select_bbox(
            dense_ds, lon=BBOX_LON, lat=BBOX_LAT, query_delta=DELTA
        )
        parent_spacing = 58.6 / 2 ** (LEVEL - DELTA)
        lat = result["latitude"].values
        assert lat.min() > BBOX_LAT[0] - 2 * parent_spacing
        assert lat.max() < BBOX_LAT[1] + 2 * parent_spacing
        assert result.sizes[HEALPIX_INDEX] < NPIX // 100

    def test_query_delta_alignment(self, dense_ds: xr.Dataset) -> None:
        """Selected cells arrive in whole parent blocks of 4**delta."""
        result = select_bbox(
            dense_ds, lon=BBOX_LON, lat=BBOX_LAT, query_delta=DELTA
        )
        assert result.sizes[HEALPIX_INDEX] % 4**DELTA == 0
        parents = np.unique(result[HEALPIX_INDEX].values >> (2 * DELTA))
        assert result.sizes[HEALPIX_INDEX] == parents.size * 4**DELTA


class TestSelectCone:
    def test_contains_centre_cell(self, dense_ds: xr.Dataset) -> None:
        from healpix_geo import nested

        centre = nested.lonlat_to_healpix(
            np.array([13.4]), np.array([52.5]), depth=LEVEL, ellipsoid="sphere"
        )[0]
        result = select_cone(
            dense_ds, lon=13.4, lat=52.5, radius=0.5, query_delta=DELTA
        )
        assert int(centre) in result[HEALPIX_INDEX].values
        assert np.allclose(result["t2m"].values, result[HEALPIX_INDEX].values % 97)


class TestWriteCoords:
    @staticmethod
    def _with_coords(ds: xr.Dataset) -> xr.Dataset:
        from grid_doctor.remap import _attach_healpix_coords

        return _attach_healpix_coords(ds, level=LEVEL, nest=True)

    def test_false_drops_coordinate_arrays(
        self, dense_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        save_pyramid(
            {LEVEL: self._with_coords(dense_ds)},
            str(tmp_path),
            write_coords=False,
        )
        opened = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr", chunks=None)
        assert "latitude" not in opened and "longitude" not in opened
        assert HEALPIX_INDEX not in opened.coords
        assert "crs" in opened.coords
        assert int(opened.attrs["grid_doctor_implicit_coords"]) == 1
        assert opened.attrs["healpix_level"] == LEVEL

    def test_true_keeps_coordinate_arrays(
        self, dense_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        save_pyramid(
            {LEVEL: self._with_coords(dense_ds)},
            str(tmp_path),
            write_coords=True,
        )
        opened = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr", chunks=None)
        assert {"latitude", "longitude", HEALPIX_INDEX} <= set(opened.coords)
        assert "grid_doctor_implicit_coords" not in opened.attrs

    def test_auto_uses_level_threshold(
        self, dense_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        assert LEVEL <= WRITE_COORDS_MAX_LEVEL
        save_pyramid(
            {LEVEL: self._with_coords(dense_ds)},
            str(tmp_path),
            write_coords="auto",
        )
        opened = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr", chunks=None)
        assert "latitude" in opened.coords

    def test_round_trip_store_to_region(
        self, dense_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Coordinate-less store -> open chunks=None -> bbox -> values."""
        save_pyramid(
            {LEVEL: dense_ds},
            str(tmp_path),
            write_coords=False,
            encoding={LEVEL: {"t2m": {"chunks": (4**DELTA,)}}},
        )
        opened = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr", chunks=None)
        region = select_bbox(
            opened, lon=BBOX_LON, lat=BBOX_LAT, query_delta=DELTA
        )
        assert np.allclose(region["t2m"].values, region[HEALPIX_INDEX].values % 97)
        assert int(region.attrs["grid_doctor_sparse"]) == 1
