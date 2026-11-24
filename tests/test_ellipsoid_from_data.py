"""Code reading existing data or weights uses their ellipsoid, not the config.

The config only decides the ellipsoid of *new* grids.  Every test in
``TestRecordedEllipsoid`` runs for both ellipsoids with the config set to
the other one, so a reader that fell back to the config would fail.  The
sphere case doubles as the case of data written before the ellipsoid was
recorded: spherical metadata records none.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import grid_doctor as gd
from grid_doctor import select
from grid_doctor._geometry.healpix import (
    _attach_healpix_coords,
    _healpix_centres,
)
from grid_doctor.metadata import ELLIPSOID_ATTR, HealpixGrid, HealpixMetadata
from grid_doctor.pyramid._levels import assemble_coarse_level
from grid_doctor.types import Ellipsoid

LEVEL = 2
NPIX = 12 * 4**LEVEL

_POINTS = xr.Dataset(
    {"t": ("obs", np.arange(4.0))},
    coords={"lat": ("obs", [10.0, 40.0, -30.0, 60.0]), "lon": ("obs", [0.0, 90.0, 180.0, 270.0])},
)


def _latitudes(ellipsoid: Ellipsoid, level: int = LEVEL) -> np.ndarray:
    """Cell-centre latitudes of every cell at *level* on *ellipsoid*."""
    return _healpix_centres(HealpixGrid(level=level, ellipsoid=ellipsoid))[0]


def _recorded(ds: xr.Dataset) -> Ellipsoid:
    return HealpixMetadata.from_dataset(ds).grid.ellipsoid


def _dense(ellipsoid: Ellipsoid) -> xr.Dataset:
    """A dense HEALPix dataset at ``LEVEL`` built on *ellipsoid*."""
    ds = xr.Dataset({"t": ("cell", np.arange(float(NPIX)))})
    return _attach_healpix_coords(ds, HealpixGrid(level=LEVEL, ellipsoid=ellipsoid), method="conservative")


def _weight_file(path: Path, source: xr.Dataset, ellipsoid: Ellipsoid | None) -> Path:
    """Toy nearest weights from *source* to ``LEVEL``; ``None`` records no ellipsoid."""
    n_source = source.sizes["y"] * source.sizes["x"]
    attrs: dict[str, object] = {
        "grid_doctor_level": LEVEL,
        "grid_doctor_order": "nested",
        "grid_doctor_method": "nearest",
        "grid_doctor_source_dims": json.dumps(["y", "x"]),
        "grid_doctor_source_size": n_source,
    }
    if ellipsoid is not None:
        attrs[ELLIPSOID_ATTR] = ellipsoid
    xr.Dataset(
        {
            "row": ("nnz", np.arange(1, NPIX + 1, dtype=np.int32)),
            "col": ("nnz", np.arange(NPIX, dtype=np.int32) % n_source + 1),
            "S": ("nnz", np.ones(NPIX)),
        },
        attrs=attrs,
    ).to_netcdf(path)
    return path


class TestRecordedEllipsoid:
    @pytest.fixture(params=["sphere", "WGS84"])
    def ellipsoid(self, request: pytest.FixtureRequest) -> Iterator[Ellipsoid]:
        """The ellipsoid the data is built on; the config is set to the other one."""
        ellipsoid: Ellipsoid = request.param
        with gd.config.set(ellipsoid="WGS84" if ellipsoid == "sphere" else "sphere"):
            yield ellipsoid

    def test_apply_weight_file_uses_the_weight_file(
        self, ellipsoid: Ellipsoid, tmp_path: Path, regular_ds: xr.Dataset
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, ellipsoid)
        result = gd.apply_weight_file(regular_ds, weights)
        assert _recorded(result) == ellipsoid
        np.testing.assert_array_equal(result["latitude"].values, _latitudes(ellipsoid))

    def test_apply_weight_file_rejects_another_ellipsoid(
        self, ellipsoid: Ellipsoid, tmp_path: Path, regular_ds: xr.Dataset
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, ellipsoid)
        other: Ellipsoid = "WGS84" if ellipsoid == "sphere" else "sphere"
        with pytest.raises(ValueError, match=f"was computed on '{ellipsoid}', not '{other}'"):
            gd.apply_weight_file(regular_ds, weights, ellipsoid=other)

    def test_regrid_to_healpix_keeps_an_existing_weight_file(
        self, ellipsoid: Ellipsoid, tmp_path: Path, regular_ds: xr.Dataset
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, ellipsoid)
        result = gd.regrid_to_healpix(regular_ds, LEVEL, method="nearest", weights_path=weights)
        assert _recorded(result) == ellipsoid

    def test_coarsening_keeps_the_ellipsoid(self, ellipsoid: Ellipsoid) -> None:
        coarse = gd.coarsen_healpix(_dense(ellipsoid), LEVEL - 1)
        assert _recorded(coarse) == ellipsoid
        np.testing.assert_array_equal(coarse["latitude"].values, _latitudes(ellipsoid, LEVEL - 1))

    def test_sparse_to_dense_keeps_the_ellipsoid(self, ellipsoid: Ellipsoid) -> None:
        sparse = gd.bin_to_healpix(_POINTS, LEVEL, dense=False, ellipsoid=ellipsoid)
        dense = gd.sparse_to_dense(sparse)
        assert _recorded(dense) == ellipsoid
        np.testing.assert_array_equal(dense["latitude"].values, _latitudes(ellipsoid))

    def test_select_cells_uses_the_dataset(self, ellipsoid: Ellipsoid) -> None:
        cells = [0, 5, 100]
        subset = gd.select_cells(_dense(ellipsoid), cells)
        assert _recorded(subset) == ellipsoid
        np.testing.assert_array_equal(subset["latitude"].values, _latitudes(ellipsoid)[cells])

    def test_attach_cell_coords_uses_the_dataset(self, ellipsoid: Ellipsoid) -> None:
        cells = np.array([3, 4], dtype=np.int64)
        subset = _dense(ellipsoid).isel(cell=cells).drop_vars(["cell", "latitude", "longitude"])
        result = gd.attach_cell_coords(subset, cells, level=LEVEL)
        assert _recorded(result) == ellipsoid
        np.testing.assert_array_equal(result["latitude"].values, _latitudes(ellipsoid)[cells])

    @pytest.mark.parametrize(
        "query",
        [
            pytest.param(lambda ds: gd.select_bbox(ds, lon=(0.0, 40.0), lat=(10.0, 50.0)), id="bbox"),
            pytest.param(lambda ds: gd.select_cone(ds, lon=20.0, lat=30.0, radius=10.0), id="cone"),
        ],
    )
    def test_region_queries_use_the_dataset(
        self,
        ellipsoid: Ellipsoid,
        query: Callable[[xr.Dataset], xr.Dataset],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        queried: list[Ellipsoid] = []
        real = select._require_healpix_geo_module

        def spy(grid: HealpixGrid) -> object:
            queried.append(grid.ellipsoid)
            return real(grid)

        monkeypatch.setattr(select, "_require_healpix_geo_module", spy)
        subset = query(_dense(ellipsoid))
        assert queried and set(queried) == {ellipsoid}
        assert _recorded(subset) == ellipsoid


class TestUnrecordedEllipsoid:
    """Weight files and datasets that record no ellipsoid.

    Weight files and HEALPix metadata without figure-of-the-earth
    attributes are spherical.  A dataset with no HEALPix metadata at all
    gets the configured ellipsoid when selecting, and is refused by
    coarsening.
    """

    def test_weight_file_without_ellipsoid_is_a_sphere(self, tmp_path: Path, regular_ds: xr.Dataset) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, None)
        with gd.config.set(ellipsoid="WGS84"):
            result = gd.apply_weight_file(regular_ds, weights)
        assert _recorded(result) == "sphere"
        np.testing.assert_array_equal(result["latitude"].values, _latitudes("sphere"))

    def test_weight_file_without_ellipsoid_warns(
        self, tmp_path: Path, regular_ds: xr.Dataset, caplog: pytest.LogCaptureFixture
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, None)
        with caplog.at_level(logging.WARNING, logger="grid_doctor"):
            gd.apply_weight_file(regular_ds, weights)
        assert f"Weight file {weights} records no ellipsoid" in caplog.text
        assert "assuming 'sphere'" in caplog.text

    def test_weight_file_with_ellipsoid_does_not_warn(
        self, tmp_path: Path, regular_ds: xr.Dataset, caplog: pytest.LogCaptureFixture
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, "sphere")
        with caplog.at_level(logging.WARNING, logger="grid_doctor"):
            gd.apply_weight_file(regular_ds, weights)
        assert "records no ellipsoid" not in caplog.text

    def test_weight_file_with_unsupported_ellipsoid_is_rejected(
        self, tmp_path: Path, regular_ds: xr.Dataset
    ) -> None:
        weights = _weight_file(tmp_path / "weights.nc", regular_ds, "GRS80")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unsupported ellipsoid 'GRS80'"):
            gd.apply_weight_file(regular_ds, weights)

    def test_coarsening_requires_healpix_metadata(self) -> None:
        ds = xr.Dataset({"t": ("cell", np.arange(float(NPIX)))})
        with pytest.raises(ValueError, match="without HEALPix metadata.*ellipsoid is unknown"):
            assemble_coarse_level(ds, {}, LEVEL - 1, LEVEL, coarsen_mode="mean", min_valid_fraction=0.5)

    @pytest.mark.parametrize("configured", ["sphere", "WGS84"])
    def test_dataset_without_metadata_uses_the_config(self, configured: Ellipsoid) -> None:
        ds = xr.Dataset({"t": ("cell", np.zeros(2))})
        cells = np.array([3, 4], dtype=np.int64)
        with gd.config.set(ellipsoid=configured):
            result = gd.attach_cell_coords(ds, cells, level=LEVEL)
        assert _recorded(result) == configured
        np.testing.assert_array_equal(result["latitude"].values, _latitudes(configured)[cells])
