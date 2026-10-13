"""Tests for `grid_doctor.utils`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.utils import (
    cache_dir,
    cached_open_dataset,
    cached_weights,
    get_s3_options,
    chunk_for_target_store_size,
)
from grid_doctor.utils import _key_hash


class TestChunkSizeEstimator:
    def test_map_default_uses_full_cell_dimension(self) -> None:
        result = chunk_for_target_store_size(level=8)

        nside = 2**8
        ncell = 12 * nside * nside
        itemsize = np.dtype("float32").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_time = max(1, target_uncompressed_bytes // (itemsize * ncell))

        assert result == {"time": int(expected_time), "cell": int(ncell)}

    def test_map_respects_max_cell_chunk(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            max_cell_chunk=100_000,
        )

        itemsize = np.dtype("float32").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_time = max(1, target_uncompressed_bytes // (itemsize * 100_000))

        assert result == {"time": int(expected_time), "cell": 100_000}

    def test_map_uses_given_dtype(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            dtype="float64",
            max_cell_chunk=50_000,
        )

        itemsize = np.dtype("float64").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_time = max(1, target_uncompressed_bytes // (itemsize * 50_000))

        assert result == {"time": int(expected_time), "cell": 50_000}

    def test_time_series_uses_ntime_when_no_max_time_chunk_given(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            access="time_series",
            ntime=365,
        )

        nside = 2**8
        ncell = 12 * nside * nside
        itemsize = np.dtype("float32").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_cell = max(1, target_uncompressed_bytes // (itemsize * 365))
        expected_cell = min(expected_cell, ncell)

        assert result == {"time": 365, "cell": int(expected_cell)}

    def test_time_series_prefers_max_time_chunk_over_ntime(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            access="time_series",
            ntime=365,
            max_time_chunk=30,
        )

        nside = 2**8
        ncell = 12 * nside * nside
        itemsize = np.dtype("float32").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_cell = max(1, target_uncompressed_bytes // (itemsize * 30))
        expected_cell = min(expected_cell, ncell)

        assert result == {"time": 30, "cell": int(expected_cell)}

    def test_time_series_caps_max_time_chunk_by_ntime(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            access="time_series",
            ntime=10,
            max_time_chunk=30,
        )

        nside = 2**8
        ncell = 12 * nside * nside
        itemsize = np.dtype("float32").itemsize
        target_stored_bytes = int(16.0 * 1024 * 1024)
        target_uncompressed_bytes = int(target_stored_bytes * 2.0)
        expected_cell = max(1, target_uncompressed_bytes // (itemsize * 10))
        expected_cell = min(expected_cell, ncell)

        assert result == {"time": 10, "cell": int(expected_cell)}

    def test_time_series_respects_max_cell_chunk(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            access="time_series",
            ntime=365,
            max_cell_chunk=20_000,
        )

        assert result["time"] == 365
        assert result["cell"] == 20_000

    def test_time_series_requires_ntime_or_max_time_chunk(self) -> None:
        with pytest.raises(
            ValueError,
            match="provide either ntime or max_time_chunk",
        ):
            chunk_for_target_store_size(level=8, access="time_series")

    def test_invalid_access_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported access mode"):
            chunk_for_target_store_size(level=8, access="foobar")  # type: ignore[arg-type]

    def test_map_time_chunk_is_at_least_one(self) -> None:
        result = chunk_for_target_store_size(
            level=15,
            target_stored_mib=0.001,
            compression_ratio=1.0,
        )

        assert result["time"] == 1
        assert result["cell"] == 12 * (2**15) * (2**15)

    def test_time_series_cell_chunk_is_at_least_one(self) -> None:
        result = chunk_for_target_store_size(
            level=8,
            access="time_series",
            ntime=10_000_000,
            target_stored_mib=0.001,
            compression_ratio=1.0,
        )

        assert result["time"] == 10_000_000
        assert result["cell"] == 1


class TestGetS3Options:
    def test_reads_secrets(self, tmp_path: Path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"accessKey": "AK", "secretKey": "SK"}))
        options = get_s3_options("https://s3.example.com", creds)
        assert options["endpoint_url"] == "https://s3.example.com"
        assert options["key"] == "AK"
        assert options["secret"] == "SK"

    def test_merges_extra_kwargs(self, tmp_path: Path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"accessKey": "AK", "secretKey": "SK"}))
        options = get_s3_options("https://s3.example.com", creds, region="eu")
        assert options["region"] == "eu"


class TestCacheDir:
    def test_returns_writable_path(self) -> None:
        with mock.patch.dict(os.environ, {"USER": "tester"}):
            result = cache_dir()
            assert isinstance(result, Path)
            assert result.is_dir()


class TestCachedOpenDataset:
    def test_opens_and_caches(self, tmp_path: Path) -> None:
        source = tmp_path / "input.nc"
        xr.Dataset({"t": (("x",), np.arange(5.0))}).to_netcdf(source)
        with mock.patch("grid_doctor.utils.cache_dir", return_value=tmp_path):
            result = cached_open_dataset([str(source)])
        assert "t" in result.data_vars
        assert list(tmp_path.glob("*.pickle"))

    def test_loads_from_cache(self, tmp_path: Path) -> None:
        source = tmp_path / "input.nc"
        xr.Dataset({"t": (("x",), np.arange(5.0))}).to_netcdf(source)
        with mock.patch("grid_doctor.utils.cache_dir", return_value=tmp_path):
            result1 = cached_open_dataset([str(source)])
            result2 = cached_open_dataset([str(source)])
        xr.testing.assert_equal(result1, result2)


class TestCachedWeights:
    def test_creates_cached_weight_file(
        self, regular_ds: xr.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_compute(ds: xr.Dataset, level: int, **kwargs: object) -> Path:
            del ds, level, kwargs
            target = tmp_path / "weights_fake.nc"
            xr.Dataset(
                {
                    "row": ("nnz", np.array([1])),
                    "col": ("nnz", np.array([1])),
                    "S": ("nnz", np.array([1.0])),
                }
            ).to_netcdf(target)
            return target

        monkeypatch.setattr("grid_doctor.remap.compute_healpix_weights", fake_compute)
        path = cached_weights(regular_ds, 2, cache_path=tmp_path)
        assert path.exists()
        assert path.suffix == ".nc"

    def test_reuses_cached_weight_file(
        self, regular_ds: xr.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"count": 0}

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: object) -> Path:
            del ds, level, kwargs
            calls["count"] += 1
            target = tmp_path / "cache.nc"
            xr.Dataset(
                {
                    "row": ("nnz", np.array([1])),
                    "col": ("nnz", np.array([1])),
                    "S": ("nnz", np.array([1.0])),
                }
            ).to_netcdf(target)
            return target

        monkeypatch.setattr("grid_doctor.remap.compute_healpix_weights", fake_compute)
        path1 = cached_weights(regular_ds, 2, cache_path=tmp_path / "cache.nc")
        path2 = cached_weights(regular_ds, 2, cache_path=tmp_path / "cache.nc")
        assert path1 == path2
        assert calls["count"] == 1

    def test_hash_depends_on_method_and_level(
        self, regular_ds: xr.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[Path] = []

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: object) -> Path:
            del ds, kwargs
            target = tmp_path / f"weights_{level}_{len(created)}.nc"
            xr.Dataset(
                {
                    "row": ("nnz", np.array([1])),
                    "col": ("nnz", np.array([1])),
                    "S": ("nnz", np.array([1.0])),
                }
            ).to_netcdf(target)
            created.append(target)
            return target

        monkeypatch.setattr("grid_doctor.remap.compute_healpix_weights", fake_compute)
        path1 = cached_weights(regular_ds, 1, method="nearest", cache_path=tmp_path)
        path2 = cached_weights(regular_ds, 2, method="nearest", cache_path=tmp_path)
        assert path1 != path2


def _wrf_ds(offset: float = 0.0, lat: str = "XLAT", lon: str = "XLONG") -> xr.Dataset:
    xlong, xlat = np.meshgrid(np.linspace(0.0, 10.0, 4) + offset, np.linspace(40.0, 50.0, 3))
    return xr.Dataset(coords={lat: (("south_north", "west_east"), xlat), lon: (("south_north", "west_east"), xlong)})


class TestKeyHash:
    def test_different_wrf_grids_differ(self) -> None:
        assert _key_hash(_wrf_ds(), 5) != _key_hash(_wrf_ds(offset=1.0), 5)

    def test_same_grid_different_names_or_dtype_match(self) -> None:
        ref = _key_hash(_wrf_ds(), 5)
        assert _key_hash(_wrf_ds(lat="lat", lon="lon"), 5) == ref

    def test_float32_and_float64_match(self) -> None:
        f64 = _wrf_ds(offset=0.1)
        f32 = f64.assign_coords({k: f64[k].astype(np.float32) for k in ("XLAT", "XLONG")})
        f64 = f32.assign_coords({k: f32[k].astype(np.float64) for k in ("XLAT", "XLONG")})
        assert _key_hash(f32, 5) == _key_hash(f64, 5)

    def test_lat_lon_order_matters(self) -> None:
        ds = _wrf_ds()
        swapped = ds.rename({"XLAT": "tmp"}).rename({"XLONG": "XLAT", "tmp": "XLONG"})
        assert _key_hash(ds, 5) != _key_hash(swapped, 5)

    def test_shape_is_hashed(self) -> None:
        ds = _wrf_ds()
        flat = xr.Dataset(coords={k: ("cell", ds[k].values.ravel()) for k in ("XLAT", "XLONG")})
        assert _key_hash(ds, 5) != _key_hash(flat, 5)

    def test_vertices_are_hashed(self) -> None:
        clat, clon = np.array([0.0, 10.0]), np.array([0.0, 20.0])
        base = {"clat": ("cell", clat), "clon": ("cell", clon)}
        verts = lambda d: {
            "clat_vertices": (("cell", "nv"), clat[:, None] + [-d, 0.0, d]),
            "clon_vertices": (("cell", "nv"), clon[:, None] + [-d, 0.0, d]),
        }
        hashes = {
            _key_hash(xr.Dataset(coords=base), 5),
            _key_hash(xr.Dataset(coords=base | verts(1.0)), 5),
            _key_hash(xr.Dataset(coords=base | verts(2.0)), 5),
        }
        assert len(hashes) == 3

    def test_raises_without_coordinates(self) -> None:
        with pytest.raises(ValueError, match="Could not locate latitude/longitude"):
            _key_hash(xr.Dataset(coords={"x": [0, 1]}), 5)


class TestCanonicalWeightsKey:
    LON = np.arange(8) * 45.0  # 0, 45, ..., 315

    @staticmethod
    def _grid(lon: np.ndarray = LON, radians: bool = False) -> xr.Dataset:
        lat = np.linspace(-80.0, 80.0, 5)
        if radians:
            lat, lon = np.radians(lat), np.radians(lon)
        units = "rad" if radians else "degrees"
        return xr.Dataset(coords={"lat": ("lat", lat, {"units": units}), "lon": ("lon", lon, {"units": units})})

    @staticmethod
    def _key(ds: xr.Dataset) -> str:
        from grid_doctor.remap_backend import TargetDescription, WeightsDescription, describe_source

        return WeightsDescription(
            source=describe_source(ds), target=TargetDescription(level=2, order="nest"), method="conservative"
        ).key

    def test_same_grid_any_convention_shares_key(self) -> None:
        ref = self._key(self._grid())
        assert self._key(self._grid(radians=True)) == ref  # radians vs degrees
        assert self._key(self._grid(self.LON + 360.0)) == ref  # lon shifted by a full turn

    def test_0_360_and_minus180_180_share_key(self) -> None:
        """The same cells in the same order share weights, whatever the lon convention."""
        lon_180 = (self.LON + 180.0) % 360.0 - 180.0  # 0, ..., 135, -180, ..., -45
        assert self._key(self._grid(lon_180)) == self._key(self._grid())

    def test_reordered_cells_get_own_key(self) -> None:
        """Weight columns follow the data's cell order, so a sorted grid must not reuse weights."""
        lon_sorted = np.sort((self.LON + 180.0) % 360.0 - 180.0)  # -180, ..., 135
        assert self._key(self._grid(lon_sorted)) != self._key(self._grid())

    def test_description_is_lazy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import grid_doctor.remap_backend as backend

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("mesh built eagerly")

        monkeypatch.setattr(backend, "_source_mesh", boom)
        monkeypatch.setattr(backend, "_target_healpix_mesh", boom)
        self._key(self._grid())

    def test_cached_weights_uses_description_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Path] = []

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
            calls.append(kwargs["weights_path"])
            xr.Dataset({"S": ("nnz", np.array([1.0]))}).to_netcdf(kwargs["weights_path"])
            return kwargs["weights_path"]

        monkeypatch.setattr("grid_doctor.remap.compute_healpix_weights", fake_compute)
        path = cached_weights(self._grid(), 2, cache_path=tmp_path)
        assert path.name == f"weights_{self._key(self._grid())[:16]}.nc"
        # same grid in radians hits the cache
        assert cached_weights(self._grid(radians=True), 2, cache_path=tmp_path) == path
        assert len(calls) == 1
