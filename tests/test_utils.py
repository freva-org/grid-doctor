"""Tests for `grid_doctor.utils`."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
    """``cached_weights`` looks weights up by their description: local copy, store, else compute."""

    @staticmethod
    @pytest.fixture(autouse=True)
    def _default_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Keep the default cache in ``tmp_path``, never in the user's cache."""
        monkeypatch.setattr("grid_doctor.utils.cache_dir", lambda: tmp_path / "default-cache")

    @staticmethod
    @pytest.fixture
    def computed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        """Fake ``compute_healpix_weights`` writing to ``weights_path``; records each call."""
        calls: list[dict[str, object]] = []

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: object) -> Path:
            calls.append({"level": level, **kwargs})
            target = Path(str(kwargs["weights_path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            from grid_doctor.metadata import _keys as keys

            data = {"row": ("nnz", np.array([1])), "col": ("nnz", np.array([1])), "S": ("nnz", np.array([1.0]))}
            xr.Dataset(data, attrs={keys.LEVEL_ATTR: level, keys.ORDER_ATTR: "nested"}).to_netcdf(target)
            return target

        monkeypatch.setattr("grid_doctor.remap.compute_healpix_weights", fake_compute)
        return calls

    @staticmethod
    def _ref(ds: xr.Dataset, level: int, method: str = "conservative") -> str:
        from grid_doctor.metadata import HealpixGrid
        from grid_doctor.weights import describe_source
        from grid_doctor.weights import WeightsDescription

        return WeightsDescription(describe_source(ds), HealpixGrid(level=level), method).ref.path  # type: ignore[arg-type]

    def test_computes_into_store_layout(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        path = cached_weights(regular_ds, 2, store=tmp_path)
        assert path == tmp_path / self._ref(regular_ds, 2)
        assert path.exists()
        assert computed[0]["source_description"] is not None  # source described once

    @pytest.mark.usefixtures("computed")
    def test_cache_hit_does_not_mesh(
        self,
        regular_ds: xr.Dataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = cached_weights(regular_ds, 2, store=tmp_path)

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("mesh built on a cache hit")

        monkeypatch.setattr("grid_doctor.weights._source_mesh", boom)
        assert cached_weights(regular_ds, 2, store=tmp_path) == path

    def test_reuses_weights(self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]) -> None:
        assert cached_weights(regular_ds, 2, store=tmp_path) == cached_weights(regular_ds, 2, store=tmp_path)
        assert len(computed) == 1

    def test_same_grid_in_radians_shares_weights(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        radians = regular_ds.assign_coords(
            lat=np.radians(regular_ds["lat"]).assign_attrs(units="radian"),
            lon=np.radians(regular_ds["lon"]).assign_attrs(units="radian"),
        )
        assert cached_weights(radians, 2, store=tmp_path) == cached_weights(regular_ds, 2, store=tmp_path)
        assert len(computed) == 1

    def test_growing_along_time_reuses_weights(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        longer = xr.concat([regular_ds, regular_ds.assign_coords(time=regular_ds.time + 100)], "time")
        paths = {
            cached_weights(ds, 2, store=tmp_path)
            for ds in (regular_ds.isel(time=slice(0, 1)), regular_ds, longer)
        }
        assert len(paths) == 1
        assert len(computed) == 1

    def test_nest_is_deprecated(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        """Weight files are nested; ring order is chosen when applying them."""
        with pytest.warns(DeprecationWarning, match="nest=False"):
            ring = cached_weights(regular_ds, 2, nest=False, store=tmp_path)
        assert ring == cached_weights(regular_ds, 2, store=tmp_path) == tmp_path / self._ref(regular_ds, 2)
        assert computed[0]["nest"] is True
        assert len(computed) == 1

    def test_ignore_unmapped_false_checks_cache_hits(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        """Weights stored while ignoring unmapped cells must not pass a full-coverage request."""
        from grid_doctor.metadata import _keys as keys

        path = cached_weights(regular_ds, 2, store=tmp_path, ignore_unmapped=True)
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        ds.assign_attrs({keys.UNMAPPED_CELLS_ATTR: 5}).to_netcdf(path)  # e.g. a regional grid

        with pytest.raises(ValueError, match="5 HEALPix target cells"):
            cached_weights(regular_ds, 2, store=tmp_path, ignore_unmapped=False)
        assert cached_weights(regular_ds, 2, store=tmp_path) == path  # default: unmapped cells accepted
        assert len(computed) == 1

    def test_level_derived_before_lookup(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        derived = cached_weights(regular_ds, store=tmp_path)
        level = computed[0]["level"]
        assert isinstance(level, int)
        assert cached_weights(regular_ds, level, store=tmp_path) == derived
        assert len(computed) == 1

    @pytest.mark.usefixtures("computed")
    def test_method_and_level_get_own_files(
        self, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        paths = {
            cached_weights(regular_ds, 1, method="nearest", store=tmp_path),
            cached_weights(regular_ds, 2, method="nearest", store=tmp_path),
            cached_weights(regular_ds, 2, method="conservative", store=tmp_path),
        }
        assert len(paths) == 3

    def test_publishes_to_remote_store(
        self, regular_ds: xr.Dataset, tmp_path: Path, computed: list[dict[str, object]]
    ) -> None:
        import fsspec

        from grid_doctor.weights import WeightsStore

        fsspec.filesystem("memory").store.clear()
        store = WeightsStore("memory://weights", cache_dir=tmp_path / "cache")
        path = cached_weights(regular_ds, 2, store=store)
        assert path == tmp_path / "cache" / self._ref(regular_ds, 2)
        assert store.exists(self._ref(regular_ds, 2))

        # another user with an empty local cache downloads instead of computing
        other = WeightsStore("memory://weights", cache_dir=tmp_path / "other")
        assert cached_weights(regular_ds, 2, store=other).exists()
        assert len(computed) == 1
        fsspec.filesystem("memory").store.clear()

    @pytest.mark.usefixtures("computed")
    def test_read_only_store_keeps_local_copy(
        self, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        import fsspec

        from grid_doctor.weights import WeightsStore

        fsspec.filesystem("memory").store.clear()
        store = WeightsStore("memory://weights", cache_dir=tmp_path, read_only=True)
        assert cached_weights(regular_ds, 2, store=store).exists()
        assert not store.exists(self._ref(regular_ds, 2))

    @pytest.mark.usefixtures("computed")
    def test_cache_path_is_deprecated(
        self, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        with pytest.warns(DeprecationWarning, match="cache_path"):
            path = cached_weights(regular_ds, 2, cache_path=tmp_path)
        assert path == tmp_path / self._ref(regular_ds, 2)


