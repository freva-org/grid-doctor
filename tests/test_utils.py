"""Tests for grid_doctor.utils."""

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
)


class TestGetS3Options:
    def test_reads_secrets(self, tmp_path: Path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"accessKey": "AK", "secretKey": "SK"}))

        opts = get_s3_options("https://s3.example.com", creds)
        assert opts["endpoint_url"] == "https://s3.example.com"
        assert opts["key"] == "AK"
        assert opts["secret"] == "SK"

    def test_extra_kwargs(self, tmp_path: Path) -> None:
        creds = tmp_path / "creds.json"
        creds.write_text(json.dumps({"accessKey": "AK", "secretKey": "SK"}))

        opts = get_s3_options("https://s3.example.com", creds, region="eu")
        assert opts["region"] == "eu"

    def test_raises_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_s3_options("https://s3.example.com", "/no/such/file.json")


class TestCacheDir:
    def test_falls_back_to_tmpdir(self) -> None:
        """When /scratch doesn't exist, returns tempdir."""
        with mock.patch.dict(os.environ, {"USER": "testuser"}):
            result = cache_dir()
            assert result.is_dir()

    def test_returns_path(self) -> None:
        with mock.patch.dict(os.environ, {"USER": "testuser"}):
            result = cache_dir()
            assert isinstance(result, Path)


class TestCachedOpenDataset:
    def test_opens_and_caches(self, tmp_path: Path) -> None:
        nc_file = tmp_path / "test.nc"
        ds = xr.Dataset({"t": (("x",), np.arange(5.0))})
        ds.to_netcdf(nc_file)

        with mock.patch.dict(os.environ, {"USER": "testuser"}):
            with mock.patch("grid_doctor.utils.cache_dir", return_value=tmp_path):
                result = cached_open_dataset([str(nc_file)])
                assert "t" in result.data_vars

    def test_loads_from_cache(self, tmp_path: Path) -> None:
        nc_file = tmp_path / "test.nc"
        ds = xr.Dataset({"t": (("x",), np.arange(5.0))})
        ds.to_netcdf(nc_file)

        with mock.patch("grid_doctor.utils.cache_dir", return_value=tmp_path):
            # First call: opens and caches
            result1 = cached_open_dataset([str(nc_file)])
            # Second call: loads from pickle
            result2 = cached_open_dataset([str(nc_file)])
            xr.testing.assert_equal(result1, result2)


class TestCachedWeights:
    def test_computes_and_caches(
        self, unstructured_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        weights = cached_weights(
            unstructured_ds, level=2, cache_path=tmp_path
        )
        assert isinstance(weights, xr.Dataset)
        assert "tgt_idx" in weights.dims

        # Verify a .nc file was written
        nc_files = list(tmp_path.glob("weights_*.nc"))
        assert len(nc_files) == 1

    def test_loads_from_cache(
        self, unstructured_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        w1 = cached_weights(unstructured_ds, level=2, cache_path=tmp_path)
        w2 = cached_weights(unstructured_ds, level=2, cache_path=tmp_path)
        xr.testing.assert_equal(w1, w2)

    def test_explicit_file_path(
        self, unstructured_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "my_weights.nc"
        weights = cached_weights(
            unstructured_ds, level=2, cache_path=fpath
        )
        assert fpath.exists()
        assert isinstance(weights, xr.Dataset)

    def test_different_level_different_cache(
        self, unstructured_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        cached_weights(unstructured_ds, level=1, cache_path=tmp_path)
        cached_weights(unstructured_ds, level=2, cache_path=tmp_path)
        nc_files = list(tmp_path.glob("weights_*.nc"))
        assert len(nc_files) == 2
