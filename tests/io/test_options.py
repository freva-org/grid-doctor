"""Chunk sizing and S3 options (``grid_doctor.io._options``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grid_doctor.io._options import (
    chunk_for_target_store_size,
    get_s3_options,
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
