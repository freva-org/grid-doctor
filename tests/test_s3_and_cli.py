"""Tests for grid_doctor.helpers.save_pyramid_to_s3 and grid_doctor.cli."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.cli.parser import get_parser, setup_logging_from_args
from grid_doctor.cli.script_utils import AutoRaiseSession, get_scratch


# ── save_pyramid_to_s3 ────────────────────────────────────────────
class TestSavePyramidToS3:
    def _make_pyramid(self) -> dict[int, xr.Dataset]:
        """Create a tiny pyramid for testing."""
        import healpy as hp

        pyramid: dict[int, xr.Dataset] = {}
        for level in [0, 1]:
            nside = 2**level
            npix = hp.nside2npix(nside)
            ds = xr.Dataset(
                {"t": (("cell",), np.zeros(npix, dtype="float32"))},
                coords={"cell": np.arange(npix)},
                attrs={
                    "healpix_nside": nside,
                    "healpix_level": level,
                    "healpix_order": "nested",
                },
            )
            pyramid[level] = ds
        return pyramid

    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_calls_to_zarr(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        from grid_doctor.helpers import save_pyramid_to_s3

        pyramid = self._make_pyramid()
        mock_store = mock.MagicMock()
        mock_s3map.return_value = mock_store

        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
            assert mock_zarr.call_count == 2

    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_zarr_format_3(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        from grid_doctor.helpers import save_pyramid_to_s3

        pyramid = self._make_pyramid()
        mock_store = mock.MagicMock()
        mock_s3map.return_value = mock_store

        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(
                pyramid,
                "s3://bucket/test",
                s3_options={},
                zarr_format=3,
            )
            # zarr_format=3 should NOT add consolidated=True
            for call in mock_zarr.call_args_list:
                assert "consolidated" not in call.kwargs


# ── CLI parser ─────────────────────────────────────────────────────
class TestGetParser:
    def test_creates_parser(self) -> None:
        parser = get_parser("test-prog", "A test program.")
        assert isinstance(parser, argparse.ArgumentParser)

    def test_s3_bucket_required(self) -> None:
        parser = get_parser("test-prog")
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing s3_bucket

    def test_defaults(self) -> None:
        parser = get_parser("test-prog")
        args = parser.parse_args(["--s3-bucket", "my-bucket"])
        assert args.s3_bucket == "my-bucket"
        assert args.verbose == 0
        assert "s3" in args.s3_endpoint

    def test_verbosity_count(self) -> None:
        parser = get_parser("test-prog")
        args = parser.parse_args(["--s3-bucket", "my-bucket", "-vvv"])
        assert args.verbose == 3

    def test_credentials_file_default(self) -> None:
        parser = get_parser("test-prog")
        args = parser.parse_args(["--s3-bucket", "my-bucket"])
        assert args.s3_credentials_file == Path.home() / ".s3-credentials.json"


class TestSetupLoggingFromArgs:
    def test_sets_verbosity(self) -> None:
        args = argparse.Namespace(verbose=2)
        setup_logging_from_args(args)

        import grid_doctor.log as log_mod

        assert log_mod.get_level() == 10  # DEBUG


# ── script_utils ───────────────────────────────────────────────────
class TestGetScratch:
    def test_falls_back_to_tmp(self) -> None:
        with mock.patch("grid_doctor.cli.script_utils.getuser", return_value="nobody"):
            result = get_scratch("subdir")
            assert str(result).startswith("/tmp")
            assert result.name == "subdir"

    def test_appends_args(self) -> None:
        with mock.patch("grid_doctor.cli.script_utils.getuser", return_value="nobody"):
            result = get_scratch("a", "b", "c")
            assert result == Path("/tmp/a/b/c")


class TestAutoRaiseSession:
    @mock.patch("grid_doctor.cli.script_utils.requests.Session.request")
    def test_raises_on_error(self, mock_request: mock.Mock) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = Exception("404")
        mock_request.return_value = mock_response

        session = AutoRaiseSession()
        with pytest.raises(Exception, match="404"):
            session.get("http://example.com")
