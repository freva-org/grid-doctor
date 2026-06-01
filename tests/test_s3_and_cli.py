"""Tests for CLI and S3 helper behaviour."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.cli.parser import get_parser, setup_logging_from_args
from grid_doctor.cli.script_utils import AutoRaiseSession, get_scratch
from grid_doctor.helpers import save_pyramid_to_s3


def _tiny_pyramid() -> dict[int, xr.Dataset]:
    """a two-level HEALPix-like pyramid for write tests."""

    def level_ds(npix: int) -> xr.Dataset:
        return xr.Dataset(
            {"tas": ("cell", np.arange(npix, dtype="float32"))},
            coords={"cell": np.arange(npix)},
        )

    return {0: level_ds(12), 1: level_ds(48)}


class TestGetParser:
    def test_creates_parser(self) -> None:
        parser = get_parser("test-prog", "A test program.")
        assert isinstance(parser, argparse.ArgumentParser)

    def test_s3_bucket_required(self) -> None:
        parser = get_parser("test-prog")
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = get_parser("test-prog")
        args = parser.parse_args(["--s3-bucket", "my-bucket"])
        assert args.s3_bucket == "my-bucket"
        assert args.verbose == 0


class TestSetupLoggingFromArgs:
    def test_sets_verbosity(self) -> None:
        args = argparse.Namespace(verbose=2)
        setup_logging_from_args(args)
        import grid_doctor.log as log_mod

        assert log_mod.get_level() == 10


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
        response = mock.Mock()
        response.raise_for_status.side_effect = Exception("404")
        mock_request.return_value = response
        session = AutoRaiseSession()
        with pytest.raises(Exception, match="404"):
            session.get("http://example.com")


class TestSavePyramidToS3Local:
    def test_writes_levels_to_disk(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "pyramid"
        save_pyramid_to_s3(pyramid, str(out), mode="w")
        for level in pyramid:
            assert (out / f"level_{level}.zarr").is_dir()

    def test_round_trips_values(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "pyramid"
        save_pyramid_to_s3(pyramid, str(out), mode="w")
        for level, dataset in pyramid.items():
            reloaded = xr.open_zarr(out / f"level_{level}.zarr")
            xr.testing.assert_allclose(reloaded[["tas"]], dataset[["tas"]])

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "nested" / "deeper" / "pyramid"
        save_pyramid_to_s3(pyramid, str(out), mode="w")
        assert (out / "level_0.zarr").is_dir()

    def test_s3_options_optional_for_local(self, tmp_path: Path) -> None:
        # since no s3_options supplied a local write must
        # not construct an S3 client.
        out = tmp_path / "pyramid"
        with mock.patch("grid_doctor.helpers.s3fs.S3FileSystem") as fs:
            save_pyramid_to_s3(_tiny_pyramid(), str(out), mode="w")
        fs.assert_not_called()


class TestSavePyramidToS3Remote:
    def test_uses_s3_map_store_for_s3_path(self) -> None:
        class _FakeDataset:
            store: object = None

            def to_zarr(self, store: object, **kwargs: object) -> None:
                self.store = store

        dataset = _FakeDataset()
        with (
            mock.patch("grid_doctor.helpers.s3fs.S3FileSystem") as fs,
            mock.patch(
                "grid_doctor.helpers.s3fs.S3Map", return_value="s3-store"
            ) as s3_map,
        ):
            save_pyramid_to_s3(
                {0: dataset},  # type: ignore[dict-item]
                "s3://bucket/pyr",
                {"key": "x", "secret": "y"},
                mode="w",
            )
        fs.assert_called_once()
        s3_map.assert_called_once()
        assert s3_map.call_args.kwargs["root"] == "s3://bucket/pyr/level_0.zarr"
        assert dataset.store == "s3-store"
