"""Tests for CLI and S3 helper behaviour."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest

from grid_doctor.cli.parser import get_parser, setup_logging_from_args
from grid_doctor.cli.script_utils import AutoRaiseSession, get_scratch


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
        with TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(verbose=2, log_dir=temp_dir)
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
