"""Command line parser and helpers (``grid_doctor.cli``)."""

from __future__ import annotations

import argparse
from unittest import mock

import pytest

from grid_doctor.cli.parser import get_parser, setup_logging_from_args
from grid_doctor.cli.script_utils import AutoRaiseSession


class TestGetParser:
    def test_creates_parser(self) -> None:
        parser = get_parser("test-prog", "A test program.")
        assert isinstance(parser, argparse.ArgumentParser)

    def test_s3_bucket_required(self) -> None:
        parser = get_parser("test-prog")
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_s3_endpoint_required(self) -> None:
        parser = get_parser("test-prog")
        with pytest.raises(SystemExit):
            parser.parse_args(["--s3-bucket", "my-bucket"])

    def test_defaults(self) -> None:
        parser = get_parser("test-prog")
        args = parser.parse_args(["--s3-bucket", "my-bucket", "--s3-endpoint", "https://s3.example.org"])
        assert args.s3_bucket == "my-bucket"
        assert args.s3_endpoint == "https://s3.example.org"
        assert args.verbose == 0


class TestSetupLoggingFromArgs:
    def test_sets_verbosity(self) -> None:
        args = argparse.Namespace(verbose=2)
        setup_logging_from_args(args)
        import grid_doctor._log as log_mod

        assert log_mod.get_level() == 10


class TestAutoRaiseSession:
    @mock.patch("grid_doctor.cli.script_utils.requests.Session.request")
    def test_raises_on_error(self, mock_request: mock.Mock) -> None:
        response = mock.Mock()
        response.raise_for_status.side_effect = Exception("404")
        mock_request.return_value = response
        session = AutoRaiseSession()
        with pytest.raises(Exception, match="404"):
            session.get("http://example.com")
