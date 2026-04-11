"""Tests for ``grid_doctor.s3``.

Covers:
- ``get_s3_options``        — credential loading and botocore config injection
- ``_with_retry``           — exponential-backoff retry logic
- ``_inspect_store``        — store introspection and ``WritePlan`` production
- ``_execute_write_plan``   — two-step write decomposition
- ``save_pyramid_to_s3``    — public API, explicit modes and ``mode="auto"``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.s3 import (
    _chunk_key_v2,
    _chunk_key_v3,
    _execute_write_plan,
    _inspect_store,
    _with_retry,
    get_s3_options,
    save_pyramid_to_s3,
)
from grid_doctor.types import WritePlan


# ===================================================================
# Shared test helpers
# ===================================================================


def _creds_file(tmp_path: Path) -> Path:
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"accessKey": "AKID", "secretKey": "SECRET"}))
    return path


def _healpix_ds(
    level: int = 2,
    n_time: int = 3,
    vars: list[str] | None = None,
) -> xr.Dataset:
    """Small in-memory HEALPix dataset for testing."""
    npix = 12 * (4**level)
    if vars is None:
        vars = ["tas"]
    rng = np.random.default_rng(0)
    data_vars = {
        v: (("time", "cell"), rng.random((n_time, npix)).astype("float32"))
        for v in vars
    }
    return xr.Dataset(
        data_vars,
        coords={"time": np.arange(n_time), "cell": np.arange(npix)},
        attrs={
            "healpix_nside": 2**level,
            "healpix_level": level,
            "healpix_order": "nested",
        },
    )


def _pyramid(levels: tuple[int, ...] = (0, 1), n_time: int = 3) -> dict[int, xr.Dataset]:
    return {level: _healpix_ds(level=level, n_time=n_time) for level in levels}



def _existing_store_ds(
    vars: list[str],
    n_time: int,
    chunk_time: int = 1,
) -> xr.Dataset:
    """Fake dataset representing what xr.open_zarr would return for an existing store.

    Each data variable carries ``encoding["chunks"]`` so the validate path
    in ``_inspect_store`` can derive chunk indices without touching S3.
    """
    npix = 12
    rng = np.random.default_rng(1)
    data_vars = {}
    for v in vars:
        arr = xr.DataArray(
            rng.random((n_time, npix)).astype("float32"),
            dims=("time", "cell"),
        )
        arr.encoding["chunks"] = (chunk_time, npix)
        data_vars[v] = arr
    return xr.Dataset(
        data_vars,
        coords={"time": np.arange(n_time), "cell": np.arange(npix)},
    )


def _fs_mock(
    *,
    store_non_empty: bool = True,
    last_chunk_exists: bool = True,
    is_v2: bool = True,
) -> mock.MagicMock:
    """Minimal mock ``s3fs.S3FileSystem`` for ``_inspect_store`` tests.

    Since ``_inspect_store`` now delegates all metadata reading to
    ``xr.open_zarr``, the filesystem mock only needs to handle:

    1. ``fs.exists(level_path)`` + ``fs.ls(level_path)`` — store presence.
    2. ``fs.exists(f"{level_path}/.zmetadata")`` — zarr version detection
       for the validate path (``is_v2`` controls the return value).
    3. ``fs.exists(chunk_key)`` — validate chunk presence.

    ``xr.open_zarr`` itself is mocked separately in each test via
    ``mock.patch``.
    """
    fs = mock.MagicMock()
    if not store_non_empty:
        fs.exists.return_value = False
        fs.ls.return_value = []
        return fs

    fs.ls.return_value = ["level_2.zarr/.zgroup"]
    # exists calls: level_path, [.zmetadata for version detect], [chunk_key]
    fs.exists.side_effect = [True, is_v2, last_chunk_exists]
    return fs


# ===================================================================
# get_s3_options
# ===================================================================


class TestGetS3Options:
    def test_returns_endpoint_key_and_secret(self, tmp_path: Path) -> None:
        opts = get_s3_options(
            "https://s3.example.com", _creds_file(tmp_path)
        )
        assert opts["endpoint_url"] == "https://s3.example.com"
        assert opts["key"] == "AKID"
        assert opts["secret"] == "SECRET"

    def test_client_kwargs_contains_botocore_config(self, tmp_path: Path) -> None:
        pytest.importorskip("botocore")
        opts = get_s3_options("https://s3.example.com", _creds_file(tmp_path))
        cfg = opts["client_kwargs"]["config"]
        assert cfg.read_timeout == 300
        assert cfg.connect_timeout == 90
        assert cfg.retries == {"max_attempts": 10, "mode": "adaptive"}

    def test_custom_timeout_values_forwarded(self, tmp_path: Path) -> None:
        pytest.importorskip("botocore")
        opts = get_s3_options(
            "https://s3.example.com",
            _creds_file(tmp_path),
            read_timeout=120,
            connect_timeout=30,
            max_attempts=3,
        )
        cfg = opts["client_kwargs"]["config"]
        assert cfg.read_timeout == 120
        assert cfg.connect_timeout == 30
        assert cfg.retries["max_attempts"] == 3

    def test_extra_kwargs_merged(self, tmp_path: Path) -> None:
        opts = get_s3_options(
            "https://s3.example.com",
            _creds_file(tmp_path),
            anon=False,
        )
        assert opts["anon"] is False

    def test_extra_client_kwargs_merged(self, tmp_path: Path) -> None:
        opts = get_s3_options(
            "https://s3.example.com",
            _creds_file(tmp_path),
            client_kwargs={"region_name": "eu-west-1"},
        )
        assert opts["client_kwargs"]["region_name"] == "eu-west-1"


# ===================================================================
# _with_retry
# ===================================================================


class TestWithRetry:
    def test_succeeds_on_first_call(self) -> None:
        fn = mock.Mock()
        _with_retry(fn, max_retries=3, backoff=2.0)
        assert fn.call_count == 1

    def test_retries_on_exception_then_succeeds(self) -> None:
        fn = mock.Mock(side_effect=[RuntimeError("boom"), RuntimeError("boom"), None])
        with mock.patch("grid_doctor.s3._time.sleep"):
            _with_retry(fn, max_retries=3, backoff=2.0)
        assert fn.call_count == 3

    def test_reraises_after_max_retries(self) -> None:
        fn = mock.Mock(side_effect=RuntimeError("persistent"))
        with mock.patch("grid_doctor.s3._time.sleep"):
            with pytest.raises(RuntimeError, match="persistent"):
                _with_retry(fn, max_retries=2, backoff=2.0)
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_exponential_backoff_delays(self) -> None:
        fn = mock.Mock(side_effect=[ValueError, ValueError, None])
        with mock.patch("grid_doctor.s3._time.sleep") as mock_sleep:
            _with_retry(fn, max_retries=3, backoff=2.0, base_delay=1.0)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]  # base * 2^0, base * 2^1

    def test_zero_retries_fails_immediately(self) -> None:
        fn = mock.Mock(side_effect=OSError("no retry"))
        with pytest.raises(OSError):
            _with_retry(fn, max_retries=0, backoff=2.0)
        assert fn.call_count == 1


# ===================================================================
# _inspect_store
# ===================================================================


class TestInspectStore:
    """Tests for ``_inspect_store``.

    ``xr.open_zarr`` is patched to return a fake dataset describing the
    existing store, so tests are independent of Zarr format (v2/v3) and
    require no JSON fixture factories.  ``fs`` is a minimal mock that only
    needs to handle the store-presence check and the validate chunk-key
    lookup.
    """

    LEVEL_PATH = "s3://bucket/level_2.zarr"

    def _run(
        self,
        incoming_ds: xr.Dataset,
        existing_ds: xr.Dataset | None,
        *,
        validate: bool = False,
        last_chunk_exists: bool = True,
        is_v2: bool = True,
    ) -> WritePlan:
        """Run ``_inspect_store`` with ``xr.open_zarr`` mocked.

        If *existing_ds* is ``None``, ``xr.open_zarr`` raises, simulating an
        unreadable store.
        """
        fs = _fs_mock(
            store_non_empty=existing_ds is not None,
            last_chunk_exists=last_chunk_exists,
            is_v2=is_v2,
        )
        open_zarr_side = (
            existing_ds
            if existing_ds is not None
            else mock.Mock(side_effect=FileNotFoundError("no store"))
        )
        with mock.patch("grid_doctor.s3.xr.open_zarr", return_value=open_zarr_side):
            return _inspect_store(fs, self.LEVEL_PATH, incoming_ds, validate=validate)

    def test_absent_store_returns_full_write_plan(self) -> None:
        ds = _healpix_ds(vars=["tas"])
        plan = self._run(ds, existing_ds=None)
        assert plan.mode == "w"
        assert plan.new_vars == ["tas"]
        assert plan.existing_vars == []
        assert plan.n_existing_time is None
        assert plan.append_time is False

    def test_unreadable_store_falls_back_to_w(self) -> None:
        """xr.open_zarr raising any exception falls back to mode='w'."""
        ds = _healpix_ds(vars=["tas"])
        fs = _fs_mock(store_non_empty=True)
        with mock.patch(
            "grid_doctor.s3.xr.open_zarr",
            side_effect=Exception("bad metadata"),
        ):
            plan = _inspect_store(fs, self.LEVEL_PATH, ds)
        assert plan.mode == "w"

    def test_same_schema_same_time_returns_r_plus(self) -> None:
        ds = _healpix_ds(vars=["tas"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3)
        plan = self._run(ds, existing)
        assert plan.mode == "r+"
        assert plan.new_vars == []
        assert plan.existing_vars == ["tas"]
        assert plan.n_existing_time == 3
        assert plan.append_time is False

    def test_new_variable_detected(self) -> None:
        """Incoming dataset has 'pr' in addition to existing 'tas'."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3)
        plan = self._run(ds, existing)
        assert plan.mode == "a"
        assert "pr" in plan.new_vars
        assert "tas" in plan.existing_vars
        assert plan.append_time is False

    def test_more_time_steps_detected(self) -> None:
        """Incoming dataset has 5 time steps but store only has 3."""
        ds = _healpix_ds(vars=["tas"], n_time=5)
        existing = _existing_store_ds(["tas"], n_time=3)
        plan = self._run(ds, existing)
        assert plan.mode == "a"
        assert plan.new_vars == []
        assert plan.n_existing_time == 3
        assert plan.append_time is True

    def test_new_vars_and_more_time(self) -> None:
        """Both a new variable and additional time steps present."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=5)
        existing = _existing_store_ds(["tas"], n_time=3)
        plan = self._run(ds, existing)
        assert plan.mode == "a"
        assert "pr" in plan.new_vars
        assert plan.n_existing_time == 3
        assert plan.append_time is True

    def test_validate_clean_store_keeps_mode(self) -> None:
        """validate=True with all chunks present must not change the mode."""
        ds = _healpix_ds(vars=["tas"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3, chunk_time=1)
        plan = self._run(ds, existing, validate=True, last_chunk_exists=True)
        assert plan.mode == "r+"

    def test_validate_missing_last_chunk_falls_back_to_w(self) -> None:
        """validate=True with a missing chunk must fall back to mode='w'."""
        ds = _healpix_ds(vars=["tas"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3, chunk_time=1)
        plan = self._run(ds, existing, validate=True, last_chunk_exists=False)
        assert plan.mode == "w"
        assert plan.new_vars == ["tas"]
        assert plan.n_existing_time is None

    def test_validate_uses_v2_chunk_key_format(self) -> None:
        """Validate path must use dot-separated keys for v2 stores."""
        ds = _healpix_ds(vars=["tas"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3, chunk_time=1)
        checked_keys: list[str] = []

        def _exists(path: str) -> bool:
            checked_keys.append(path)
            return True  # all chunks present

        fs = _fs_mock(store_non_empty=True, last_chunk_exists=True, is_v2=True)
        fs.exists.side_effect = _exists
        with mock.patch("grid_doctor.s3.xr.open_zarr", return_value=existing):
            _inspect_store(fs, self.LEVEL_PATH, ds, validate=True)

        chunk_keys = [k for k in checked_keys if "/tas/" in k]
        assert chunk_keys, "No chunk key was checked"
        assert "." in chunk_keys[-1], f"Expected dot-separated v2 key, got {chunk_keys[-1]!r}"

    def test_validate_uses_v3_chunk_key_format(self) -> None:
        """Validate path must use 'c/' prefix for v3 stores."""
        ds = _healpix_ds(vars=["tas"], n_time=3)
        existing = _existing_store_ds(["tas"], n_time=3, chunk_time=1)
        checked_keys: list[str] = []

        def _exists(path: str) -> bool:
            checked_keys.append(path)
            return True

        fs = _fs_mock(store_non_empty=True, last_chunk_exists=True, is_v2=False)
        fs.exists.side_effect = _exists
        with mock.patch("grid_doctor.s3.xr.open_zarr", return_value=existing):
            _inspect_store(fs, self.LEVEL_PATH, ds, validate=True)

        chunk_keys = [k for k in checked_keys if "/tas/" in k]
        assert chunk_keys, "No chunk key was checked"
        assert "/c/" in chunk_keys[-1], f"Expected v3 'c/' key, got {chunk_keys[-1]!r}"

    def test_no_time_dimension_no_append(self) -> None:
        """Dataset without a time dimension should never trigger append_time."""
        ds = xr.Dataset(
            {"mask": (("cell",), np.zeros(48, dtype="float32"))},
            coords={"cell": np.arange(48)},
        )
        existing = xr.Dataset(
            {"mask": (("cell",), np.zeros(48, dtype="float32"))},
            coords={"cell": np.arange(48)},
        )
        plan = self._run(ds, existing)
        assert plan.append_time is False
        assert plan.n_existing_time is None



class TestChunkKeyHelpers:
    """Unit tests for the chunk-key builder functions."""

    def test_v2_single_dim(self) -> None:
        assert _chunk_key_v2("s3://b/l.zarr", "time", [0]) == "s3://b/l.zarr/time/0"

    def test_v2_two_dims(self) -> None:
        assert _chunk_key_v2("s3://b/l.zarr", "tas", [2, 0]) == "s3://b/l.zarr/tas/2.0"

    def test_v3_single_dim(self) -> None:
        assert _chunk_key_v3("s3://b/l.zarr", "time", [0]) == "s3://b/l.zarr/time/c/0"

    def test_v3_two_dims(self) -> None:
        assert _chunk_key_v3("s3://b/l.zarr", "tas", [2, 0]) == "s3://b/l.zarr/tas/c/2/0"



class TestExecuteWritePlan:
    """Tests for the write-plan executor.

    ``_with_retry`` is patched to call ``fn()`` directly (no sleeping),
    isolating decomposition logic from retry behaviour.
    ``xr.Dataset.to_zarr`` is patched to record call kwargs.
    """

    def _plan(self, **kwargs: Any) -> WritePlan:
        defaults: dict[str, Any] = dict(
            mode="w",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=None,
            append_time=False,
        )
        defaults.update(kwargs)
        return WritePlan(**defaults)

    def _run(
        self,
        ds: xr.Dataset,
        plan: WritePlan,
        *,
        zarr_format: int = 2,
    ) -> list[dict[str, Any]]:
        """Execute plan and return list of kwargs dicts passed to to_zarr."""
        store = mock.MagicMock()
        calls: list[dict[str, Any]] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> None:
            calls.append(kwargs)

        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds,
                store,
                plan,
                zarr_format=zarr_format,
                compute=True,
                max_retries=0,
                retry_backoff=0.0,
            )
        return calls

    def test_mode_w_single_call(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"))
        assert len(calls) == 1
        assert calls[0]["mode"] == "w"

    def test_mode_r_plus_single_call(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="r+"))
        assert len(calls) == 1
        assert calls[0]["mode"] == "r+"

    def test_zarr_format_2_adds_consolidated(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"), zarr_format=2)
        assert calls[0].get("consolidated") is True

    def test_zarr_format_3_omits_consolidated(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"), zarr_format=3)
        assert "consolidated" not in calls[0]

    def test_new_vars_only_single_append_call(self) -> None:
        ds = _healpix_ds(vars=["tas", "pr"], n_time=3)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=False,
        )
        calls = self._run(ds, plan)
        assert len(calls) == 1
        assert calls[0]["mode"] == "a"
        assert "append_dim" not in calls[0]

    def test_new_vars_sliced_to_existing_time(self) -> None:
        """New variable must be written only for the time range already in store."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        sliced_times: list[int] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> None:
            if "append_dim" not in kwargs:
                # First call: adding new variable
                sliced_times.append(self_ds.sizes.get("time", -1))

        store = mock.MagicMock()
        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds, store, plan,
                zarr_format=2, compute=True, max_retries=0, retry_backoff=0.0,
            )
        assert sliced_times == [3], "New variable must be written for existing 3 time steps only"

    def test_append_time_only_uses_append_dim(self) -> None:
        ds = _healpix_ds(vars=["tas"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        calls = self._run(ds, plan)
        assert len(calls) == 1
        assert calls[0].get("append_dim") == "time"

    def test_append_time_sliced_correctly(self) -> None:
        """Append call must contain only the new time steps."""
        ds = _healpix_ds(vars=["tas"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        appended_times: list[int] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> None:
            if kwargs.get("append_dim") == "time":
                appended_times.append(self_ds.sizes.get("time", -1))

        store = mock.MagicMock()
        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds, store, plan,
                zarr_format=2, compute=True, max_retries=0, retry_backoff=0.0,
            )
        assert appended_times == [2], "Must append only the 2 new time steps"

    def test_new_vars_and_new_time_two_calls(self) -> None:
        """Both a new variable and more time steps → two to_zarr calls."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        calls = self._run(ds, plan)
        assert len(calls) == 2
        # Step 1: add new var, no append_dim
        assert calls[0]["mode"] == "a"
        assert "append_dim" not in calls[0]
        # Step 2: append time for all vars
        assert calls[1]["mode"] == "a"
        assert calls[1].get("append_dim") == "time"

    def test_encoding_forwarded(self) -> None:
        ds = _healpix_ds()
        store = mock.MagicMock()
        calls: list[dict[str, Any]] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> None:
            calls.append(kwargs)

        enc = {"tas": {"dtype": "float32", "compressor": None}}
        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds, store, self._plan(mode="w"),
                zarr_format=2, compute=True,
                max_retries=0, retry_backoff=0.0,
                encoding=enc,
            )
        assert calls[0]["encoding"] == enc


# ===================================================================
# save_pyramid_to_s3 — explicit modes
# ===================================================================


class TestSavePyramidToS3ExplicitModes:
    """Tests for the public function when an explicit mode is given.

    s3fs is mocked at the module level; ``to_zarr`` is mocked on the class.
    """

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3.s3fs.S3Map")
    def test_mode_w_calls_to_zarr_once_per_level(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        mock_s3map.return_value = mock.MagicMock()
        pyramid = _pyramid(levels=(0, 1))
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
        assert mock_zarr.call_count == len(pyramid)

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3.s3fs.S3Map")
    def test_level_paths_contain_level_number(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        pyramid = _pyramid(levels=(2, 3))
        with mock.patch.object(xr.Dataset, "to_zarr"):
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
        calls = mock_s3map.call_args_list
        roots = {call.kwargs.get("root", call.args[0] if call.args else "") for call in calls}
        assert any("level_2" in r for r in roots)
        assert any("level_3" in r for r in roots)

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3.s3fs.S3Map")
    def test_zarr_format_2_writes_consolidated(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        mock_s3map.return_value = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode="w", zarr_format=2
            )
        for call in mock_zarr.call_args_list:
            assert call.kwargs.get("consolidated") is True

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3.s3fs.S3Map")
    def test_zarr_format_3_omits_consolidated(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        mock_s3map.return_value = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode="w", zarr_format=3
            )
        for call in mock_zarr.call_args_list:
            assert "consolidated" not in call.kwargs

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3.s3fs.S3Map")
    def test_per_level_encoding_forwarded(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        mock_s3map.return_value = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        enc = {0: {"tas": {"dtype": "float32"}}}
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode="w", encoding=enc
            )
        assert mock_zarr.call_args_list[0].kwargs["encoding"] == enc[0]


# ===================================================================
# save_pyramid_to_s3 — mode="auto"
# ===================================================================


class TestSavePyramidToS3Auto:
    """Tests for ``mode='auto'`` path.

    ``_inspect_store`` and ``_execute_write_plan`` are mocked so these tests
    verify only the glue logic inside ``save_pyramid_to_s3``: plan routing
    and validate pass-through.
    """

    def _run_auto(
        self,
        pyramid: dict[int, xr.Dataset],
        plan: WritePlan,
        *,
        validate: bool = False,
    ) -> mock.Mock:
        """Run save_pyramid_to_s3(mode='auto') with mocked internals.

        Returns the mock for ``_execute_write_plan`` so callers can assert
        on how it was called.
        """
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._inspect_store", return_value=plan) as mock_inspect,
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
        ):
            mock_s3map.return_value = mock.MagicMock()
            save_pyramid_to_s3(
                pyramid,
                "s3://bucket/test",
                s3_options={},
                mode="auto",
                validate=validate,
            )
            return mock_execute

    def _clean_plan(self, **kwargs: Any) -> WritePlan:
        defaults: dict[str, Any] = dict(
            mode="r+",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=False,
        )
        defaults.update(kwargs)
        return WritePlan(**defaults)

    def test_clean_plan_is_executed(self) -> None:
        pyramid = _pyramid(levels=(0,))
        plan = self._clean_plan(mode="r+")
        mock_execute = self._run_auto(pyramid, plan)
        assert mock_execute.call_count == 1

    def test_validate_passed_to_inspect_store(self) -> None:
        pyramid = _pyramid(levels=(0,))
        plan = self._clean_plan()
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._inspect_store", return_value=plan) as mock_inspect,
            mock.patch("grid_doctor.s3._execute_write_plan"),
        ):
            mock_s3map.return_value = mock.MagicMock()
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={},
                mode="auto", validate=True,
            )
        _, kwargs = mock_inspect.call_args
        assert kwargs.get("validate") is True

    def test_inconsistent_store_plan_executed_as_w(self) -> None:
        """_inspect_store returning mode='w' (inconsistent store) must be
        executed without further intervention by save_pyramid_to_s3."""
        pyramid = _pyramid(levels=(0,))
        plan = self._clean_plan(mode="w", new_vars=["tas"], existing_vars=[])
        mock_execute = self._run_auto(pyramid, plan)
        assert mock_execute.call_count == 1
        executed_plan: WritePlan = mock_execute.call_args[0][2]
        assert executed_plan.mode == "w"

    def test_auto_processes_all_levels(self) -> None:
        pyramid = _pyramid(levels=(0, 1, 2))
        plan = self._clean_plan()
        mock_execute = self._run_auto(pyramid, plan)
        assert mock_execute.call_count == len(pyramid)

    def test_explicit_mode_bypasses_inspect(self) -> None:
        """Passing mode='w' must not call _inspect_store at all."""
        pyramid = _pyramid(levels=(0,))
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch.object(xr.Dataset, "to_zarr"),
        ):
            mock_s3map.return_value = mock.MagicMock()
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode="w"
            )
        mock_inspect.assert_not_called()
