"""Tests for ``grid_doctor.s3``.

Covers:
- ``get_s3_options``        — credential loading and S3 timeout configuration
- ``_with_retry``           — exponential-backoff retry logic
- ``_inspect_store``        — store introspection and ``WritePlan`` production
- ``_execute_write_plan``   — two-step write decomposition
- ``save_pyramid_to_s3``    — public API, explicit modes and ``mode="auto"``
- ``_build_write_delayed``  — delayed write construction and two-step fallback
- ``_save_pyramid_parallel``— parallel upload via distributed client
- ``_get_or_create_client`` — context manager for client lifecycle management
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.s3 import (
    _build_write_delayed,
    _chunk_key_v2,
    _chunk_key_v3,
    _execute_write_plan,
    _get_or_create_client,
    _inspect_store,
    _is_gpu_backed,
    _save_pyramid_parallel,
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

    def test_config_kwargs_are_json_serialisable(self, tmp_path: Path) -> None:
        """config_kwargs must be a plain dict so fsspec can serialise the
        filesystem to JSON (required by zarr when reconstructing async stores)."""
        import json
        opts = get_s3_options("https://s3.example.com", _creds_file(tmp_path))
        cfg = opts["config_kwargs"]
        assert cfg["read_timeout"] == 300
        assert cfg["connect_timeout"] == 90
        assert cfg["retries"] == {"max_attempts": 10, "mode": "adaptive"}
        # Must be JSON-serialisable — no botocore.Config objects
        json.dumps(cfg)  # raises if not serialisable

    def test_custom_timeout_values_forwarded(self, tmp_path: Path) -> None:
        opts = get_s3_options(
            "https://s3.example.com",
            _creds_file(tmp_path),
            read_timeout=120,
            connect_timeout=30,
            max_attempts=3,
        )
        cfg = opts["config_kwargs"]
        assert cfg["read_timeout"] == 120
        assert cfg["connect_timeout"] == 30
        assert cfg["retries"]["max_attempts"] == 3

    def test_extra_kwargs_merged(self, tmp_path: Path) -> None:
        opts = get_s3_options(
            "https://s3.example.com",
            _creds_file(tmp_path),
            anon=False,
        )
        assert opts["anon"] is False



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
            # .zmetadata absent → v3 store; everything else (chunk key) present
            return not path.endswith(".zmetadata")

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



# ===================================================================
# _is_gpu_backed
# ===================================================================


class TestIsGpuBacked:
    """Tests for ``_is_gpu_backed``.

    Now simply checks ``ds.attrs["grid_doctor_backend"] == "cupy"``,
    so tests just need plain xarray datasets with the right attrs.
    """

    def test_cupy_backend_returns_true(self) -> None:
        ds = _healpix_ds()
        ds.attrs["grid_doctor_backend"] = "cupy"
        assert _is_gpu_backed(ds) is True

    def test_scipy_backend_returns_false(self) -> None:
        ds = _healpix_ds()
        ds.attrs["grid_doctor_backend"] = "scipy"
        assert _is_gpu_backed(ds) is False

    def test_numba_backend_returns_false(self) -> None:
        ds = _healpix_ds()
        ds.attrs["grid_doctor_backend"] = "numba"
        assert _is_gpu_backed(ds) is False

    def test_missing_attr_returns_false(self) -> None:
        """Datasets not produced by grid-doctor have no backend attr."""
        ds = _healpix_ds()
        assert "grid_doctor_backend" not in ds.attrs
        assert _is_gpu_backed(ds) is False

    def test_real_healpix_ds_without_attr_returns_false(self) -> None:
        assert _is_gpu_backed(_healpix_ds()) is False


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

    def test_mode_w_two_phase_write(self) -> None:
        """mode='w' produces two to_zarr calls: metadata init then chunk write."""
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"))
        assert len(calls) == 2
        # Phase 1: metadata init — mode=w, compute=False, consolidated=False
        assert calls[0]["mode"] == "w"
        assert calls[0]["compute"] is False
        assert calls[0].get("consolidated") is False
        # Phase 2: chunk write — mode=r+, never clears store
        assert calls[1]["mode"] == "r+"

    def test_mode_r_plus_single_call(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="r+"))
        assert len(calls) == 1
        assert calls[0]["mode"] == "r+"

    def test_zarr_format_2_adds_consolidated_to_chunk_write(self) -> None:
        """consolidated=True must appear on the chunk-write call (phase 2), not the metadata init."""
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"), zarr_format=2)
        # Phase 1 (metadata init) always uses consolidated=False
        assert calls[0].get("consolidated") is False
        # Phase 2 (chunk write) carries consolidated=True
        assert calls[1].get("consolidated") is True

    def test_zarr_format_3_omits_consolidated(self) -> None:
        ds = _healpix_ds()
        calls = self._run(ds, self._plan(mode="w"), zarr_format=3)
        # Neither call should have consolidated for v3
        for call in calls:
            assert "consolidated" not in call or call["consolidated"] is False

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


    def test_mode_w_initialises_metadata_before_retry(self) -> None:
        """mode='w' must write metadata once eagerly, then retry chunks with r+.

        Verifies the two-phase approach that prevents retries from clearing
        successfully-written chunks: the first to_zarr call uses mode='w'
        with compute=False (metadata only), and the retried call uses mode='r+'
        (chunk writes, no store clearing).
        """
        ds = _healpix_ds()
        store = mock.MagicMock()
        calls: list[dict[str, Any]] = []

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return mock.MagicMock()  # return a mock delayed

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds,
                store,
                self._plan(mode="w"),
                zarr_format=2,
                compute=True,
                max_retries=3,
                retry_backoff=2.0,
            )

        assert len(calls) == 2, f"Expected 2 to_zarr calls, got {len(calls)}"
        # Phase 1: metadata init
        assert calls[0]["mode"] == "w"
        assert calls[0]["compute"] is False
        assert calls[0]["consolidated"] is False
        # Phase 2: chunk write (retried safely)
        assert calls[1]["mode"] == "r+"
        assert calls[1].get("compute") is not False  # compute=True (from base_opts)

    def test_mode_w_retry_does_not_rewipe_store(self) -> None:
        """Retrying mode='w' must use mode='r+', never mode='w' again."""
        ds = _healpix_ds()
        store = mock.MagicMock()
        retry_modes: list[str] = []

        def _capturing_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            # Simulate one failure then success
            try:
                fn()
            except Exception:
                pass
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> Any:
            retry_modes.append(kwargs.get("mode", ""))
            return mock.MagicMock()

        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_capturing_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds,
                store,
                self._plan(mode="w"),
                zarr_format=2,
                compute=True,
                max_retries=3,
                retry_backoff=2.0,
            )

        # All calls inside the retry loop must use r+, never w again
        retry_call_modes = retry_modes[1:]  # skip the initial metadata init call
        assert all(m == "r+" for m in retry_call_modes), (
            f"Retry calls must use mode='r+', got: {retry_call_modes}"
        )

    def test_mode_r_plus_uses_single_retried_call(self) -> None:
        """mode='r+' must use a single retried to_zarr call (no two-phase split)."""
        ds = _healpix_ds()
        calls: list[dict[str, Any]] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return None

        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds,
                mock.MagicMock(),
                self._plan(mode="r+"),
                zarr_format=2,
                compute=True,
                max_retries=3,
                retry_backoff=2.0,
            )

        assert len(calls) == 1
        assert calls[0]["mode"] == "r+"

    def test_mode_a_uses_single_retried_call(self) -> None:
        """mode='a' must use a single retried to_zarr call (no two-phase split)."""
        ds = _healpix_ds(vars=["tas"], n_time=5)
        calls: list[dict[str, Any]] = []

        def _fake_retry(fn: Any, *, max_retries: int, backoff: float) -> None:
            fn()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> Any:
            calls.append(dict(kwargs))
            return None

        plan = self._plan(
            mode="a",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        with (
            mock.patch("grid_doctor.s3._with_retry", side_effect=_fake_retry),
            mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr),
        ):
            _execute_write_plan(
                ds,
                mock.MagicMock(),
                plan,
                zarr_format=2,
                compute=True,
                max_retries=3,
                retry_backoff=2.0,
            )

        assert all(c["mode"] == "a" for c in calls), (
            "Append mode must never use mode='w' or mode='r+'"
        )

# ===================================================================
# save_pyramid_to_s3 — explicit modes
# ===================================================================


class TestSavePyramidToS3ExplicitModes:
    """Tests for save_pyramid_to_s3 with explicit modes.

    Since save_pyramid_to_s3 now always routes through _save_pyramid_parallel,
    these tests verify that the correct arguments are forwarded.  Write-level
    behaviour (consolidated, encoding, to_zarr call count) is already tested
    in TestBuildWriteDelayed and TestSavePyramidParallel.
    """

    def _run(
        self,
        pyramid: dict[int, xr.Dataset],
        *,
        mode: str = "w",
        **kwargs: Any,
    ) -> mock.MagicMock:
        """Call save_pyramid_to_s3 and return the mock for _save_pyramid_parallel."""
        mock_client = mock.MagicMock()

        @contextmanager
        def _fake_ctx(*args: Any, **kw: Any):  # type: ignore[no-untyped-def]
            yield mock_client

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._get_or_create_client", side_effect=_fake_ctx),
            mock.patch("grid_doctor.s3._save_pyramid_parallel") as mock_parallel,
        ):
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode=mode, **kwargs
            )
        return mock_parallel

    def test_parallel_path_called_for_explicit_mode(self) -> None:
        mock_parallel = self._run(_pyramid(levels=(0, 1)), mode="w")
        assert mock_parallel.called

    def test_mode_forwarded_to_parallel(self) -> None:
        mock_parallel = self._run(_pyramid(levels=(0,)), mode="w")
        assert mock_parallel.call_args[1]["mode"] == "w"

    def test_s3_path_normalised(self) -> None:
        """s3:// prefix and trailing slash must be stripped before handoff."""
        mock_parallel = self._run(_pyramid(levels=(0,)))
        passed_path = mock_parallel.call_args[0][2]
        assert not passed_path.startswith("s3://")
        assert not passed_path.endswith("/")

    def test_zarr_format_2_forwarded(self) -> None:
        mock_parallel = self._run(_pyramid(levels=(0,)), zarr_format=2)
        assert mock_parallel.call_args[1]["zarr_format"] == 2

    def test_zarr_format_3_forwarded(self) -> None:
        mock_parallel = self._run(_pyramid(levels=(0,)), zarr_format=3)
        assert mock_parallel.call_args[1]["zarr_format"] == 3

    def test_per_level_encoding_forwarded(self) -> None:
        enc = {0: {"tas": {"dtype": "float32"}}}
        mock_parallel = self._run(_pyramid(levels=(0,)), encoding=enc)
        assert mock_parallel.call_args[1]["encoding"] == enc

    def test_max_retries_forwarded(self) -> None:
        mock_parallel = self._run(_pyramid(levels=(0,)), max_retries=7)
        assert mock_parallel.call_args[1]["max_retries"] == 7


# ===================================================================
# save_pyramid_to_s3 — mode="auto"
# ===================================================================


class TestSavePyramidToS3Auto:
    """Tests for save_pyramid_to_s3 with mode='auto'.

    With the parallel-first design, 'auto' mode is forwarded to
    _save_pyramid_parallel.  Inspect logic and plan routing within the
    parallel path are tested in TestSavePyramidParallel.
    """

    def _run_auto(
        self,
        pyramid: dict[int, xr.Dataset],
        *,
        validate: bool = False,
    ) -> mock.MagicMock:
        """Call save_pyramid_to_s3(mode='auto') and return the _save_pyramid_parallel mock."""
        mock_client = mock.MagicMock()

        @contextmanager
        def _fake_ctx(*args: Any, **kw: Any):  # type: ignore[no-untyped-def]
            yield mock_client

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._get_or_create_client", side_effect=_fake_ctx),
            mock.patch("grid_doctor.s3._save_pyramid_parallel") as mock_parallel,
        ):
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={},
                mode="auto", validate=validate,
            )
        return mock_parallel

    def test_auto_mode_forwarded_to_parallel(self) -> None:
        mock_parallel = self._run_auto(_pyramid(levels=(0,)))
        assert mock_parallel.call_args[1]["mode"] == "auto"

    def test_validate_forwarded_to_parallel(self) -> None:
        mock_parallel = self._run_auto(_pyramid(levels=(0,)), validate=True)
        assert mock_parallel.call_args[1]["validate"] is True

    def test_parallel_called_once_with_full_pyramid(self) -> None:
        """_save_pyramid_parallel receives the entire pyramid in one call."""
        pyramid = _pyramid(levels=(0, 1, 2))
        mock_parallel = self._run_auto(pyramid)
        assert mock_parallel.call_count == 1
        assert mock_parallel.call_args[0][0] is pyramid

    def test_explicit_mode_bypasses_inspect(self) -> None:
        """_inspect_store must not be called by save_pyramid_to_s3 directly."""
        mock_client = mock.MagicMock()

        @contextmanager
        def _fake_ctx(*args: Any, **kw: Any):  # type: ignore[no-untyped-def]
            yield mock_client

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._get_or_create_client", side_effect=_fake_ctx),
            mock.patch("grid_doctor.s3._save_pyramid_parallel"),
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
        ):
            save_pyramid_to_s3(
                _pyramid(levels=(0,)), "s3://bucket/test", s3_options={}, mode="w"
            )
        # _inspect_store is called inside _save_pyramid_parallel, not here
        mock_inspect.assert_not_called()


# ===================================================================
# _build_write_delayed
# ===================================================================


class TestBuildWriteDelayed:
    """Tests for ``_build_write_delayed``.

    ``xr.Dataset.to_zarr`` is patched to return a sentinel value and record
    call kwargs, so tests can assert on what arguments were passed without
    touching S3 or Dask.
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
        encoding: dict[str, Any] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Call _build_write_delayed and return (return_value, list_of_to_zarr_kwargs)."""
        store = mock.MagicMock()
        calls: list[dict[str, Any]] = []
        sentinel = object()

        def _fake_to_zarr(self_ds: xr.Dataset, store_arg: Any, **kwargs: Any) -> Any:
            calls.append({"ds": self_ds, **kwargs})
            return sentinel

        with mock.patch.object(xr.Dataset, "to_zarr", _fake_to_zarr):
            result = _build_write_delayed(
                ds,
                store,
                plan,
                zarr_format=zarr_format,
                encoding=encoding,
            )
        return result, calls

    def test_mode_w_returns_delayed(self) -> None:
        ds = _healpix_ds()
        result, calls = self._run(ds, self._plan(mode="w"))
        assert result is not None
        assert len(calls) == 1
        assert calls[0]["mode"] == "w"

    def test_mode_r_plus_returns_delayed(self) -> None:
        ds = _healpix_ds()
        result, calls = self._run(ds, self._plan(mode="r+"))
        assert result is not None
        assert calls[0]["mode"] == "r+"

    def test_always_uses_compute_false(self) -> None:
        ds = _healpix_ds()
        _, calls = self._run(ds, self._plan(mode="w"))
        assert calls[0]["compute"] is False

    def test_always_uses_consolidated_false(self) -> None:
        """Consolidated metadata must be written manually after gather."""
        ds = _healpix_ds()
        _, calls = self._run(ds, self._plan(mode="w"))
        assert calls[0]["consolidated"] is False

    def test_new_vars_only_single_append_call(self) -> None:
        ds = _healpix_ds(vars=["tas", "pr"], n_time=3)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=False,
        )
        result, calls = self._run(ds, plan)
        assert result is not None
        assert len(calls) == 1
        assert calls[0]["mode"] == "a"
        assert "append_dim" not in calls[0]

    def test_new_vars_sliced_to_existing_time(self) -> None:
        """New variable write must cover only existing time range."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=False,
        )
        _, calls = self._run(ds, plan)
        assert calls[0]["ds"].sizes["time"] == 3

    def test_append_time_only_uses_append_dim(self) -> None:
        ds = _healpix_ds(vars=["tas"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        result, calls = self._run(ds, plan)
        assert result is not None
        assert calls[0].get("append_dim") == "time"

    def test_append_time_sliced_to_new_steps(self) -> None:
        """Append write must contain only the new time steps."""
        ds = _healpix_ds(vars=["tas"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=[],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        _, calls = self._run(ds, plan)
        assert calls[0]["ds"].sizes["time"] == 2

    def test_new_vars_and_append_time_returns_none(self) -> None:
        """Two-step append must return None to trigger sequential fallback."""
        ds = _healpix_ds(vars=["tas", "pr"], n_time=5)
        plan = self._plan(
            mode="a",
            new_vars=["pr"],
            existing_vars=["tas"],
            n_existing_time=3,
            append_time=True,
        )
        result, calls = self._run(ds, plan)
        assert result is None
        assert calls == []  # to_zarr must not be called

    def test_encoding_forwarded(self) -> None:
        ds = _healpix_ds()
        enc = {"tas": {"dtype": "float32"}}
        _, calls = self._run(ds, self._plan(mode="w"), encoding=enc)
        assert calls[0]["encoding"] == enc

    def test_zarr_format_forwarded(self) -> None:
        ds = _healpix_ds()
        _, calls = self._run(ds, self._plan(mode="w"), zarr_format=3)
        assert calls[0]["zarr_format"] == 3


# ===================================================================
# _save_pyramid_parallel
# ===================================================================


class TestSavePyramidParallel:
    """Tests for ``_save_pyramid_parallel``.

    The distributed client and zarr.consolidate_metadata are mocked so
    tests run without a real cluster or S3 connection.
    ``_build_write_delayed`` is also mocked to return a sentinel delayed
    object, isolating the orchestration logic from the write decomposition
    (which is already tested in TestBuildWriteDelayed).
    """

    SENTINEL_DELAYED = object()

    def _run(
        self,
        pyramid: dict[int, xr.Dataset],
        *,
        mode: str = "w",
        zarr_format: int = 2,
        encoding: dict[int, Any] | None = None,
        two_step_levels: set[int] | None = None,
    ) -> tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        """Run _save_pyramid_parallel with mocked client and internals.

        Returns (mock_client, mock_build_delayed, mock_consolidate).
        *two_step_levels* controls which levels return None from
        _build_write_delayed (simulating the two-step append fallback).
        """
        two_step_levels = two_step_levels or set()
        client = mock.MagicMock()
        client.compute.return_value = [mock.MagicMock()]
        client.gather.return_value = None
        fs = mock.MagicMock()

        def _fake_build(ds: Any, store: Any, plan: Any, **kw: Any) -> Any:
            # Identify level from store root path
            root: str = store.root if hasattr(store, "root") else ""
            for lvl in two_step_levels:
                if f"level_{lvl}" in root:
                    return None
            return self.SENTINEL_DELAYED

        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch(
                "grid_doctor.s3._build_write_delayed", side_effect=_fake_build
            ) as mock_build,
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr") as mock_zarr,
        ):
            mock_s3map.return_value = mock.MagicMock()
            mock_s3map.return_value.root = "s3://bucket/test/level_X.zarr"
            # Make S3Map return a store whose root contains the level number
            def _s3map_factory(*a: Any, **kw: Any) -> mock.MagicMock:
                m = mock.MagicMock()
                root = kw.get("root", "")
                m.root = root
                return m
            mock_s3map.side_effect = _s3map_factory

            mock_inspect.return_value = WritePlan(
                mode=mode if mode != "auto" else "w",
                new_vars=["tas"],
                existing_vars=[],
                n_existing_time=None,
                append_time=False,
            )

            _save_pyramid_parallel(
                pyramid,
                fs,
                "bucket/test",
                mode=mode,
                zarr_format=zarr_format,
                encoding=encoding,
                validate=False,
                max_retries=3,
                retry_backoff=2.0,
                client=client,
            )

        return client, mock_build, mock_zarr

    def test_client_compute_called(self) -> None:
        pyramid = _pyramid(levels=(0, 1))
        client, _, _ = self._run(pyramid)
        assert client.compute.called

    def test_client_gather_called_after_compute(self) -> None:
        pyramid = _pyramid(levels=(0, 1))
        client, _, _ = self._run(pyramid)
        assert client.gather.called
        # gather must be called with the result of compute
        gather_arg = client.gather.call_args[0][0]
        assert gather_arg is client.compute.return_value

    def test_all_parallelisable_levels_submitted(self) -> None:
        """All non-two-step levels must appear in client.compute call."""
        pyramid = _pyramid(levels=(0, 1, 2))
        client, _, _ = self._run(pyramid)
        delayed_list = client.compute.call_args[0][0]
        assert len(delayed_list) == 3

    def test_max_retries_forwarded_to_client_compute(self) -> None:
        pyramid = _pyramid(levels=(0,))
        client, _, _ = self._run(pyramid)
        kwargs = client.compute.call_args[1]
        assert kwargs.get("retries") == 3

    def test_two_step_level_not_in_parallel_batch(self) -> None:
        """Level with new_vars AND append_time must not be passed to client.compute."""
        pyramid = _pyramid(levels=(0, 1))
        client, _, _ = self._run(pyramid, two_step_levels={1})
        delayed_list = client.compute.call_args[0][0]
        # Only level 0 is parallelisable
        assert len(delayed_list) == 1

    def test_two_step_level_executed_sequentially(self) -> None:
        """Level falling back to sequential must go through _execute_write_plan."""
        pyramid = _pyramid(levels=(0, 1))
        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch(
                "grid_doctor.s3._build_write_delayed",
                side_effect=lambda ds, store, plan, **kw: (
                    None if "level_1" in store.root else object()
                ),
            ),
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr"),
        ):
            def _s3map_factory(*a: Any, **kw: Any) -> mock.MagicMock:
                m = mock.MagicMock()
                m.root = kw.get("root", "")
                return m
            mock_s3map.side_effect = _s3map_factory
            client = mock.MagicMock()
            client.compute.return_value = [mock.MagicMock()]
            mock_inspect.return_value = WritePlan(
                mode="w", new_vars=["tas"], existing_vars=[],
                n_existing_time=None, append_time=False,
            )
            _save_pyramid_parallel(
                pyramid, mock.MagicMock(), "bucket/test",
                mode="w", zarr_format=2, encoding=None,
                validate=False, max_retries=3, retry_backoff=2.0, client=client,
            )
        assert mock_execute.call_count == 1

    def test_consolidate_metadata_called_for_v2(self) -> None:
        """zarr.consolidate_metadata must be called once per parallelised store for v2."""
        pyramid = _pyramid(levels=(0, 1))
        _, _, mock_zarr = self._run(pyramid, zarr_format=2)
        assert mock_zarr.consolidate_metadata.call_count == 2

    def test_consolidate_metadata_not_called_for_v3(self) -> None:
        pyramid = _pyramid(levels=(0, 1))
        _, _, mock_zarr = self._run(pyramid, zarr_format=3)
        mock_zarr.consolidate_metadata.assert_not_called()

    def test_inspect_store_called_for_auto_mode(self) -> None:
        pyramid = _pyramid(levels=(0,))
        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._build_write_delayed", return_value=object()),
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr"),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root",""))
            mock_inspect.return_value = WritePlan(
                mode="w", new_vars=["tas"], existing_vars=[],
                n_existing_time=None, append_time=False,
            )
            client = mock.MagicMock()
            client.compute.return_value = [mock.MagicMock()]
            _save_pyramid_parallel(
                pyramid, mock.MagicMock(), "bucket/test",
                mode="auto", zarr_format=2, encoding=None,
                validate=False, max_retries=3, retry_backoff=2.0, client=client,
            )
        assert mock_inspect.called

    def test_gpu_backed_dataset_routed_to_sequential(self) -> None:
        """GPU-backed levels must bypass _build_write_delayed and go to
        _execute_write_plan.  Workers lack GPU context and would OOM on
        multi-TB datasets if they recomputed CuPy tasks."""
        # Dataset with grid_doctor_backend='cupy' attr — triggers GPU routing
        gpu_ds = _healpix_ds(vars=["tas"])
        gpu_ds.attrs["grid_doctor_backend"] = "cupy"
        pyramid = {0: gpu_ds}

        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._build_write_delayed") as mock_build,
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr"),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root", ""))
            mock_inspect.return_value = WritePlan(
                mode="w", new_vars=["tas"], existing_vars=[],
                n_existing_time=None, append_time=False,
            )
            client = mock.MagicMock()
            client.compute.return_value = [mock.MagicMock()]
            _save_pyramid_parallel(
                pyramid, mock.MagicMock(), "bucket/test",
                mode="auto", zarr_format=2, encoding=None,
                validate=False, max_retries=3, retry_backoff=2.0, client=client,
            )

        # GPU level must go to _execute_write_plan, never to _build_write_delayed
        mock_build.assert_not_called()
        assert mock_execute.call_count == 1
        # client.compute must not be called for GPU-only pyramid
        client.compute.assert_not_called()

    def test_cpu_backed_dataset_routed_to_parallel(self) -> None:
        """CPU-backed levels must go through _build_write_delayed and client.compute."""
        import dask.array as da
        pyramid = _pyramid(levels=(0,))

        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._build_write_delayed", return_value=object()) as mock_build,
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr"),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root", ""))
            mock_inspect.return_value = WritePlan(
                mode="w", new_vars=["tas"], existing_vars=[],
                n_existing_time=None, append_time=False,
            )
            client = mock.MagicMock()
            client.compute.return_value = [mock.MagicMock()]
            _save_pyramid_parallel(
                pyramid, mock.MagicMock(), "bucket/test",
                mode="auto", zarr_format=2, encoding=None,
                validate=False, max_retries=3, retry_backoff=2.0, client=client,
            )

        mock_build.assert_called_once()
        mock_execute.assert_not_called()
        client.compute.assert_called_once()

    def test_inspect_store_not_called_for_explicit_mode(self) -> None:
        pyramid = _pyramid(levels=(0,))
        with (
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.s3._build_write_delayed", return_value=object()),
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
            mock.patch("grid_doctor.s3.zarr"),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root",""))
            client = mock.MagicMock()
            client.compute.return_value = [mock.MagicMock()]
            _save_pyramid_parallel(
                pyramid, mock.MagicMock(), "bucket/test",
                mode="w", zarr_format=2, encoding=None,
                validate=False, max_retries=3, retry_backoff=2.0, client=client,
            )
        mock_inspect.assert_not_called()


# ===================================================================
# save_pyramid_to_s3 — client parameter
# ===================================================================


class TestGetOrCreateClient:
    """Tests for the ``_get_or_create_client`` context manager."""

    def test_yields_provided_client_unchanged(self) -> None:
        """A caller-supplied client must be yielded as-is without closing."""
        client = mock.MagicMock()
        with _get_or_create_client(client, n_workers=4, threads_per_worker=2) as c:
            assert c is client
        client.close.assert_not_called()

    def test_reuses_existing_distributed_client(self) -> None:
        """If a distributed client already exists in context, reuse it."""
        existing = mock.MagicMock()
        with mock.patch("grid_doctor.s3._get_distributed_client", return_value=existing):
            with _get_or_create_client(
                None, n_workers=4, threads_per_worker=2
            ) as c:
                assert c is existing
        existing.close.assert_not_called()

    def test_creates_local_cluster_when_none_exists(self) -> None:
        """No client + no existing cluster → create LocalCluster and Client."""
        mock_cluster = mock.MagicMock()
        mock_client = mock.MagicMock()
        with (
            mock.patch(
                "grid_doctor.s3._get_distributed_client",
                side_effect=ValueError("no client"),
            ),
            mock.patch("grid_doctor.s3.LocalCluster", return_value=mock_cluster),
            mock.patch("grid_doctor.s3.Client", return_value=mock_client),
        ):
            with _get_or_create_client(
                None, n_workers=4, threads_per_worker=2
            ) as c:
                assert c is mock_client
            mock_client.close.assert_called_once()
            mock_cluster.close.assert_called_once()

    def test_cluster_closed_even_on_exception(self) -> None:
        """LocalCluster must be closed even when the body raises."""
        mock_cluster = mock.MagicMock()
        mock_client = mock.MagicMock()
        with (
            mock.patch(
                "grid_doctor.s3._get_distributed_client",
                side_effect=ValueError("no client"),
            ),
            mock.patch("grid_doctor.s3.LocalCluster", return_value=mock_cluster),
            mock.patch("grid_doctor.s3.Client", return_value=mock_client),
        ):
            with pytest.raises(RuntimeError):
                with _get_or_create_client(
                    None, n_workers=4, threads_per_worker=2
                ):
                    raise RuntimeError("upload failed")
            mock_client.close.assert_called_once()
            mock_cluster.close.assert_called_once()

    def test_no_existing_client_creates_cluster(self) -> None:
        """When no existing client exists, a LocalCluster must be created."""
        with (
            mock.patch(
                "grid_doctor.s3._get_distributed_client",
                side_effect=ValueError("no client"),
            ),
            mock.patch("grid_doctor.s3.LocalCluster") as mock_lc,
            mock.patch("grid_doctor.s3.Client") as mock_cl,
        ):
            mock_lc.return_value = mock.MagicMock()
            mock_cl.return_value = mock.MagicMock()
            with _get_or_create_client(None, n_workers=4, threads_per_worker=2):
                pass
        assert mock_lc.called
        assert mock_cl.called

    def test_n_workers_forwarded_to_local_cluster(self) -> None:
        mock_cluster = mock.MagicMock()
        with (
            mock.patch(
                "grid_doctor.s3._get_distributed_client",
                side_effect=ValueError("no client"),
            ),
            mock.patch(
                "grid_doctor.s3.LocalCluster", return_value=mock_cluster
            ) as mock_lc,
            mock.patch("grid_doctor.s3.Client"),
        ):
            with _get_or_create_client(None, n_workers=8, threads_per_worker=3):
                pass
        _, kwargs = mock_lc.call_args
        assert kwargs["n_workers"] == 8
        assert kwargs["threads_per_worker"] == 3


class TestSavePyramidToS3ClientParam:
    """Tests that ``save_pyramid_to_s3`` correctly routes through
    ``_get_or_create_client`` and into the parallel or sequential path."""

    def _mock_context(
        self, active_client: Any | None
    ) -> mock.MagicMock:
        """Return a mock that makes _get_or_create_client yield active_client."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            yield active_client

        return mock.patch("grid_doctor.s3._get_or_create_client", side_effect=_ctx)

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3._save_pyramid_parallel")
    def test_parallel_path_used_when_client_available(
        self, mock_parallel: mock.Mock, mock_fs: mock.Mock
    ) -> None:
        client = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        with self._mock_context(client):
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
        assert mock_parallel.called

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3._save_pyramid_parallel")
    def test_sequential_fallback_when_no_distributed(
        self, mock_parallel: mock.Mock, mock_fs: mock.Mock
    ) -> None:
        """When context manager yields None, sequential path must be used."""
        pyramid = _pyramid(levels=(0,))
        with (
            self._mock_context(None),
            mock.patch.object(xr.Dataset, "to_zarr"),
        ):
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
        mock_parallel.assert_not_called()

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3._save_pyramid_parallel")
    def test_s3_path_normalised_before_parallel(
        self, mock_parallel: mock.Mock, mock_fs: mock.Mock
    ) -> None:
        client = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        with self._mock_context(client):
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test/", s3_options={}, mode="w"
            )
        passed_path = mock_parallel.call_args[0][2]
        assert not passed_path.startswith("s3://")
        assert not passed_path.endswith("/")

    @mock.patch("grid_doctor.s3.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.s3._save_pyramid_parallel")
    def test_all_params_forwarded_to_parallel(
        self, mock_parallel: mock.Mock, mock_fs: mock.Mock
    ) -> None:
        client = mock.MagicMock()
        pyramid = _pyramid(levels=(0,))
        enc = {0: {"tas": {"dtype": "float32"}}}
        with self._mock_context(client):
            save_pyramid_to_s3(
                pyramid,
                "s3://bucket/test",
                s3_options={},
                mode="auto",
                zarr_format=2,
                encoding=enc,
                validate=True,
                max_retries=7,
                retry_backoff=3.0,
            )
        kw = mock_parallel.call_args[1]
        assert kw["mode"] == "auto"
        assert kw["zarr_format"] == 2
        assert kw["encoding"] == enc
        assert kw["validate"] is True
        assert kw["max_retries"] == 7
        assert kw["retry_backoff"] == 3.0
