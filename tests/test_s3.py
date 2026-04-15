"""Tests for ``grid_doctor.s3``.

Covers:
- ``get_s3_options``        — credential loading and S3 timeout configuration
- ``_with_retry``           — exponential-backoff retry logic
- ``_inspect_store``        — store introspection and ``WritePlan`` production
- ``_execute_write_plan``   — two-step write decomposition
- ``save_pyramid_to_s3``    — public API, explicit modes and ``mode="auto"``
- ``create_and_upload_healpix_pyramid`` — combined remap + level-by-level upload
"""

from __future__ import annotations

import json
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
    _save_pyramid_sequential,
    _with_retry,
    create_and_upload_healpix_pyramid,
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

    Since save_pyramid_to_s3 now always routes through _save_pyramid_sequential,
    these tests verify that the correct arguments are forwarded.
    """

    def _run(
        self,
        pyramid: dict[int, xr.Dataset],
        *,
        mode: str = "w",
        **kwargs: Any,
    ) -> mock.MagicMock:
        """Call save_pyramid_to_s3 and return the mock for _save_pyramid_sequential."""
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._save_pyramid_sequential") as mock_seq,
        ):
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, mode=mode, **kwargs
            )
        return mock_seq

    def test_sequential_called_for_explicit_mode(self) -> None:
        mock_seq = self._run(_pyramid(levels=(0, 1)), mode="w")
        assert mock_seq.called

    def test_mode_forwarded_to_sequential(self) -> None:
        mock_seq = self._run(_pyramid(levels=(0,)), mode="w")
        assert mock_seq.call_args[1]["mode"] == "w"

    def test_s3_path_normalised(self) -> None:
        """s3:// prefix and trailing slash must be stripped before handoff."""
        mock_seq = self._run(_pyramid(levels=(0,)))
        passed_path = mock_seq.call_args[0][2]
        assert not passed_path.startswith("s3://")
        assert not passed_path.endswith("/")

    def test_zarr_format_2_forwarded(self) -> None:
        mock_seq = self._run(_pyramid(levels=(0,)), zarr_format=2)
        assert mock_seq.call_args[1]["zarr_format"] == 2

    def test_zarr_format_3_forwarded(self) -> None:
        mock_seq = self._run(_pyramid(levels=(0,)), zarr_format=3)
        assert mock_seq.call_args[1]["zarr_format"] == 3

    def test_per_level_encoding_forwarded(self) -> None:
        enc = {0: {"tas": {"dtype": "float32"}}}
        mock_seq = self._run(_pyramid(levels=(0,)), encoding=enc)
        assert mock_seq.call_args[1]["encoding"] == enc

    def test_max_retries_forwarded(self) -> None:
        mock_seq = self._run(_pyramid(levels=(0,)), max_retries=7)
        assert mock_seq.call_args[1]["max_retries"] == 7


# ===================================================================
# save_pyramid_to_s3 — mode="auto"
# ===================================================================


class TestSavePyramidToS3Auto:
    """Tests for save_pyramid_to_s3 with mode='auto'.

    With the parallel-first design, 'auto' mode is forwarded to
    _save_pyramid_sequential.  Inspect logic and plan routing are tested
    in integration via _run_auto.
    """

    def _run_auto(
        self,
        pyramid: dict[int, xr.Dataset],
        *,
        validate: bool = False,
    ) -> mock.MagicMock:
        """Call save_pyramid_to_s3(mode='auto') and return the _save_pyramid_sequential mock."""
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._save_pyramid_sequential") as mock_seq,
        ):
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={},
                mode="auto", validate=validate,
            )
        return mock_seq

    def test_auto_mode_forwarded_to_sequential(self) -> None:
        mock_seq = self._run_auto(_pyramid(levels=(0,)))
        assert mock_seq.call_args[1]["mode"] == "auto"

    def test_validate_forwarded_to_parallel(self) -> None:
        mock_seq = self._run_auto(_pyramid(levels=(0,)), validate=True)
        assert mock_seq.call_args[1]["validate"] is True

    def test_sequential_called_once_with_full_pyramid(self) -> None:
        """_save_pyramid_sequential receives the entire pyramid in one call."""
        pyramid = _pyramid(levels=(0, 1, 2))
        mock_seq = self._run_auto(pyramid)
        assert mock_seq.call_count == 1
        assert mock_seq.call_args[0][0] is pyramid

    def test_explicit_mode_bypasses_inspect(self) -> None:
        """_inspect_store must not be called by save_pyramid_to_s3 directly."""
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3._save_pyramid_sequential"),
            mock.patch("grid_doctor.s3._inspect_store") as mock_inspect,
        ):
            save_pyramid_to_s3(
                _pyramid(levels=(0,)), "s3://bucket/test", s3_options={}, mode="w"
            )
        # _inspect_store is delegated to _save_pyramid_sequential, not called here
        mock_inspect.assert_not_called()



# ===================================================================
# create_and_upload_healpix_pyramid
# ===================================================================


class TestCreateAndUploadHealpixPyramid:
    """Tests for ``create_and_upload_healpix_pyramid``.

    The key design under test is the level-by-level pipeline:
    remap → upload → read-back-from-S3 → coarsen → upload → ...

    Reading back from S3 severs the upstream Dask graph so CuPy tasks
    are never re-executed by downstream coarsening steps.
    """

    def _run(
        self,
        levels: tuple[int, ...] = (2, 1, 0),
        backend: str = "scipy",
        **kwargs: Any,
    ) -> tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        """Run create_and_upload_healpix_pyramid with mocked internals.

        Returns (mock_regrid, mock_coarsen, mock_execute).
        """
        max_level = max(levels)
        min_level = min(levels)
        finest = _healpix_ds(level=max_level)
        finest.attrs["grid_doctor_backend"] = backend

        def _fake_coarsen(ds: xr.Dataset, target_level: int, **kw: Any) -> xr.Dataset:
            result = _healpix_ds(level=target_level)
            result.attrs["grid_doctor_backend"] = backend
            return result

        def _fake_open_zarr(store: Any, **kw: Any) -> xr.Dataset:
            # Return a fresh dataset as if read from S3
            level = int(store.root.split("level_")[1].replace(".zarr", ""))
            ds = _healpix_ds(level=level)
            ds.attrs["grid_doctor_backend"] = backend
            return ds

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.remap.regrid_to_healpix", return_value=finest) as mock_regrid,
            mock.patch("grid_doctor.helpers.coarsen_healpix", side_effect=_fake_coarsen) as mock_coarsen,
            mock.patch("grid_doctor.s3.xr.open_zarr", side_effect=_fake_open_zarr),
            mock.patch("grid_doctor.s3._execute_write_plan") as mock_execute,
            mock.patch("grid_doctor.s3.zarr.consolidate_metadata"),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(
                root=kw.get("root", "s3://bucket/test/level_X.zarr")
            )
            create_and_upload_healpix_pyramid(
                _healpix_ds(),
                "s3://bucket/test",
                s3_options={},
                max_level=max_level,
                min_level=min_level,
                **kwargs,
            )
        return mock_regrid, mock_coarsen, mock_execute

    def test_regrid_called_once_at_max_level(self) -> None:
        mock_regrid, _, _ = self._run(levels=(2, 1, 0))
        assert mock_regrid.call_count == 1
        assert mock_regrid.call_args[0][1] == 2  # max_level positional arg

    def test_coarsen_called_for_each_lower_level(self) -> None:
        """One coarsen call per level below max."""
        _, mock_coarsen, _ = self._run(levels=(2, 1, 0))
        assert mock_coarsen.call_count == 2  # level 2→1, 1→0

    def test_upload_called_for_every_level(self) -> None:
        """_execute_write_plan must be called once per level."""
        _, _, mock_execute = self._run(levels=(2, 1, 0))
        assert mock_execute.call_count == 3

    def test_coarsen_not_called_when_max_equals_min(self) -> None:
        """Single-level pyramid: remap and upload, no coarsening."""
        _, mock_coarsen, mock_execute = self._run(levels=(2,))
        mock_coarsen.assert_not_called()
        assert mock_execute.call_count == 1

    def test_level_paths_contain_level_number(self) -> None:
        """S3Map must be called with the correct level path for each level."""
        max_level, min_level = 2, 0
        finest = _healpix_ds(level=max_level)

        store_roots: list[str] = []

        def _fake_open_zarr(store: Any, **kw: Any) -> xr.Dataset:
            level = int(store.root.split("level_")[1].replace(".zarr", ""))
            return _healpix_ds(level=level)

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.remap.regrid_to_healpix", return_value=finest),
            mock.patch("grid_doctor.helpers.coarsen_healpix", side_effect=lambda ds, lvl, **kw: _healpix_ds(level=lvl)),
            mock.patch("grid_doctor.s3.xr.open_zarr", side_effect=_fake_open_zarr),
            mock.patch("grid_doctor.s3._execute_write_plan"),
            mock.patch("grid_doctor.s3.zarr.consolidate_metadata"),
        ):
            def _s3map_factory(*a: Any, **kw: Any) -> mock.MagicMock:
                m = mock.MagicMock()
                root = kw.get("root", "")
                m.root = root
                store_roots.append(root)
                return m
            mock_s3map.side_effect = _s3map_factory
            create_and_upload_healpix_pyramid(
                _healpix_ds(), "s3://bucket/test", s3_options={},
                max_level=max_level, min_level=min_level,
            )

        assert any("level_2" in r for r in store_roots)
        assert any("level_1" in r for r in store_roots)
        assert any("level_0" in r for r in store_roots)

    def test_gpu_backend_uses_synchronous_scheduler(self) -> None:
        """GPU-backed datasets must trigger synchronous Dask scheduler."""
        import dask
        scheduler_used: list[str] = []

        original_set = dask.config.set

        def _capture_set(**kw: Any) -> Any:
            if "scheduler" in kw:
                scheduler_used.append(kw["scheduler"])
            return original_set(**kw)

        finest = _healpix_ds(level=1)
        finest.attrs["grid_doctor_backend"] = "cupy"

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.remap.regrid_to_healpix", return_value=finest),
            mock.patch("grid_doctor.helpers.coarsen_healpix",
                       side_effect=lambda ds, lvl, **kw: _healpix_ds(level=lvl)),
            mock.patch("grid_doctor.s3.xr.open_zarr",
                       return_value=_healpix_ds(level=0)),
            mock.patch("grid_doctor.s3._execute_write_plan"),
            mock.patch("grid_doctor.s3.zarr.consolidate_metadata"),
            mock.patch("grid_doctor.s3.dask.config.set", side_effect=_capture_set),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root",""))
            create_and_upload_healpix_pyramid(
                _healpix_ds(), "s3://bucket/test", s3_options={},
                max_level=1, min_level=0,
            )

        assert "synchronous" in scheduler_used

    def test_cpu_backend_does_not_force_synchronous(self) -> None:
        """CPU-backed datasets must not force the synchronous scheduler."""
        import dask
        scheduler_used: list[str] = []

        original_set = dask.config.set

        def _capture_set(**kw: Any) -> Any:
            if "scheduler" in kw:
                scheduler_used.append(kw["scheduler"])
            return original_set(**kw)

        finest = _healpix_ds(level=1)
        finest.attrs["grid_doctor_backend"] = "scipy"

        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.remap.regrid_to_healpix", return_value=finest),
            mock.patch("grid_doctor.helpers.coarsen_healpix",
                       side_effect=lambda ds, lvl, **kw: _healpix_ds(level=lvl)),
            mock.patch("grid_doctor.s3.xr.open_zarr",
                       return_value=_healpix_ds(level=0)),
            mock.patch("grid_doctor.s3._execute_write_plan"),
            mock.patch("grid_doctor.s3.zarr.consolidate_metadata"),
            mock.patch("grid_doctor.s3.dask.config.set", side_effect=_capture_set),
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root",""))
            create_and_upload_healpix_pyramid(
                _healpix_ds(), "s3://bucket/test", s3_options={},
                max_level=1, min_level=0,
            )

        assert "synchronous" not in scheduler_used

    def test_kwargs_forwarded_to_regrid(self) -> None:
        """Extra kwargs (method, backend, …) must reach regrid_to_healpix."""
        mock_regrid, _, _ = self._run(levels=(1, 0), method="nearest")
        assert mock_regrid.call_args[1].get("method") == "nearest"

    def test_zarr_v2_consolidates_metadata(self) -> None:
        """zarr.consolidate_metadata must be called for every level with v2."""
        finest = _healpix_ds(level=1)
        with (
            mock.patch("grid_doctor.s3.s3fs.S3FileSystem"),
            mock.patch("grid_doctor.s3.s3fs.S3Map") as mock_s3map,
            mock.patch("grid_doctor.remap.regrid_to_healpix", return_value=finest),
            mock.patch("grid_doctor.helpers.coarsen_healpix",
                       side_effect=lambda ds, lvl, **kw: _healpix_ds(level=lvl)),
            mock.patch("grid_doctor.s3.xr.open_zarr",
                       return_value=_healpix_ds(level=0)),
            mock.patch("grid_doctor.s3._execute_write_plan"),
            mock.patch("grid_doctor.s3.zarr.consolidate_metadata") as mock_cons,
        ):
            mock_s3map.side_effect = lambda *a, **kw: mock.MagicMock(root=kw.get("root",""))
            create_and_upload_healpix_pyramid(
                _healpix_ds(), "s3://bucket/test", s3_options={},
                max_level=1, min_level=0, zarr_format=2,
            )
        assert mock_cons.call_count == 2  # one per level
