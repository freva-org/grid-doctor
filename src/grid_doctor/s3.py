"""S3 upload helpers for HEALPix pyramids.

This module owns everything related to writing pyramid datasets to
S3-backed Zarr stores:

- :func:`get_s3_options` — build an authenticated, timeout-hardened
  :class:`s3fs.S3FileSystem` configuration dictionary.
- :func:`save_pyramid_to_s3` — write a pyramid (``dict[int, xr.Dataset]``)
  level-by-level, with automatic mode inference, retry logic, and dirty-store
  detection.

The three private helpers drive the ``mode="auto"`` path:

- :func:`_inspect_store` — reads ``.zmetadata`` and returns a
  :class:`~grid_doctor.types.WritePlan`.
- :func:`_execute_write_plan` — executes the plan as one or two
  ``to_zarr`` calls.
- :func:`_with_retry` — wraps any zero-argument callable with exponential
  backoff.
"""

from __future__ import annotations

import json
import logging
import math
import time as _time
from pathlib import Path
from typing import Any, Callable, Literal

import s3fs
import xarray as xr

from .types import OnDirty, WriteMode, WritePlan

logger = logging.getLogger(__name__)


# ===================================================================
# S3 filesystem configuration
# ===================================================================


def get_s3_options(
    endpoint_url: str,
    secrets_file: str | Path,
    *,
    read_timeout: int = 300,
    connect_timeout: int = 60,
    max_attempts: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an S3 options dictionary from an endpoint and credentials file.

    The returned dictionary is suitable for passing directly to
    :class:`s3fs.S3FileSystem`.  Sensible timeout and retry defaults are
    included to guard against transient network issues on the DKRZ S3
    instance; all can be overridden via *kwargs*.

    Parameters
    ----------
    endpoint_url:
        S3-compatible endpoint URL.
    secrets_file:
        JSON file containing ``accessKey`` and ``secretKey``.
    read_timeout:
        Seconds to wait for a response from the server after the request has
        been sent.  The default (300 s) is generous enough for slow S3
        acknowledgements under load.
    connect_timeout:
        Seconds to wait for the initial TCP connection to be established.
    max_attempts:
        Maximum number of retry attempts made by the AWS SDK for each
        individual S3 call (``adaptive`` mode backs off on throttling and
        timeouts automatically).
    **kwargs:
        Additional options merged into the returned dictionary, taking
        precedence over the defaults.

    Returns
    -------
    dict[str, Any]
        Options for :class:`s3fs.S3FileSystem`.
    """
    try:
        import botocore.config as _bc_config

        _boto_config = _bc_config.Config(
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            retries={"max_attempts": max_attempts, "mode": "adaptive"},
        )
        client_kwargs: dict[str, Any] = {"config": _boto_config}
    except ImportError:
        logger.warning(
            "botocore not available — S3 timeout and retry config will not be "
            "applied. Install aiobotocore or botocore to enable these defaults."
        )
        client_kwargs = {}

    secrets: dict[str, str] = json.loads(Path(secrets_file).read_text())
    options: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "secret": secrets["secretKey"],
        "key": secrets["accessKey"],
        "client_kwargs": client_kwargs,
    }
    # Allow callers to override or extend client_kwargs selectively.
    extra_client_kwargs = kwargs.pop("client_kwargs", {})
    options["client_kwargs"].update(extra_client_kwargs)
    options.update(kwargs)
    return options


# ===================================================================
# Internal helpers
# ===================================================================


def _with_retry(
    fn: Callable[[], None],
    *,
    max_retries: int,
    backoff: float,
    base_delay: float = 2.0,
) -> None:
    """Call *fn*, retrying up to *max_retries* times with exponential backoff.

    All exceptions are caught and retried.  After *max_retries* failed
    attempts the final exception is re-raised.

    Args:
        fn: Zero-argument callable to invoke.
        max_retries: Maximum number of additional attempts after the first.
        backoff: Multiplicative factor applied to the delay between attempts.
        base_delay: Initial delay in seconds before the second attempt.
    """
    for attempt in range(max_retries + 1):
        try:
            fn()
            return
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (backoff**attempt)
            logger.warning(
                "Upload attempt %d/%d failed (%s: %s). Retrying in %.1f s.",
                attempt + 1,
                max_retries + 1,
                type(exc).__name__,
                exc,
                delay,
            )
            _time.sleep(delay)


def _inspect_store(
    fs: s3fs.S3FileSystem,
    level_path: str,
    ds: xr.Dataset,
    *,
    validate: bool = False,
) -> WritePlan:
    """Inspect an existing Zarr v2 store and produce a :class:`WritePlan`.

    The inspection reads only consolidated metadata (``.zmetadata``) — a
    single cheap S3 call.  When *validate* is ``True``, an additional
    existence check is performed for the last chunk along the time dimension
    of each data variable already present in the store.

    Args:
        fs: Authenticated S3 filesystem.
        level_path: Full S3 path to the ``.zarr`` store (no trailing slash).
        ds: Incoming dataset that we intend to write.
        validate: When ``True``, verify that the last time-chunk of every
            existing variable is present.  Sets :attr:`WritePlan.dirty` to
            ``True`` if any chunk is missing.

    Returns:
        A :class:`WritePlan` describing the required write operations.
    """
    meta_key = f"{level_path}/.zmetadata"

    # --- Does the store have any content at all? ----------------------------
    store_non_empty = fs.exists(level_path) and bool(fs.ls(level_path, detail=False))
    if not store_non_empty:
        return WritePlan(
            mode="w",
            new_vars=list(map(str, ds.data_vars)),
            existing_vars=[],
            n_existing_time=None,
            append_time=False,
            dirty=False,
        )

    # --- Store exists but consolidated metadata is absent → dirty -----------
    if not fs.exists(meta_key):
        logger.warning(
            "Store at %s has content but no .zmetadata — treating as dirty.",
            level_path,
        )
        return WritePlan(
            mode="w",
            new_vars=list(map(str, ds.data_vars)),
            existing_vars=[],
            n_existing_time=None,
            append_time=False,
            dirty=True,
        )

    # --- Read consolidated metadata (single S3 GET) -------------------------
    with fs.open(meta_key, "r") as fh:
        zmeta: dict[str, Any] = json.load(fh)
    metadata: dict[str, Any] = zmeta.get("metadata", {})

    # Extract names of arrays already in the store.
    store_arrays = {
        key.split("/")[0] for key in metadata if key.endswith("/.zarray") and "/" in key
    }

    incoming_data_vars = set(ds.data_vars)
    new_vars = [v for v in incoming_data_vars if v not in store_arrays]
    existing_vars = [v for v in incoming_data_vars if v in store_arrays]

    # --- Determine existing time length -------------------------------------
    n_existing_time: int | None = None
    if "time/.zarray" in metadata:
        time_meta = metadata["time/.zarray"]
        if isinstance(time_meta, str):
            time_meta = json.loads(time_meta)
        shape = time_meta.get("shape", [])
        if shape:
            n_existing_time = int(shape[0])

    incoming_time = ds.sizes.get("time")
    append_time = (
        incoming_time is not None
        and n_existing_time is not None
        and incoming_time > n_existing_time
    )

    # --- Validation: check last chunk per existing variable -----------------
    dirty = False
    if validate and existing_vars and n_existing_time is not None:
        for var in existing_vars:
            var_meta_key = f"{var}/.zarray"
            if var_meta_key not in metadata:
                continue
            var_meta = metadata[var_meta_key]
            if isinstance(var_meta, str):
                var_meta = json.loads(var_meta)

            shape = var_meta.get("shape", [])
            chunks: list[int] = var_meta.get("chunks", [])
            if not shape or not chunks or len(shape) != len(chunks):
                continue

            # Check that the last chunk along the time dimension (dim 0)
            # exists.  We use index 0 for all other dimensions to keep the
            # check cheap: one HEAD request per variable, not per chunk.
            n_time_chunks = math.ceil(shape[0] / chunks[0])
            chunk_indices = [0] * len(shape)
            chunk_indices[0] = n_time_chunks - 1
            chunk_key = f"{level_path}/{var}/" + ".".join(str(i) for i in chunk_indices)
            if not fs.exists(chunk_key):
                logger.warning(
                    "Expected chunk %s is missing — store may be incomplete.",
                    chunk_key,
                )
                dirty = True
                break  # one missing chunk is enough evidence

    # --- Resolve base write mode --------------------------------------------
    if dirty:
        resolved_mode: Literal["w", "a", "r+"] = "w"  # overridden by on_dirty
    elif not new_vars and not append_time:
        resolved_mode = "r+"  # pure idempotent retry
    else:
        resolved_mode = "a"

    return WritePlan(
        mode=resolved_mode,
        new_vars=list(map(str, new_vars)),
        existing_vars=list(map(str, existing_vars)),
        n_existing_time=n_existing_time,
        append_time=append_time,
        dirty=dirty,
    )


def _execute_write_plan(
    dataset: xr.Dataset,
    store: s3fs.S3Map,
    plan: WritePlan,
    *,
    zarr_format: Literal[2, 3],
    compute: bool,
    max_retries: int,
    retry_backoff: float,
    encoding: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Execute a :class:`WritePlan` against *store*, with per-call retries.

    The write is decomposed into at most two ``to_zarr`` calls:

    1. **Add new variables** (for the time range already in the store).
    2. **Append new time steps** (for all variables, including any just added
       in step 1).

    Each call is wrapped in :func:`_with_retry` independently so a transient
    S3 timeout in step 2 does not force a repeat of step 1.

    Args:
        dataset: Incoming dataset.
        store: Target Zarr store.
        plan: Write plan produced by :func:`_inspect_store`.
        zarr_format: Zarr format version (2 or 3).
        compute: Whether to trigger Dask computation immediately.
        max_retries: Maximum retry attempts per ``to_zarr`` call.
        retry_backoff: Exponential backoff multiplier between retries.
        encoding: Variable-level encoding options forwarded to ``to_zarr``.
    """
    base_opts: dict[str, Any] = dict(compute=compute, zarr_format=zarr_format)
    if zarr_format == 2:
        base_opts["consolidated"] = True
    if encoding is not None:
        base_opts["encoding"] = encoding

    def _zarr_write(ds_slice: xr.Dataset, **extra: Any) -> None:
        opts = {**base_opts, **extra}
        _with_retry(
            lambda: ds_slice.to_zarr(store, **opts),
            max_retries=max_retries,
            backoff=retry_backoff,
        )

    mode = plan.mode

    # ------------------------------------------------------------------
    # "w" or "r+" — single call covers everything
    # ------------------------------------------------------------------
    if mode in {"w", "r+"}:
        _zarr_write(dataset, mode=mode)
        return

    # ------------------------------------------------------------------
    # "a" — may require two calls
    # ------------------------------------------------------------------

    # Step 1: write new variables for the time range already in the store.
    if plan.new_vars:
        new_var_ds = dataset[plan.new_vars]
        if plan.n_existing_time is not None:
            new_var_ds = new_var_ds.isel(time=slice(0, plan.n_existing_time))
        logger.info("Adding new variable(s) %s to existing store.", plan.new_vars)
        _zarr_write(new_var_ds, mode="a")

    # Step 2: append new time steps across ALL variables.
    if plan.append_time and plan.n_existing_time is not None:
        append_ds = dataset.isel(time=slice(plan.n_existing_time, None))
        logger.info(
            "Appending %d new time step(s) to existing store.",
            dataset.sizes.get("time", 0) - plan.n_existing_time,
        )
        _zarr_write(append_ds, mode="a", append_dim="time")


# ===================================================================
# Public API
# ===================================================================


def save_pyramid_to_s3(
    pyramid: dict[int, xr.Dataset],
    s3_path: str,
    s3_options: dict[str, Any],
    *,
    mode: WriteMode = "a",
    compute: bool = True,
    region: Literal["auto"] | dict[str, slice] = "auto",
    zarr_format: Literal[2, 3] = 2,
    encoding: dict[int, dict[str, dict[str, Any]]] | None = None,
    validate: bool = False,
    on_dirty: OnDirty = "raise",
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> None:
    """Write a HEALPix pyramid to S3-backed Zarr stores.

    Each level is stored below ``"<s3_path>/level_<level>.zarr"``.

    When *mode* is ``"auto"`` the function inspects each level's existing
    store and determines the minimal set of operations required:

    - **Store absent** → full write (``"w"``).
    - **Same schema, same time** → idempotent overwrite of data (``"r+"``);
      safe to call after a timeout mid-write.
    - **New variables only** → append new arrays for the existing time range.
    - **New time steps only** → append along the time dimension.
    - **Both** → add new variables first (for existing time), then append the
      new time steps across all variables.

    Every ``to_zarr`` call is individually wrapped in retry logic with
    exponential backoff so that a transient S3 timeout does not abort hours
    of upstream computation.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset.
    s3_path:
        S3 prefix such as ``"s3://bucket/pyramid"``.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.  Use
        :func:`get_s3_options` to build this with sensible timeout defaults.
    mode:
        ``"w"`` — overwrite entirely.
        ``"a"`` — append / add.
        ``"r+"`` — overwrite data only (store must already exist).
        ``"auto"`` — inspect each store and infer the correct operations.
    compute:
        Trigger Dask execution immediately when ``True``.
    region:
        Region writes for partial updates (ignored when *mode* is ``"auto"``).
    zarr_format:
        Zarr format version (2 or 3).
    encoding:
        Per-level encoding dictionaries, keyed by level number.
    validate:
        When ``True`` and *mode* is ``"auto"``, verify that the last
        time-chunk of each existing variable is present on S3 before trusting
        the store's state.  Adds one :meth:`s3fs.S3FileSystem.exists` call
        per data variable.
    on_dirty:
        Policy applied when a store is found to be in an inconsistent state
        (missing metadata or chunks).  ``"raise"`` aborts with a
        :class:`RuntimeError`; ``"overwrite"`` deletes the store and rewrites
        from scratch; ``"skip"`` leaves the level untouched.
    max_retries:
        Maximum number of retry attempts per individual ``to_zarr`` call.
    retry_backoff:
        Multiplicative factor for the exponential back-off between retries.
        With the default of ``2.0`` and a 2 s base delay, the waits are
        2 s, 4 s, 8 s, … up to *max_retries* attempts.
    """
    fs = s3fs.S3FileSystem(**s3_options)

    for level, dataset in pyramid.items():
        level_path = f"{s3_path}/level_{level}.zarr"
        logger.info("Writing HEALPix level %s to %s", level, level_path)
        store = s3fs.S3Map(root=level_path, s3=fs)
        level_encoding = encoding[level] if encoding is not None else None

        # ------------------------------------------------------------------
        # Resolve write plan
        # ------------------------------------------------------------------
        if mode == "auto":
            plan = _inspect_store(fs, level_path, dataset, validate=validate)

            if plan.dirty:
                if on_dirty == "raise":
                    raise RuntimeError(
                        f"Store at {level_path} is in an inconsistent state "
                        f"(missing metadata or chunks). Pass on_dirty='overwrite' "
                        f"to force a full rewrite, or on_dirty='skip' to ignore "
                        f"this level."
                    )
                if on_dirty == "skip":
                    logger.warning(
                        "Skipping dirty store at %s (on_dirty='skip').", level_path
                    )
                    continue
                # on_dirty == "overwrite"
                logger.warning(
                    "Store at %s is dirty — overwriting (on_dirty='overwrite').",
                    level_path,
                )
                plan = WritePlan(
                    mode="w",
                    new_vars=list(map(str, dataset.data_vars)),
                    existing_vars=[],
                    n_existing_time=None,
                    append_time=False,
                    dirty=False,
                )
        else:
            # Explicit mode: build a minimal plan that just records the mode.
            # _execute_write_plan handles "w" and "r+" as single calls, and
            # "a" without decomposition (caller's responsibility).
            plan = WritePlan(
                mode=mode,
                new_vars=list(map(str, dataset.data_vars)),
                existing_vars=[],
                n_existing_time=None,
                append_time=False,
                dirty=False,
            )

        # ------------------------------------------------------------------
        # Handle region writes (only for explicit non-auto modes)
        # ------------------------------------------------------------------
        if mode != "auto" and region != "auto":
            region_keys = set(region)
            to_drop = (
                {
                    name
                    for name, var in dataset.data_vars.items()
                    if region_keys.isdisjoint(map(str, var.dims))
                }
                | {str(dim) for dim in dataset.dims}
                | {str(coord) for coord in dataset.coords}
            )
            sliced = dataset.drop_vars(to_drop, errors="ignore").isel(region)
            zarr_options: dict[str, Any] = dict(
                compute=compute,
                mode=plan.mode,
                zarr_format=zarr_format,
                region=region,
            )
            if zarr_format == 2:
                zarr_options["consolidated"] = True
            if level_encoding is not None:
                zarr_options["encoding"] = level_encoding
            _with_retry(
                lambda: sliced.to_zarr(store, **zarr_options),
                max_retries=max_retries,
                backoff=retry_backoff,
            )
            continue

        # ------------------------------------------------------------------
        # Normal (non-region) write via plan executor
        # ------------------------------------------------------------------
        _execute_write_plan(
            dataset,
            store,
            plan,
            zarr_format=zarr_format,
            compute=compute,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            encoding=level_encoding,
        )

        if mode == "w" and not compute:
            coord_opts: dict[str, Any] = dict(
                compute=compute, mode="w", zarr_format=zarr_format
            )
            if zarr_format == 2:
                coord_opts["consolidated"] = True
            dataset[list(dataset.coords)].to_zarr(store, **coord_opts)


__all__ = [
    "get_s3_options",
    "save_pyramid_to_s3",
]
