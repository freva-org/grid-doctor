"""S3 upload helpers for HEALPix pyramids.

This module owns everything related to writing pyramid datasets to
S3-backed Zarr stores:

- [`get_s3_options`][get_s3_options] — build an authenticated, timeout
  `s3fs.S3FileSystem` configuration dictionary.
- [`save_pyramid_to_s3`][save_pyramid_to_s3] — write a pyramid
  (``dict[int, xr.Dataset]``) level-by-level, with automatic mode inference,
  retry logic, and dirty-store detection.

The three private helpers drive the ``mode="auto"`` path:

- `_inspect_store` — opens the existing store with `xarray.open_zarr`
  and returns a `~grid_doctor.types.WritePlan`.
- `_execute_write_plan` — executes the plan as one or two
  ``to_zarr`` calls.
- `_with_retry` — wraps any zero-argument callable with exponential
  backoff.
"""

from __future__ import annotations

import json
import logging
import math
import time as _time
from pathlib import Path
from typing import Any, Callable, Literal

import dask
import dask.config
import s3fs
import xarray as xr
import zarr

from .helpers import get_latlon_resolution, resolution_to_healpix_level
from .types import WriteMode, WritePlan

logger = logging.getLogger(__name__)


def get_s3_options(
    endpoint_url: str,
    secrets_file: str | Path,
    *,
    read_timeout: int = 300,
    connect_timeout: int = 90,
    max_attempts: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an S3 options dictionary from an endpoint and credentials file.

    The returned dictionary is suitable for passing directly to
    `s3fs.S3FileSystem`.  Sensible timeout and retry defaults are
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
        Options for `s3fs.S3FileSystem`.
    """
    secrets_file = Path(secrets_file).expanduser()
    secrets: dict[str, str] = json.loads(secrets_file.read_text())
    # config_kwargs is a plain dict passed to botocore.config.Config()
    # internally by s3fs.  Using a plain dict (rather than a Config object)
    # keeps the filesystem JSON-serialisable, which zarr requires when it
    # reconstructs an async filesystem from an S3Map.
    options: dict[str, Any] = {
        "endpoint_url": endpoint_url,
        "secret": secrets["secretKey"],
        "key": secrets["accessKey"],
        "config_kwargs": {
            "read_timeout": read_timeout,
            "connect_timeout": connect_timeout,
            "retries": {"max_attempts": max_attempts, "mode": "adaptive"},
        },
    }
    options.update(kwargs)
    return options


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


def _chunk_key_v2(level_path: str, var: str, chunk_indices: list[int]) -> str:
    """Build a Zarr v2 chunk key (dot-separated indices)."""
    return f"{level_path}/{var}/" + ".".join(str(i) for i in chunk_indices)


def _chunk_key_v3(level_path: str, var: str, chunk_indices: list[int]) -> str:
    """Build a Zarr v3 chunk key (``c/`` prefix, slash-separated indices)."""
    return f"{level_path}/{var}/c/" + "/".join(str(i) for i in chunk_indices)


def _inspect_store(
    fs: s3fs.S3FileSystem,
    level_path: str,
    ds: xr.Dataset,
    *,
    validate: bool = False,
) -> WritePlan:
    """Inspect an existing Zarr store (v2 or v3) and produce a `WritePlan`.

    The store is opened lazily with `xarray.open_zarr` — only consolidated
    metadata is fetched, no data is loaded.  Zarr format (v2/v3) is detected
    automatically by zarr-python; no manual JSON parsing is required.

    When the store cannot be opened (missing or unrecognised metadata), a
    warning is logged and the plan falls back to ``mode="w"``.

    When *validate* is ``True``, an additional `s3fs.S3FileSystem.exists`
    check is performed for the last chunk along the time dimension of each
    data variable already present in the store.  A missing chunk also
    triggers a ``mode="w"`` fallback.

    Args:
        fs: Authenticated S3 filesystem.
        level_path: Full S3 path to the ``.zarr`` store (no trailing slash).
        ds: Incoming dataset that we intend to write.
        validate: When ``True``, verify that the last time-chunk of every
            existing variable is present.  Falls back to ``mode="w"`` if any
            chunk is missing.

    Returns:
        A [`WritePlan`][WritePlan] describing the required write operations.
    """
    # --- Empty store → full write ----------------------------------------
    store_non_empty = fs.exists(level_path) and bool(fs.ls(level_path, detail=False))
    if not store_non_empty:
        return WritePlan(
            mode="w",
            new_vars=list(map(str, ds.data_vars)),
            existing_vars=[],
            n_existing_time=None,
            append_time=False,
        )

    # --- Open existing store lazily (metadata only) ----------------------
    store = s3fs.S3Map(root=level_path, s3=fs)
    try:
        existing_ds = xr.open_zarr(
            store,
            chunks={},
            decode_cf=False,
            mask_and_scale=False,
            decode_times=False,
            decode_timedelta=False,
            decode_coords=False,
            consolidated=None,
        )
    except Exception as exc:
        logger.warning(
            "Could not open store at %s (%s: %s) — falling back to full overwrite.",
            level_path,
            type(exc).__name__,
            exc,
        )
        return WritePlan(
            mode="w",
            new_vars=list(map(str, ds.data_vars)),
            existing_vars=[],
            n_existing_time=None,
            append_time=False,
        )

    # --- Compare variables -----------------------------------------------
    store_arrays = set(map(str, existing_ds.data_vars))
    incoming_vars = set(map(str, ds.data_vars))
    new_vars = sorted(incoming_vars - store_arrays)
    existing_vars = sorted(incoming_vars & store_arrays)

    # --- Compare time dimension ------------------------------------------
    n_existing_time: int | None = existing_ds.sizes.get("time")
    incoming_time = ds.sizes.get("time")
    append_time = (
        incoming_time is not None
        and n_existing_time is not None
        and incoming_time > n_existing_time
    )

    # --- Validate last chunk per existing variable -----------------------
    if validate and existing_vars and n_existing_time is not None:
        # Detect format for chunk key construction: v2 uses dot-separated
        # indices, v3 uses a "c/" prefix with slash-separated indices.
        chunk_key_fn = (
            _chunk_key_v2 if fs.exists(f"{level_path}/.zmetadata") else _chunk_key_v3
        )
        for var in existing_vars:
            data_var = existing_ds[var]
            enc_chunks = data_var.encoding.get("chunks")
            if enc_chunks is None:
                continue
            shape = list(data_var.shape)
            chunk_list = list(enc_chunks)
            if not shape or not chunk_list or len(shape) != len(chunk_list):
                continue

            n_time_chunks = math.ceil(shape[0] / chunk_list[0])
            chunk_indices = [0] * len(shape)
            chunk_indices[0] = n_time_chunks - 1
            chunk_key = chunk_key_fn(level_path, var, chunk_indices)
            if not fs.exists(chunk_key):
                logger.warning(
                    "Store at %s appears incomplete (missing chunk %s) — "
                    "falling back to full overwrite.",
                    level_path,
                    chunk_key,
                )
                return WritePlan(
                    mode="w",
                    new_vars=list(map(str, ds.data_vars)),
                    existing_vars=[],
                    n_existing_time=None,
                    append_time=False,
                )

    # --- Resolve write mode ----------------------------------------------
    if not new_vars and not append_time:
        resolved_mode: Literal["w", "a", "r+"] = "r+"
    else:
        resolved_mode = "a"

    return WritePlan(
        mode=resolved_mode,
        new_vars=new_vars,
        existing_vars=existing_vars,
        n_existing_time=n_existing_time,
        append_time=append_time,
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
    """Execute a [`WritePlan`][WritePlan] against *store*, with per-call retries.

    The write is decomposed into at most two ``to_zarr`` calls:

    1. **Add new variables** (for the time range already in the store).
    2. **Append new time steps** (for all variables, including any just added
       in step 1).

    Each call is wrapped in `_with_retry` independently so a transient
    S3 timeout in step 2 does not force a repeat of step 1.

    Args:
        dataset: Incoming dataset.
        store: Target Zarr store.
        plan: Write plan produced by `_inspect_store`.
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
        """Write *ds_slice* to *store*, retrying on transient S3 failures.

        For ``mode="w"``, the write is split into two phases to prevent
        retries from wiping successfully-written chunks:

        1. **Metadata initialisation** — ``to_zarr(compute=False, mode="w")``
           creates the Zarr group and array metadata on S3 (small, fast,
           runs once).  Opening a store with ``mode="w"`` is what clears
           existing content, so we do this exactly once.
        2. **Chunk writes** — ``to_zarr(compute=True, mode="r+")`` computes
           the Dask graph and writes chunk data into the already-initialised
           store.  ``mode="r+"`` overwrites chunk files in place without
           clearing the store, so retries never lose successfully-written
           chunks.

        For ``mode="a"`` and ``mode="r+"`` the store is never cleared on
        open, so a single retried call is safe.
        """
        write_mode: str = extra.get("mode", base_opts.get("mode", "w"))
        opts = {**base_opts, **extra}

        if write_mode == "w":
            # Phase 1: initialise store metadata once (clears existing store).
            init_opts = {**opts, "compute": False, "consolidated": False}
            ds_slice.to_zarr(store, **init_opts)

            # Phase 2: retry chunk writes without re-clearing the store.
            data_opts = {**opts, "mode": "r+"}
            _with_retry(
                lambda: ds_slice.to_zarr(store, **data_opts),
                max_retries=max_retries,
                backoff=retry_backoff,
            )
        else:
            _with_retry(
                lambda: ds_slice.to_zarr(store, **opts),
                max_retries=max_retries,
                backoff=retry_backoff,
            )

    mode = plan.mode

    # "w" or "r+" - single call covers everything
    if mode in {"w", "r+"}:
        _zarr_write(dataset, mode=mode)
        return

    # "a" - may require two calls

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


def _is_gpu_backed(ds: xr.Dataset) -> bool:
    """Return ``True`` if *ds* was computed with the CuPy GPU backend.

    Reads the ``grid_doctor_backend`` attribute stamped on the dataset by
    `regrid_to_healpix` and propagated through `coarsen_healpix`.  This is
    reliable because the attribute is set at graph-construction time from the
    resolved backend, not inferred from dask internals (which always report
    numpy meta regardless of whether CuPy is used inside ``apply_ufunc``).

    Returns ``False`` for datasets that do not carry the attribute (e.g.
    datasets not produced by grid-doctor).

    Args:
        ds: Dataset to inspect.

    Returns:
        ``True`` if ``ds.attrs["grid_doctor_backend"] == "cupy"``.
    """
    return ds.attrs.get("grid_doctor_backend") == "cupy"


def _save_pyramid_sequential(
    pyramid: dict[int, xr.Dataset],
    fs: s3fs.S3FileSystem,
    s3_path: str,
    *,
    mode: WriteMode,
    compute: bool,
    region: Literal["auto"] | dict[str, slice],
    zarr_format: Literal[2, 3],
    encoding: dict[int, dict[str, dict[str, Any]]] | None,
    validate: bool,
    max_retries: int,
    retry_backoff: float,
) -> None:
    """Write all pyramid levels sequentially, level by level.

    Used in two situations:

    1. **GPU-backed pyramids** — all levels carry
       ``grid_doctor_backend="cupy"``.  The synchronous Dask scheduler is
       forced for each level so that CuPy kernels never run concurrently
       in multiple threads, which would exhaust GPU memory.

    2. **Distributed not available** — fallback when the ``distributed``
       package is not installed or when no workers are reachable.

    Args:
        pyramid: Mapping of HEALPix level to dataset.
        fs: Authenticated S3 filesystem.
        s3_path: Normalised S3 prefix (no ``s3://`` prefix, no trailing slash).
        mode: Write mode — same semantics as `save_pyramid_to_s3`.
        compute: Trigger Dask execution immediately when ``True``.
        region: Region writes for partial updates.
        zarr_format: Zarr format version.
        encoding: Per-level encoding dictionaries.
        validate: Whether to validate last chunks in existing stores.
        max_retries: Maximum retry attempts per ``to_zarr`` call.
        retry_backoff: Exponential backoff multiplier between retries.
    """
    for level, dataset in pyramid.items():
        level_path = f"s3://{s3_path}/level_{level}.zarr"
        logger.info("Writing HEALPix level %s to %s", level, level_path)
        store = s3fs.S3Map(root=level_path, s3=fs)
        level_encoding = encoding[level] if encoding is not None else None

        if mode == "auto":
            plan = _inspect_store(fs, level_path, dataset, validate=validate)
        else:
            plan = WritePlan(
                mode=mode,
                new_vars=list(map(str, dataset.data_vars)),
                existing_vars=[],
                n_existing_time=None,
                append_time=False,
            )

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

        # GPU-backed levels must use the synchronous (single-threaded) Dask
        # scheduler to prevent concurrent CuPy allocations across threads.
        dask_cfg: dict[str, Any] = (
            {"scheduler": "synchronous"} if _is_gpu_backed(dataset) else {}
        )
        with dask.config.set(**dask_cfg):
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
    max_retries: int = 5,
    retry_backoff: float = 2.0,
) -> None:
    """Write a HEALPix pyramid to S3-backed Zarr stores.

    Each level is stored below ``"<s3_path>/level_<level>.zarr"``.

    When *mode* is ``"auto"`` the function inspects each level's existing
    store and determines the minimal set of operations required:

    - **Store absent or inconsistent** → full write (``"w"``).
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
        Options forwarded to `s3fs.S3FileSystem`.  Use
        [`get_s3_options`][get_s3_options] to build this with sensible
        timeout defaults.
    mode:
        ``"w"`` — overwrite entirely.
        ``"a"`` — append / add.
        ``"r+"`` — overwrite data only (store must already exist).
        ``"auto"`` — inspect each store and infer the correct operations.
        Inconsistent stores are silently overwritten.
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
        the store's state.  Adds one `s3fs.S3FileSystem.exists` call
        per data variable.  An incomplete store falls back to ``mode="w"``.
    max_retries:
        Maximum number of retry attempts per individual ``to_zarr`` call.
    retry_backoff:
        Multiplicative factor for the exponential back-off between retries.
        With the default of ``2.0`` and a 2 s base delay, the waits are
        2 s, 4 s, 8 s, … up to *max_retries* attempts.
    """
    fs = s3fs.S3FileSystem(**s3_options)
    s3_path = s3_path.rstrip("/").removeprefix("s3://")
    _save_pyramid_sequential(
        pyramid,
        fs,
        s3_path,
        mode=mode,
        compute=compute,
        region=region,
        zarr_format=zarr_format,
        encoding=encoding,
        validate=validate,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def create_and_upload_healpix_pyramid(
    ds: xr.Dataset,
    s3_path: str,
    s3_options: dict[str, Any],
    *,
    max_level: int | None = None,
    min_level: int = 0,
    zarr_format: Literal[2, 3] = 2,
    encoding: dict[int, dict[str, dict[str, Any]]] | None = None,
    max_retries: int = 5,
    retry_backoff: float = 2.0,
    **kwargs: Any,
) -> None:
    """Remap a source dataset to HEALPix and upload each pyramid level to S3.

    This is the recommended entry point for GPU and large-data workflows.
    Rather than building the full pyramid in memory and then uploading,
    it pipelines computation and upload level by level:

    1. Remap the source dataset to *max_level*.
    2. Upload the finest level to S3.
    3. Read the uploaded data back from S3 lazily (via ``xr.open_zarr``).
    4. Coarsen to the next level using the S3-backed data.
    5. Upload and repeat down to *min_level*.

    Reading back from S3 after each upload breaks the Dask computation
    graph — the coarsening input has no upstream GPU tasks, so workers
    never recompute expensive remapping kernels.  This eliminates the
    GPU OOM errors that occur when a lazy CuPy graph is submitted to
    a distributed scheduler whose workers lack GPU context.

    The network round-trip cost is small relative to GPU compute time
    (level 7 ≈ 100 GB at ~10 GB/s IB = ~10 s) and the approach works
    identically on CPU-only machines.

    Parameters
    ----------
    ds:
        Source dataset on any supported grid (regular, curvilinear,
        or unstructured).
    s3_path:
        S3 prefix such as ``"s3://bucket/pyramid"``.
    s3_options:
        Options forwarded to `s3fs.S3FileSystem`.  Build with
        [`get_s3_options`][get_s3_options].
    max_level:
        Finest HEALPix level to generate.  When omitted, estimated
        from the source-grid resolution.
    min_level:
        Coarsest HEALPix level to keep.  Default ``0``.
    zarr_format:
        Zarr format version (2 or 3).
    encoding:
        Per-level encoding dictionaries, keyed by level number.
    max_retries:
        Maximum retry attempts per ``to_zarr`` call.
    retry_backoff:
        Exponential backoff multiplier between retries.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]
        (e.g. ``method``, ``backend``, ``weights_path``).
    """
    # Lazy imports to avoid circular dependency — helpers imports s3.
    from .helpers import coarsen_healpix
    from .remap import regrid_to_healpix

    fs = s3fs.S3FileSystem(**s3_options)
    s3_path = s3_path.rstrip("/").removeprefix("s3://")

    if max_level is None:
        max_level = resolution_to_healpix_level(get_latlon_resolution(ds))

    logger.info("Remapping source dataset to HEALPix level %d.", max_level)
    finest = regrid_to_healpix(ds, max_level, **kwargs)

    # Upload finest level, then iterate downwards using S3-backed data
    # for coarsening so the GPU graph is not retained across levels.
    current_ds = finest

    for level in range(max_level, min_level - 1, -1):
        level_path = f"s3://{s3_path}/level_{level}.zarr"
        store = s3fs.S3Map(root=level_path, s3=fs)
        level_encoding = encoding[level] if encoding is not None else None

        logger.info("Uploading HEALPix level %d to %s.", level, level_path)
        plan = WritePlan(
            mode="w",
            new_vars=list(map(str, current_ds.data_vars)),
            existing_vars=[],
            n_existing_time=None,
            append_time=False,
        )

        # Force synchronous Dask scheduler to prevent concurrent CuPy
        # allocations across threads when using the GPU backend.
        dask_cfg: dict[str, Any] = (
            {"scheduler": "synchronous"}
            if current_ds.attrs.get("grid_doctor_backend") == "cupy"
            else {}
        )
        with dask.config.set(**dask_cfg):
            _execute_write_plan(
                current_ds,
                store,
                plan,
                zarr_format=zarr_format,
                compute=True,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                encoding=level_encoding,
            )

        if zarr_format == 2:
            zarr.consolidate_metadata(store)

        if level == min_level:
            break

        # Read back from S3 lazily — severs the upstream Dask graph so
        # coarsening has no GPU tasks and workers never OOM.
        logger.debug("Reading level %d back from S3 for coarsening.", level)
        current_ds = xr.open_zarr(
            store,
            chunks={},
            consolidated=(zarr_format == 2),
        )

        next_level = level - 1
        logger.info("Coarsening level %d -> %d.", level, next_level)
        current_ds = coarsen_healpix(current_ds, next_level)


__all__ = [
    "create_and_upload_healpix_pyramid",
    "get_s3_options",
    "save_pyramid_to_s3",
]
