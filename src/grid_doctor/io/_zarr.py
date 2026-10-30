"""Single-pass Zarr writes for HEALPix pyramids.

xarray creates every store (metadata, encodings, NumPy-backed variables)
eagerly with ``to_zarr(compute=False)``.  The dask-backed data of all
stores is then written by one ``da.store`` call: separate
``to_zarr(compute=True)`` calls -- or separate Delayed objects -- would
each rebuild the shared finest-level graph and regrid every chunk once
per level.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Union, cast

import dask.array as da
import s3fs
import xarray as xr
import zarr
from xarray.backends.zarr import (
    encode_zarr_variable as _xr_encode_zarr_variable,
)

from grid_doctor.types import ZarrOptions

# ``encode_zarr_variable`` gained ``zarr_format`` in recent xarray releases.
_ENCODE_TAKES_FORMAT = (
    "zarr_format" in inspect.signature(_xr_encode_zarr_variable).parameters
)


def encode_zarr_variable(
    var: xr.Variable, *, name: str, zarr_format: Literal[2, 3]
) -> xr.Variable:
    """Encode *var* exactly as ``to_zarr`` would, across xarray versions."""
    kwargs: Dict[str, Union[str, Literal[2, 3]]] = {"name": name}
    if _ENCODE_TAKES_FORMAT:
        kwargs["zarr_format"] = zarr_format
    encoded = _xr_encode_zarr_variable(var, **kwargs)  # type: ignore
    return cast(xr.Variable, encoded)


class DeferredWrite(NamedTuple):
    """One dask-backed variable waiting to be written to its zarr array."""

    source: Any
    target: Any
    region: tuple[slice, ...]


def stored_encodings(
    store: Any, *, zarr_format: Literal[2, 3]
) -> dict[str, dict[str, Any]]:
    """CF encodings of the variables already defined in *store*.

    The encodings are what xarray decodes from the stored attributes
    (``scale_factor``, ``add_offset``, ``_FillValue``, ``dtype``, time
    ``units`` and ``calendar``), so encoding new data with them
    reproduces the values ``to_zarr`` would write into the same arrays.
    """
    existing = xr.open_zarr(store, zarr_format=zarr_format, chunks=None)
    try:
        return {
            str(name): dict(var.encoding) for name, var in existing.variables.items()
        }
    finally:
        existing.close()


def deferred_writes(
    dataset: xr.Dataset,
    store: Any,
    *,
    encoding: dict[str, dict[str, Any]] | None,
    region: dict[str, slice] | None,
    zarr_format: Literal[2, 3],
) -> list[DeferredWrite]:
    """Collect encoded dask sources and zarr targets of an initialised store."""
    # NumPy-backed variables were already written by ``to_zarr(compute=False)``.
    lazy = {
        str(name): var
        for name, var in dataset.variables.items()
        if var.chunks is not None
    }
    if not lazy:
        return []
    group = zarr.open_group(store, mode="r+", zarr_format=zarr_format)
    stored = stored_encodings(store, zarr_format=zarr_format)
    writes: list[DeferredWrite] = []
    for name, var in lazy.items():
        var = var.copy(deep=False)
        if name in stored:
            # Encode exactly as the existing array expects (packing, fill
            # value, time units), like ``to_zarr`` does for existing
            # variables.  Region updates usually carry no encoding at all.
            var.encoding = stored[name]
        else:
            var.encoding = {**var.encoding, **(encoding or {}).get(name, {})}
        encoded = encode_zarr_variable(var, name=name, zarr_format=zarr_format)
        target_region = tuple(
            (region or {}).get(str(dim), slice(None)) for dim in var.dims
        )
        writes.append(DeferredWrite(encoded.data, group[name], target_region))
    return writes


def store_all(writes: list[DeferredWrite]) -> None:
    """Write all collected variables in a single dask computation."""
    if writes:
        da.store(
            [w.source for w in writes],
            [w.target for w in writes],
            regions=[w.region for w in writes],
            lock=False,
        )


def get_zarr_options(
    mode: Literal["a", "w", "r+"],
    zarr_format: Literal[2, 3],
    encoding: dict[str, dict[str, Any]] | None,
) -> ZarrOptions:
    """Build zarr options."""
    zarr_options = ZarrOptions(compute=False, mode=mode, zarr_format=zarr_format)
    if zarr_format == 2:
        zarr_options["consolidated"] = True
    if encoding is not None:
        zarr_options["encoding"] = encoding
    return zarr_options


def get_filesystem(
    path: str,
    s3_options: dict[str, Any] | None,
) -> s3fs.S3FileSystem | None:
    """Get appropriate filesystem for path."""
    is_s3 = path.startswith("s3://")
    fs = s3fs.S3FileSystem(**(s3_options or {})) if is_s3 else None
    return fs


def get_store(path: str, fs: s3fs.S3FileSystem | None) -> Any:
    """Return a location in the provided filesystem."""
    store: Any
    if fs:
        store = s3fs.S3Map(root=path, s3=fs)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        store = path
    return store
