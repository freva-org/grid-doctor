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
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, NamedTuple, cast

import dask.array as da
import s3fs
import xarray as xr
import zarr
from xarray.backends.zarr import (
    encode_zarr_variable as _xr_encode_zarr_variable,
)

from grid_doctor.types import (
    EncodingDict,
    IndexSlice,
    S3Options,
    ZarrArray,
    ZarrOptions,
    ZarrStore,
)

# ``encode_zarr_variable`` gained ``zarr_format`` in recent xarray releases.
_ENCODE_TAKES_FORMAT = (
    "zarr_format" in inspect.signature(_xr_encode_zarr_variable).parameters
)


def encode_zarr_variable(
    var: xr.Variable, *, name: str, zarr_format: Literal[2, 3]
) -> xr.Variable:
    """Encode *var* exactly as ``to_zarr`` would, across xarray versions."""
    kwargs: dict[str, str | Literal[2, 3]] = {"name": name}
    if _ENCODE_TAKES_FORMAT:
        kwargs["zarr_format"] = zarr_format
    encoded = _xr_encode_zarr_variable(var, **kwargs)  # type: ignore
    return cast(xr.Variable, encoded)


def get_array(group: zarr.Group, name: str) -> ZarrArray:
    """Return the array *name* of *group*; a sub-group of that name is an error."""
    node = group[name]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"{name!r} in {group.store_path} is a group, not an array")
    return node


class DeferredWrite(NamedTuple):
    """One dask-backed variable waiting to be written to its zarr array."""

    source: da.Array
    target: ZarrArray
    region: tuple[IndexSlice, ...]


def stored_encodings(store: ZarrStore, *, zarr_format: Literal[2, 3]) -> EncodingDict:
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
    store: ZarrStore,
    *,
    encoding: Mapping[str, Mapping[str, object]] | None,
    region: dict[str, IndexSlice] | None,
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
        writes.append(
            DeferredWrite(encoded.data, get_array(group, name), target_region)
        )
    return writes


def store_all(writes: list[DeferredWrite]) -> None:
    """Write all collected variables in a single dask computation."""
    if writes:
        da.store(
            [w.source for w in writes],
            # zarr arrays are valid targets (they support __setitem__), but
            # dask annotates targets as array-likes.
            [w.target for w in writes],  # type: ignore[misc]
            regions=[w.region for w in writes],
            lock=False,
        )


def get_zarr_options(
    mode: Literal["a", "w", "r+"],
    zarr_format: Literal[2, 3],
    encoding: Mapping[str, Mapping[str, object]] | None,
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
    s3_options: S3Options | None,
) -> s3fs.S3FileSystem | None:
    """Get appropriate filesystem for path."""
    is_s3 = path.startswith("s3://")
    fs = s3fs.S3FileSystem(**(s3_options or {})) if is_s3 else None
    return fs


def get_store(path: str, fs: s3fs.S3FileSystem | None) -> ZarrStore:
    """Return a location in the provided filesystem."""
    if fs:
        store: ZarrStore = s3fs.S3Map(root=path, s3=fs)
        return store
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
