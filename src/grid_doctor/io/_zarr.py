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
from typing import Any, Dict, Literal, NamedTuple, Union, cast

import dask.array as da
import xarray as xr
import zarr
from xarray.backends.zarr import encode_zarr_variable as _xr_encode_zarr_variable

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
    writes: list[DeferredWrite] = []
    for name, var in lazy.items():
        var = var.copy(deep=False)
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
