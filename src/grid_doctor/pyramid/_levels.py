"""xarray layer for building coarse HEALPix levels.

Turns the array kernels in [`_blocks`][grid_doctor.pyramid._blocks] into
datasets: picks the coarsening strategy, wraps results with the right
dimensions and coordinates, and chains levels so that each one builds on
the previous one (a pyramid computed in one pass regrids every chunk only
once).  The public entry points stay in
[`grid_doctor.helpers`][grid_doctor.helpers].
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr

from ..remap import _healpix_centres, _make_crs_variable
from ..types import CoarsenMode, FloatArray
from ._blocks import MeanPartials, coarsen_mean_steps, coarsen_mode_steps

ModeKernel = Callable[..., np.ndarray]
"""Array kernel ``f(values, *, factor, min_valid_fraction)`` for mode coarsening."""


def healpix_coords(level: int, *, nest: bool) -> tuple[FloatArray, FloatArray]:
    """Return ``(lat_deg, lon_deg)`` HEALPix cell centres for *level*."""
    return _healpix_centres(level, nest=nest)


def resolve_coarsen_mode(ds: xr.Dataset, coarsen_mode: CoarsenMode) -> CoarsenMode:
    """Resolve ``"auto"`` to ``"mean"`` or ``"mode"`` from dataset attributes."""
    if coarsen_mode != "auto":
        return coarsen_mode
    method = str(ds.attrs.get("grid_doctor_method", "conservative"))
    is_categorical = method == "nearest" or method.endswith("-mode")
    return "mode" if is_categorical else "mean"


def cell_chunk_of(ds: xr.Dataset) -> int | None:
    """Cell chunk size of the first dask-backed cell variable, if any."""
    for data in ds.data_vars.values():
        if "cell" in data.dims and data.chunks is not None:
            return int(data.chunks[data.get_axis_num("cell")][0])
    return None


def wrap_cells(template: xr.DataArray, values: Any) -> xr.DataArray:
    """Wrap coarsened *values* like *template* (cell last), minus cell coords."""
    coords = {str(k): v for k, v in template.coords.items() if "cell" not in v.dims}
    return xr.DataArray(
        values,
        dims=template.dims,
        coords=coords,
        attrs=template.attrs.copy(),
        name=template.name,
    )


def assemble_coarse_level(
    template: xr.Dataset,
    data_vars: dict[str, xr.DataArray],
    target_level: int,
    source_level: int,
) -> xr.Dataset:
    """Build a coarse-level dataset with HEALPix coordinates and metadata."""
    target_nside = 2**target_level
    npix_target = 12 * target_nside**2
    result = xr.Dataset(data_vars, attrs=template.attrs.copy())
    lat_deg, lon_deg = healpix_coords(target_level, nest=True)
    result = result.assign_coords(
        cell=np.arange(npix_target, dtype=np.int64),
        latitude=("cell", lat_deg),
        longitude=("cell", lon_deg),
        crs=_make_crs_variable(
            level=target_level,
            nside=target_nside,
            order="nested",
        ),
    )

    # Tag every spatially-mapped data variable.
    for name in result.data_vars:
        if "cell" in result[name].dims:
            result[name].attrs["grid_mapping"] = "crs"

    result.attrs["healpix_nside"] = target_nside
    result.attrs["healpix_level"] = target_level
    result.attrs["healpix_order"] = "nested"
    result.attrs["grid_doctor_coarsened_from_level"] = source_level
    return result


def coarsen_dataset(
    ds: xr.Dataset,
    *,
    source_level: int,
    target_level: int,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
    mode_kernel: ModeKernel,
) -> xr.Dataset:
    """Coarsen every cell variable of *ds* from *source_level* to *target_level*.

    *coarsen_mode* must already be resolved (``"mean"`` or ``"mode"``).
    """
    steps = source_level - target_level
    cell_chunk = cell_chunk_of(ds)
    coarsened_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if "cell" not in data.dims:
            coarsened_vars[str(name)] = data
            continue
        template = data.transpose(..., "cell")
        if coarsen_mode == "mode":
            out = coarsen_mode_steps(
                template.data,
                steps,
                kernel=mode_kernel,
                min_valid_fraction=min_valid_fraction,
                cell_chunk=cell_chunk,
            )
        else:
            out = coarsen_mean_steps(
                template.data,
                steps,
                min_valid_fraction=min_valid_fraction,
                cell_chunk=cell_chunk,
            )
        coarsened_vars[str(name)] = wrap_cells(template, out)

    return assemble_coarse_level(ds, coarsened_vars, target_level, source_level)


def coarse_levels(
    finest: xr.Dataset,
    *,
    max_level: int,
    min_level: int,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
    mode_kernel: ModeKernel,
) -> dict[int, xr.Dataset]:
    """Coarsen *finest* level by level, sharing work between levels.

    Means carry sums and counts of valid finest-level cells, so every
    level is exact (issue #54) while each step is a chunk-local
    factor-4 reduction. Each level's graph builds on the previous one,
    so computing all levels together regrids every chunk only once.
    *coarsen_mode* must already be resolved (``"mean"`` or ``"mode"``).
    """
    cell_chunk = cell_chunk_of(finest)
    templates = {
        str(name): data.transpose(..., "cell")
        for name, data in finest.data_vars.items()
        if "cell" in data.dims
    }
    state: dict[str, Any] = {name: t.data for name, t in templates.items()}
    levels: dict[int, xr.Dataset] = {}
    for level in range(max_level - 1, min_level - 1, -1):
        data_vars: dict[str, xr.DataArray] = {}
        for name, data in finest.data_vars.items():
            name = str(name)
            if name not in templates:
                data_vars[name] = data
                continue
            if coarsen_mode == "mode":
                state[name] = coarsen_mode_steps(
                    state[name],
                    1,
                    kernel=mode_kernel,
                    min_valid_fraction=min_valid_fraction,
                    cell_chunk=cell_chunk,
                )
                values = state[name]
            else:
                partials = state[name]
                partials = (
                    partials.coarsen()
                    if isinstance(partials, MeanPartials)
                    else MeanPartials.from_values(partials)
                ).rechunk(cell_chunk)
                state[name] = partials
                values = partials.mean(min_valid_fraction)
            data_vars[name] = wrap_cells(templates[name], values)
        levels[level] = assemble_coarse_level(finest, data_vars, level, max_level)
    return levels
