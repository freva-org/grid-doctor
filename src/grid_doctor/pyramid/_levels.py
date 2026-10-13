"""xarray layer for building coarse HEALPix levels.

Turns the array kernels in [`_blocks`][grid_doctor.pyramid._blocks] into
datasets: picks the coarsening strategy, wraps results with the right
dimensions and coordinates, and chains levels so that each one builds on
the previous one (a pyramid computed in one pass regrids every chunk only
once).  The public entry points stay in
[`grid_doctor.helpers`][grid_doctor.helpers].
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr

from ..remap import _healpix_centres, _make_crs_variable
from ..types import CoarsenMode, FloatArray, ValidFraction
from ._blocks import (
    MeanPartials,
    coarsen_counts,
    coarsen_mean_steps,
    coarsen_mode_steps,
    count_valid_steps,
)

ModeKernel = Callable[..., np.ndarray]
"""Array kernel ``f(values, *, factor, min_valid_fraction)`` for mode coarsening."""

FractionShape = Literal["full", "static"]
"""Resolved shape of one ``<name>_valid_fraction`` variable."""


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
    *,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
) -> xr.Dataset:
    """Build a coarse-level dataset with HEALPix coordinates and metadata.

    Records how the level was derived: the source level, the resolved
    coarsening mode and the minimum valid fraction, so that readers can
    tell masked cells from missing data.
    """
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
    result.attrs["grid_doctor_coarsen_mode"] = str(coarsen_mode)
    result.attrs["grid_doctor_min_valid_fraction"] = float(min_valid_fraction)
    return result


def resolve_valid_fraction(
    spec: ValidFraction, ds: xr.Dataset
) -> dict[str, FractionShape]:
    """Map variable name to fraction shape for the requested variables."""
    cell_vars = [str(n) for n, v in ds.data_vars.items() if "cell" in v.dims]
    items: Any
    if spec is False:
        return {}
    if spec is True or spec == "static":
        items = ((name, spec) for name in cell_vars)
    elif isinstance(spec, str):
        raise ValueError(
            f"valid_fraction={spec!r}: use True, 'static', or variable names."
        )
    elif isinstance(spec, Mapping):
        items = spec.items()
    else:
        items = ((name, True) for name in spec)

    resolved: dict[str, FractionShape] = {}
    for name, shape in items:
        if name not in cell_vars:
            raise ValueError(
                f"valid_fraction: {name!r} is not a variable with a 'cell' "
                f"dimension (available: {cell_vars})."
            )
        if shape is False:
            continue
        if shape is not True and shape != "static":
            raise ValueError(f"valid_fraction[{name!r}] must be True or 'static'.")
        resolved[name] = "static" if shape == "static" else "full"

    clashes = sorted(
        f"{name}_valid_fraction"
        for name in resolved
        if f"{name}_valid_fraction" in ds.variables
    )
    if clashes:
        raise ValueError(f"valid_fraction: {clashes} already exist in the dataset.")
    return resolved


def add_fraction(
    data_vars: dict[str, xr.DataArray],
    name: str,
    template: xr.DataArray,
    counts: Any,
    *,
    n_fine: int,
    shape: FractionShape,
    source_level: int,
) -> None:
    """Add ``<name>_valid_fraction`` and link it from ``data_vars[name]``.

    *counts* holds the valid source cells per cell (cell axis last);
    *n_fine* is the number of source cells under each cell.
    """
    fraction = (counts / n_fine).astype(np.float32)
    if shape == "static":
        fraction = fraction[(0,) * (fraction.ndim - 1)]
        frac_da = xr.DataArray(fraction, dims=("cell",))
    else:
        frac_da = wrap_cells(template, fraction)
    frac_name = f"{name}_valid_fraction"
    frac_da.attrs = {
        "long_name": f"fraction of valid level-{source_level} cells of {name}",
        "units": "1",
    }
    if "grid_mapping" in template.attrs:
        frac_da.attrs["grid_mapping"] = template.attrs["grid_mapping"]
    data_vars[frac_name] = frac_da

    data = data_vars[name].copy()
    linked = str(data.attrs.get("ancillary_variables", "")).split()
    if frac_name not in linked:
        data.attrs["ancillary_variables"] = " ".join([*linked, frac_name])
    data_vars[name] = data


def with_finest_fractions(
    finest: xr.Dataset,
    fractions: dict[str, FractionShape],
    level: int,
) -> xr.Dataset:
    """Add 0/1 valid fractions to the finest level (it is its own source)."""
    if not fractions:
        return finest
    data_vars = {str(n): v for n, v in finest.data_vars.items()}
    for name, shape in fractions.items():
        template = finest[name].transpose(..., "cell")
        add_fraction(
            data_vars,
            name,
            template,
            np.isfinite(template.data),
            n_fine=1,
            shape=shape,
            source_level=level,
        )
    return finest.assign(data_vars)


def coarsen_dataset(
    ds: xr.Dataset,
    *,
    source_level: int,
    target_level: int,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
    mode_kernel: ModeKernel,
    valid_fraction: ValidFraction = False,
) -> xr.Dataset:
    """Coarsen every cell variable of *ds* from *source_level* to *target_level*.

    *coarsen_mode* must already be resolved (``"mean"`` or ``"mode"``).
    Fractions are relative to the cells of *ds*.
    """
    steps = source_level - target_level
    cell_chunk = cell_chunk_of(ds)
    fractions = resolve_valid_fraction(valid_fraction, ds)
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
        if str(name) in fractions:
            add_fraction(
                coarsened_vars,
                str(name),
                template,
                count_valid_steps(template.data, steps, cell_chunk=cell_chunk),
                n_fine=4**steps,
                shape=fractions[str(name)],
                source_level=source_level,
            )

    return assemble_coarse_level(
        ds,
        coarsened_vars,
        target_level,
        source_level,
        coarsen_mode=coarsen_mode,
        min_valid_fraction=min_valid_fraction,
    )


def coarse_levels(
    finest: xr.Dataset,
    *,
    max_level: int,
    min_level: int,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
    mode_kernel: ModeKernel,
    valid_fraction: dict[str, FractionShape] | None = None,
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
    fractions = valid_fraction or {}
    state: dict[str, Any] = {name: t.data for name, t in templates.items()}
    # Mode coarsening has no counts of its own; keep a separate chain.
    count_state: dict[str, Any] = {name: t.data for name, t in templates.items()}
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
                if name in fractions:
                    count_state[name] = (
                        count_valid_steps(count_state[name], 1, cell_chunk=cell_chunk)
                        if level == max_level - 1
                        else coarsen_counts(count_state[name], cell_chunk=cell_chunk)
                    )
                counts = count_state[name]
            else:
                partials = state[name]
                partials = (
                    partials.coarsen()
                    if isinstance(partials, MeanPartials)
                    else MeanPartials.from_values(partials)
                ).rechunk(cell_chunk)
                state[name] = partials
                values = partials.mean(min_valid_fraction)
                counts = partials.counts
            data_vars[name] = wrap_cells(templates[name], values)
            if name in fractions:
                add_fraction(
                    data_vars,
                    name,
                    templates[name],
                    counts,
                    n_fine=4 ** (max_level - level),
                    shape=fractions[name],
                    source_level=max_level,
                )
        levels[level] = assemble_coarse_level(
            finest,
            data_vars,
            level,
            max_level,
            coarsen_mode=coarsen_mode,
            min_valid_fraction=min_valid_fraction,
        )
    return levels
