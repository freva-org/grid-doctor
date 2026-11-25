"""Build HEALPix pyramids: regrid the finest level, coarsen the rest."""

from __future__ import annotations

import logging
from typing import Unpack

import xarray as xr

from .._geometry.coords import get_latlon_resolution
from .._geometry.healpix import resolution_to_healpix_level
from ..metadata import HealpixMetadata
from ..types import (
    CoarsenMode,
    RegridOptions,
    ValidFraction,
)
from ._kernels import _coarsen_array_mode
from ._levels import (
    coarse_levels,
    coarsen_dataset,
    resolve_coarsen_mode,
    resolve_valid_fraction,
    with_finest_fractions,
)

logger = logging.getLogger(__name__)


def coarsen_healpix(
    ds: xr.Dataset,
    target_level: int,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
    valid_fraction: ValidFraction = False,
) -> xr.Dataset:
    """Coarsen a HEALPix dataset to a lower-resolution level.

    The coarsening is performed as a single reshape + reduction over
    all batch dimensions simultaneously, without per-slice Python loops.

    Parameters
    ----------
    ds:
        HEALPix dataset with a ``cell`` dimension and grid-doctor
        metadata (see [`grid_doctor.metadata`][grid_doctor.metadata]).
    target_level:
        Target HEALPix level (must be lower than the current level).
    coarsen_mode:
        ``"mean"``: NaN-aware averaging for continuous fields.
        ``"mode"``: most frequent value for categorical fields.
        ``"auto"``: infer from the production method recorded for the
        dataset: ``"nearest"`` and ``"binned-mode"`` give mode,
        everything else mean.  The mode is chosen once for the whole
        dataset; coarsen continuous and categorical variables in
        separate calls when a dataset mixes both, for example after
        binning with ``agg={"radiance": "mean", "cloud_type": "mode"}``.
    min_valid_fraction:
        Minimum fraction of valid (non-NaN) children required to
        produce a valid parent cell.  Parents with fewer valid
        children are set to NaN.  Default ``0.5`` (at least half
        of the children must be valid).
    valid_fraction:
        Add ``<name>_valid_fraction`` variables: the fraction of valid
        cells of *ds* under each coarse cell (see
        [`create_healpix_pyramid`][grid_doctor.create_healpix_pyramid]).
        Fractions are relative to the level of *ds*, so coarsen from the
        finest level to get weights for exact global means.

    Returns
    -------
    xarray.Dataset
        Coarsened dataset.

    Notes
    -----
    Nested HEALPix indices have a direct parent-child relationship:
    pixel *i* at level *L* contains children ``4*i`` to ``4*i+3`` at
    level *L+1*.  Coarsening therefore reduces to grouping contiguous
    blocks of ``4**delta_level`` child cells and averaging (or taking
    the mode for categorical data).

    Ring-ordered datasets do not have contiguous parent-child layout
    and must be remapped directly at each target level.

    With ``coarsen_mode="mean"``, always coarsen from the finest level
    rather than chaining level by level.  A chained mean weights every
    valid parent equally, regardless of how many valid finest-level
    cells it was built from, so data with NaNs (land, sea ice,
    observation gaps) drifts between levels.

    Raises
    ------
    ValueError
        When the ordering is not nested, or *target_level* is not
        lower than the current level.
    """
    grid = HealpixMetadata.from_dataset(ds).grid
    if not grid.nest:
        raise ValueError(
            "coarsen_healpix only supports nested HEALPix ordering. "
            "Use create_healpix_pyramid(..., nest=False) to "
            "regenerate ring levels directly."
        )
    current_level = grid.level
    if target_level >= current_level:
        raise ValueError("target_level must be lower than the current HEALPix level.")

    return coarsen_dataset(
        ds,
        source_level=current_level,
        target_level=target_level,
        coarsen_mode=resolve_coarsen_mode(ds, coarsen_mode),
        min_valid_fraction=min_valid_fraction,
        mode_kernel=_coarsen_array_mode,
        valid_fraction=valid_fraction,
    )


# ===================================================================
# Pyramid construction
# ===================================================================


def create_healpix_pyramid(
    ds: xr.Dataset,
    max_level: int | None = None,
    min_level: int = 0,
    *,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
    valid_fraction: ValidFraction = False,
    **kwargs: Unpack[RegridOptions],
) -> dict[int, xr.Dataset]:
    """Create a multi-resolution HEALPix pyramid.

    The input dataset is first remapped to *max_level* with
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].
    For nested output ordering, lower levels are derived efficiently
    with
    [`coarsen_healpix`][grid_doctor.coarsen_healpix].
    For ring ordering, each lower level is regenerated directly from
    the source dataset.

    For dask-backed input the result is lazy and chunked along ``cell``
    (``cell_chunks``, forwarded to
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]); coarser
    levels keep that chunk size.  Every level builds on the finest one,
    so write the pyramid with
    [`save_pyramid`][grid_doctor.save_pyramid], which computes
    all levels in one pass.  Calling ``.compute()`` per level instead
    regrids the source once per level.

    Parameters
    ----------
    ds:
        Source dataset.
    max_level:
        Finest HEALPix level.
    min_level:
        Coarsest HEALPix level to keep.
    coarsen_mode:
        Coarsening strategy for building lower pyramid levels.
        ``"mean"`` for continuous data, ``"mode"`` for categorical,
        ``"auto"`` to infer from the remapping method.
    min_valid_fraction:
        Minimum fraction of valid children for a parent cell to be
        valid.  Default ``0.5``.
    valid_fraction:
        Store ``<name>_valid_fraction`` next to the selected variables on
        every level: the fraction of valid finest-level cells under each
        cell, as float32, linked via the CF ``ancillary_variables``
        attribute.  Cell values are means over their valid area, so
        global or regional means over a coarse level need these weights,
        e.g. ``ds.sst.weighted(ds.sst_valid_fraction.fillna(0)).mean("cell")``.

        ``False`` (default) stores nothing.  ``True`` gives every cell
        variable a fraction of its full shape (correct for masks that
        change over time or height).  ``"static"`` stores one ``cell``-only
        fraction from the first slice along all other dimensions -- small,
        but only correct when the mask never changes (e.g. land/sea).
        A list of names selects variables (full shape); a mapping such as
        ``{"sst": "static", "ice": True}`` sets the shape per variable.
        Requires nested ordering.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    Returns
    -------
    dict[int, xarray.Dataset]
        Pyramid keyed by level.
    """
    if max_level is None:
        max_level = resolution_to_healpix_level(get_latlon_resolution(ds))

    is_nested = bool(kwargs.get("nest", True))
    if not is_nested and valid_fraction is not False:
        raise ValueError("valid_fraction requires nested ordering (nest=True).")

    pyramid: dict[int, xr.Dataset] = {}
    finest = ds.gd.regrid_to_healpix(max_level, **kwargs)
    pyramid[max_level] = finest

    if is_nested:
        fractions = resolve_valid_fraction(valid_fraction, finest)
        pyramid[max_level] = with_finest_fractions(finest, fractions, max_level)
        pyramid.update(
            coarse_levels(
                finest,
                max_level=max_level,
                min_level=min_level,
                coarsen_mode=resolve_coarsen_mode(finest, coarsen_mode),
                min_valid_fraction=min_valid_fraction,
                mode_kernel=_coarsen_array_mode,
                valid_fraction=fractions,
            )
        )
        return pyramid

    for level in range(max_level - 1, min_level - 1, -1):
        pyramid[level] = ds.gd.regrid_to_healpix(level, **kwargs)
    return pyramid
