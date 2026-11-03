"""Writing and reading all grid-doctor metadata of a dataset in one place."""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ._cf import (
    INDEX_ATTRS,
    LAT_ATTRS,
    LON_ATTRS,
    grid_from_mapping,
    grid_mapping_attrs,
    is_healpix_grid_mapping,
    merge_conventions,
)
from ._grid import HealpixGrid
from ._keys import (
    CELL_DIM,
    CRS_VAR,
    IMPLICIT_COORDS_ATTR,
    LAT_VAR,
    LON_VAR,
    METADATA_VERSION,
    METADATA_VERSION_ATTR,
    OWNED_GLOBAL_ATTRS,
    SPARSE_ATTR,
)
from ._provenance import Provenance


@dataclass(frozen=True, slots=True)
class HealpixMetadata:
    """Everything grid-doctor records about a HEALPix dataset.

    Write sites build one of these and call
    [`apply`][grid_doctor.metadata.HealpixMetadata.apply]; readers call
    [`from_dataset`][grid_doctor.metadata.HealpixMetadata.from_dataset].
    Neither ever touches attribute names, so the stored layout can
    change here without touching any code that produces or consumes
    data.

    Parameters
    ----------
    grid:
        The HEALPix grid.
    provenance:
        How the values were produced.
    sparse:
        The ``cell`` coordinate holds a subset of global cell indices
        (output of the ``select_*`` functions and of
        ``bin_to_healpix(..., dense=False)``).  Otherwise the dataset
        is dense: position along ``cell`` equals the cell index.
    implicit_coords:
        A dense store written without ``cell``/``latitude``/``longitude``
        arrays because they are implied by position (see
        ``save_pyramid(write_coords=...)``).  Such stores are not
        identifiable by CF's ``healpix_index`` standard name.

    Examples
    --------
    >>> meta = HealpixMetadata(HealpixGrid(level=2), Provenance(method="nearest"))
    >>> ds = meta.apply(ds)  # doctest: +SKIP
    >>> HealpixMetadata.from_dataset(ds).grid.level  # doctest: +SKIP
    2
    """

    grid: HealpixGrid
    provenance: Provenance = field(default_factory=Provenance)
    sparse: bool = False
    implicit_coords: bool = False

    def __post_init__(self) -> None:
        """Reject contradictory layouts."""
        if self.sparse and self.implicit_coords:
            raise ValueError("A sparse dataset needs its `cell` coordinate; " "it cannot have implicit coordinates.")

    def replace(self, **changes: object) -> HealpixMetadata:
        """Return a copy with *changes* applied (see ``dataclasses.replace``)."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    # -----------------------------------------------------------------
    # Writing
    # -----------------------------------------------------------------

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """Return *ds* with all grid-doctor metadata written.

        Writes

        - a scalar ``crs`` grid mapping coordinate (replacing any
          existing one),
        - CF attributes on the ``cell`` index coordinate and the
          ``latitude``/``longitude`` coordinates, where present,
        - ``grid_mapping = "crs"`` on every variable along ``cell``,
        - the global ``Conventions``, provenance and layout attributes.

        Global attributes owned by grid-doctor are rewritten from this
        object, so stale values copied from an input dataset never
        survive.  Coordinate *values* are left to the caller.

        Raises
        ------
        ValueError
            When the ``cell`` dimension or coordinate contradicts the
            grid or the layout.
        """
        self._check_layout(ds)
        crs = xr.DataArray(np.float64(0.0), attrs=grid_mapping_attrs(self.grid))
        # A shallow copy owns its attribute dicts, so *ds* is left untouched.
        result = ds.copy(deep=False).drop_vars(CRS_VAR, errors="ignore")
        result = result.assign_coords({CRS_VAR: crs})

        coord_attrs = ((CELL_DIM, INDEX_ATTRS), (LAT_VAR, LAT_ATTRS), (LON_VAR, LON_ATTRS))
        for coord, attrs in coord_attrs:
            if coord in result.coords:
                result[coord].attrs.update(attrs)

        for var in result.data_vars.values():
            if CELL_DIM in var.dims:
                var.attrs["grid_mapping"] = CRS_VAR

        global_attrs: dict[Hashable, object] = {
            key: value for key, value in result.attrs.items() if key not in OWNED_GLOBAL_ATTRS
        }
        global_attrs["Conventions"] = merge_conventions(ds.attrs.get("Conventions"))
        global_attrs[METADATA_VERSION_ATTR] = METADATA_VERSION
        global_attrs.update(self.provenance.to_attrs())
        if self.sparse:
            global_attrs[SPARSE_ATTR] = 1
        if self.implicit_coords:
            global_attrs[IMPLICIT_COORDS_ATTR] = 1
        result.attrs = global_attrs
        return result

    def _check_layout(self, ds: xr.Dataset) -> None:
        """Check that *ds*'s ``cell`` dimension fits grid and layout."""
        if CELL_DIM not in ds.dims:
            return
        size = ds.sizes[CELL_DIM]
        if CELL_DIM in ds.coords:
            if self.implicit_coords:
                raise ValueError(f"implicit_coords=True but the dataset has a `{CELL_DIM}` coordinate.")
            if not np.issubdtype(ds[CELL_DIM].dtype, np.integer):
                raise ValueError(
                    f"The `{CELL_DIM}` coordinate must hold integer HEALPix indices, "
                    f"got dtype {ds[CELL_DIM].dtype}."
                )
        elif self.sparse:
            raise ValueError(f"A sparse dataset needs a `{CELL_DIM}` coordinate.")
        if self.sparse:
            if size > self.grid.npix:
                raise ValueError(
                    f"Sparse dataset has {size} cells, more than the {self.grid.npix} of level {self.grid.level}."
                )
        elif size != self.grid.npix:
            raise ValueError(
                f"Dense dataset has {size} cells but level {self.grid.level} "
                f"has {self.grid.npix}; pass sparse=True for subsets."
            )

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> HealpixMetadata:
        """Read the metadata written by [`apply`][grid_doctor.metadata.HealpixMetadata.apply].

        Only the CF grid mapping is required, so CF HEALPix datasets
        written by other tools (e.g. ``xdggs``) are read as dense,
        with unknown provenance.

        Raises
        ------
        ValueError
            When *ds* has no HEALPix grid mapping variable, or was
            written by a newer metadata layout than this grid-doctor
            understands.
        """
        version = _int_attr(ds.attrs, METADATA_VERSION_ATTR)
        if version is not None and version > METADATA_VERSION:
            raise ValueError(
                f"Dataset uses metadata version {version}; this grid-doctor "
                f"understands up to {METADATA_VERSION}.  Upgrade grid-doctor."
            )
        return cls(
            grid=grid_from_mapping(_grid_mapping_attrs(ds)),
            provenance=Provenance.from_attrs(ds.attrs),
            sparse=_flag(ds.attrs, SPARSE_ATTR),
            implicit_coords=_flag(ds.attrs, IMPLICIT_COORDS_ATTR),
        )


def _grid_mapping_attrs(ds: xr.Dataset) -> Mapping[Hashable, object]:
    """Return the attributes of *ds*'s HEALPix grid mapping variable."""
    if CRS_VAR in ds.variables and is_healpix_grid_mapping(ds[CRS_VAR].attrs):
        return ds[CRS_VAR].attrs
    candidates = [var.attrs for var in ds.variables.values() if is_healpix_grid_mapping(var.attrs)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "Dataset has no HEALPix grid mapping variable " "(a variable with grid_mapping_name = 'healpix')."
        )
    raise ValueError(f"Dataset has {len(candidates)} HEALPix grid mapping variables; expected one.")


def _int_attr(attrs: Mapping[Hashable, object], key: str) -> int | None:
    value = attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Attribute `{key}` must be an integer, got {value!r}.")
    return int(value)


def _flag(attrs: Mapping[Hashable, object], key: str) -> bool:
    return bool(_int_attr(attrs, key))


def has_healpix_metadata(ds: xr.Dataset) -> bool:
    """Whether *ds* carries a HEALPix grid mapping variable."""
    return any(is_healpix_grid_mapping(var.attrs) for var in ds.variables.values())
