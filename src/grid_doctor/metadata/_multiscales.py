"""Pyramid layout of a multiscales datatree, as Zarr conventions.

Encodes a HEALPix pyramid with two Zarr conventions on the root group:

- [``multiscales``](https://github.com/zarr-conventions/multiscales)
  (v0.1, pilot): a ``layout`` entry per level with ``asset``,
  ``derived_from`` and ``transform.scale``.
- [``dggs``](https://github.com/zarr-conventions/dggs) (v1): the grid of
  each level, embedded in its layout entry as proposed in
  zarr-conventions/dggs#25 (not merged yet): the source level carries
  the full ``dggs`` object, derived levels only what changes.

Both conventions are young, so everything specific to them lives in
this module: the pinned convention versions, how a layout entry is
spelled, and how scale factors are computed.  Each level group still
carries the CF metadata of [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata],
which tools that know neither convention can read.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import xarray as xr

from ._dataset import HealpixMetadata, has_healpix_metadata
from ._grid import HealpixGrid
from ._keys import CELL_DIM, GRID_MAPPING_NAME

# ---------------------------------------------------------------------
# Pinned conventions
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConventionRef:
    """An entry of the ``zarr_conventions`` attribute."""

    uuid: str
    name: str
    schema_url: str
    spec_url: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Return the entry as stored in ``zarr_conventions``."""
        return {
            "schema_url": self.schema_url,
            "spec_url": self.spec_url,
            "uuid": self.uuid,
            "name": self.name,
            "description": self.description,
        }


MULTISCALES_CONVENTION = ConventionRef(
    uuid="d35379db-88df-4056-af3a-620245f8e347",
    name="multiscales",
    schema_url="https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
    spec_url="https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md",
    description="Multiscale layout of zarr datasets",
)
"""The ``multiscales`` convention, pinned to its latest release (v0.1)."""

DGGS_CONVENTION = ConventionRef(
    uuid="7b255807-140c-42ca-97f6-7a1cfecdbc38",
    name="dggs",
    schema_url="https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    spec_url="https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    description="Discrete Global Grid Systems convention for zarr",
)
"""The ``dggs`` convention, pinned to its latest release (v1)."""

RESAMPLING_METHODS: Mapping[str, str] = {"mean": "average", "mode": "mode"}
"""grid-doctor coarsening modes and their ``resampling_method`` names."""


def level_group(level: int) -> str:
    """Return the datatree group (``multiscales`` asset) of *level*."""
    return f"level_{int(level)}"


# ---------------------------------------------------------------------
# Description of a pyramid
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PyramidLevel:
    """One level of a multiscales pyramid.

    Parameters
    ----------
    asset:
        Path of the level's group, relative to the root.
    grid:
        The level's HEALPix grid, ``None`` for a level without HEALPix
        metadata (it is then listed by path only).
    coordinate:
        Name of the cell-index coordinate, ``None`` when the level covers
        the full sphere without one (implicit coordinates).
    derived_from:
        Asset the level was coarsened from, if any.
    resampling_method:
        How it was coarsened (``"average"``, ``"mode"``, ...).
    """

    asset: str
    grid: HealpixGrid | None
    coordinate: str | None = CELL_DIM
    derived_from: str | None = None
    resampling_method: str | None = None


@dataclass(frozen=True, slots=True)
class PyramidMetadata:
    """Layout of a HEALPix pyramid stored as one datatree.

    Build it with [`from_pyramid`][grid_doctor.metadata.PyramidMetadata.from_pyramid]
    and write [`to_attrs`][grid_doctor.metadata.PyramidMetadata.to_attrs]
    to the root group; read it back with
    [`from_attrs`][grid_doctor.metadata.PyramidMetadata.from_attrs].

    Parameters
    ----------
    levels:
        The levels, finest first.
    """

    levels: tuple[PyramidLevel, ...]

    def __post_init__(self) -> None:
        """Check that every ``derived_from`` points to a described level."""
        assets = {level.asset: level for level in self.levels}
        if len(assets) != len(self.levels):
            raise ValueError("Pyramid levels must have unique assets.")
        for level in self.levels:
            if level.derived_from is None:
                continue
            source = assets.get(level.derived_from)
            if source is None or source.grid is None or level.grid is None:
                raise ValueError(
                    f"Level {level.asset!r} is derived from {level.derived_from!r}, "
                    "which is not a HEALPix level of this pyramid."
                )
            if source.grid.level <= level.grid.level:
                raise ValueError(f"Level {level.asset!r} must be coarser than {level.derived_from!r}.")

    # -----------------------------------------------------------------
    # Building
    # -----------------------------------------------------------------

    @classmethod
    def from_pyramid(cls, pyramid: Mapping[int, xr.Dataset]) -> PyramidMetadata:
        """Describe *pyramid* (``{level: dataset}``) from its levels' metadata.

        A level is derived from the level its provenance names in
        ``coarsened_from_level``, when that level is part of the
        pyramid.  Levels without HEALPix metadata are listed by path.
        """
        levels: list[PyramidLevel] = []
        for level in sorted(pyramid, reverse=True):
            ds = pyramid[level]
            asset = level_group(level)
            if not has_healpix_metadata(ds):
                levels.append(PyramidLevel(asset=asset, grid=None, coordinate=None))
                continue
            metadata = HealpixMetadata.from_dataset(ds)
            source = metadata.provenance.coarsened_from_level
            derived = source is not None and source != level and source in pyramid
            mode = metadata.provenance.coarsen_mode
            levels.append(
                PyramidLevel(
                    asset=asset,
                    grid=metadata.grid,
                    coordinate=CELL_DIM if CELL_DIM in ds.variables else None,
                    derived_from=level_group(source) if derived and source is not None else None,
                    resampling_method=(RESAMPLING_METHODS.get(mode, mode) if derived and mode else None),
                )
            )
        return cls(levels=tuple(levels))

    # -----------------------------------------------------------------
    # Writing
    # -----------------------------------------------------------------

    def to_attrs(self) -> dict[str, object]:
        """Return the ``zarr_conventions`` and ``multiscales`` attributes."""
        assets = {level.asset: level for level in self.levels}
        methods = {level.resampling_method for level in self.levels if level.derived_from is not None}
        shared_method = methods.pop() if len(methods) == 1 else None

        layout: list[dict[str, object]] = []
        for level in self.levels:
            entry: dict[str, object] = {"asset": level.asset}
            if level.derived_from is not None and level.grid is not None:
                source = assets[level.derived_from]
                assert source.grid is not None  # checked in __post_init__
                entry["derived_from"] = level.derived_from
                entry["transform"] = {"scale": [float(2 ** (source.grid.level - level.grid.level))]}
                if level.resampling_method is not None and shared_method is None:
                    entry["resampling_method"] = level.resampling_method
                entry["dggs"] = _changed_dggs(_dggs_object(source), _dggs_object(level))
            elif level.grid is not None:
                entry["dggs"] = _dggs_object(level)
            layout.append(entry)

        multiscales: dict[str, object] = {"layout": layout}
        if shared_method is not None:
            multiscales["resampling_method"] = shared_method
        conventions = [MULTISCALES_CONVENTION.to_dict()]
        if any(level.grid is not None for level in self.levels):
            conventions.append(DGGS_CONVENTION.to_dict())
        return {"zarr_conventions": conventions, "multiscales": multiscales}

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    @classmethod
    def from_attrs(cls, attrs: Mapping[Hashable, object]) -> PyramidMetadata:
        """Read the layout written by [`to_attrs`][grid_doctor.metadata.PyramidMetadata.to_attrs].

        Raises
        ------
        ValueError
            When *attrs* have no ``multiscales`` layout or describe a
            grid other than HEALPix.
        """
        multiscales = attrs.get("multiscales")
        if not isinstance(multiscales, Mapping) or not isinstance(multiscales.get("layout"), list):
            raise ValueError("Attributes have no `multiscales` layout.")
        entries: dict[str, Mapping[str, object]] = {}
        for entry in multiscales["layout"]:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("asset"), str):
                raise ValueError(f"Invalid multiscales layout entry: {entry!r}.")
            entries[entry["asset"]] = entry
        default_method = multiscales.get("resampling_method")

        resolved: dict[str, dict[str, object] | None] = {}

        def resolve(asset: str, seen: frozenset[str] = frozenset()) -> dict[str, object] | None:
            if asset in resolved:
                return resolved[asset]
            if asset in seen or asset not in entries:
                raise ValueError(f"Cannot resolve the dggs object of level {asset!r}.")
            entry = entries[asset]
            dggs = entry.get("dggs")
            source = entry.get("derived_from")
            if not isinstance(dggs, Mapping):
                result = None
            elif isinstance(source, str) and "name" not in dggs:
                base = resolve(source, seen | {asset})
                result = None if base is None else {**base, **dggs}
            else:
                result = dict(dggs)
            resolved[asset] = result
            return result

        levels = []
        for asset, entry in entries.items():
            dggs = resolve(asset)
            source = entry.get("derived_from")
            method = entry.get("resampling_method", default_method if source else None)
            levels.append(
                PyramidLevel(
                    asset=asset,
                    grid=None if dggs is None else _grid_from_dggs(dggs),
                    coordinate=None if dggs is None else _optional_str(dggs.get("coordinate")),
                    derived_from=source if isinstance(source, str) else None,
                    resampling_method=_optional_str(method),
                )
            )
        return cls(levels=tuple(levels))

    def asset(self, level: int) -> str:
        """Return the group path that holds HEALPix *level*."""
        for entry in self.levels:
            if entry.grid is not None and entry.grid.level == level:
                return entry.asset
        raise KeyError(f"Pyramid has no level {level}.")


# ---------------------------------------------------------------------
# dggs objects
# ---------------------------------------------------------------------


def _dggs_object(level: PyramidLevel) -> dict[str, object]:
    """Return the full ``dggs`` object of *level*."""
    grid = level.grid
    assert grid is not None
    obj: dict[str, object] = {
        "name": GRID_MAPPING_NAME,
        "refinement_level": grid.level,
        "indexing_scheme": grid.indexing_scheme,
        "spatial_dimension": CELL_DIM,
    }
    if grid.earth_radius is not None:
        obj["ellipsoid"] = {"name": "sphere", "radius": grid.earth_radius}
    if level.coordinate is not None:
        obj["coordinate"] = level.coordinate
        obj["compression"] = "none"
    return obj


def _changed_dggs(source: dict[str, object], derived: dict[str, object]) -> dict[str, object]:
    """Return the reduced ``dggs`` object of a derived level.

    Only changed keys are written, plus ``refinement_level``, which the
    reduced object requires.  When the derived level drops a key of its
    source (e.g. it has no coordinate), a reduced object cannot express
    that and the full object is returned.
    """
    if set(source) - set(derived):
        return derived
    changed = {key: value for key, value in derived.items() if source.get(key) != value}
    return {"refinement_level": derived["refinement_level"], **changed}


def _grid_from_dggs(dggs: Mapping[str, object]) -> HealpixGrid:
    if dggs.get("name") != GRID_MAPPING_NAME:
        raise ValueError(f"Not a HEALPix dggs object: name={dggs.get('name')!r}.")
    ellipsoid = dggs.get("ellipsoid")
    radius = ellipsoid.get("radius") if isinstance(ellipsoid, Mapping) else None
    return HealpixGrid(
        level=dggs.get("refinement_level"),  # type: ignore[arg-type]
        indexing_scheme=str(dggs.get("indexing_scheme", "")),  # type: ignore[arg-type]
        earth_radius=radius,
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
