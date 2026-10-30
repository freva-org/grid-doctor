"""Names of every variable, dimension and attribute grid-doctor owns.

Nothing outside [`grid_doctor.metadata`][grid_doctor.metadata] should
spell these strings out.  Renaming an attribute means changing it here
and nowhere else.
"""

from __future__ import annotations

from typing import Final

METADATA_VERSION: Final = 1
"""Version of the metadata layout described by this package.

Bump it whenever the meaning or spelling of a stored attribute changes,
so that [`HealpixMetadata.from_dataset`][grid_doctor.metadata.HealpixMetadata.from_dataset]
can tell layouts apart instead of guessing."""

CF_VERSION: Final = "CF-1.13"
"""CF version that introduced the ``healpix`` grid mapping."""

# ---------------------------------------------------------------------
# Variables and dimensions
# ---------------------------------------------------------------------

CELL_DIM: Final = "cell"
"""Name of the HEALPix cell dimension and of its index coordinate."""

CRS_VAR: Final = "crs"
"""Name of the scalar CF grid mapping variable."""

LAT_VAR: Final = "latitude"
"""Name of the cell-centre latitude coordinate."""

LON_VAR: Final = "longitude"
"""Name of the cell-centre longitude coordinate."""

# ---------------------------------------------------------------------
# CF vocabulary
# ---------------------------------------------------------------------

GRID_MAPPING_NAME: Final = "healpix"
"""Value of ``grid_mapping_name`` for HEALPix grids."""

INDEX_STANDARD_NAME: Final = "healpix_index"
"""CF standard name identifying the HEALPix index coordinate.

Accepted in standard name table v94
(cf-convention/vocabularies#290).  A DGGS-generic replacement is under
discussion (cf-convention discussions #447); if one is adopted, only
this constant changes."""

# ---------------------------------------------------------------------
# grid-doctor attributes
# ---------------------------------------------------------------------

ATTR_PREFIX: Final = "grid_doctor_"
"""Prefix of every global attribute grid-doctor writes besides CF ones."""

METADATA_VERSION_ATTR: Final = f"{ATTR_PREFIX}metadata_version"
VERSION_ATTR: Final = f"{ATTR_PREFIX}version"
METHOD_ATTR: Final = f"{ATTR_PREFIX}method"
COARSENED_FROM_ATTR: Final = f"{ATTR_PREFIX}coarsened_from_level"
COARSEN_MODE_ATTR: Final = f"{ATTR_PREFIX}coarsen_mode"
MIN_VALID_FRACTION_ATTR: Final = f"{ATTR_PREFIX}min_valid_fraction"
SPARSE_ATTR: Final = f"{ATTR_PREFIX}sparse"
IMPLICIT_COORDS_ATTR: Final = f"{ATTR_PREFIX}implicit_coords"

LEVEL_MIN_ATTR: Final = f"{ATTR_PREFIX}level_min"
"""Coarsest level of a multiscales datatree (root group only)."""

LEVEL_MAX_ATTR: Final = f"{ATTR_PREFIX}level_max"
"""Finest level of a multiscales datatree (root group only)."""

SOURCE_ATTRS_ATTR: Final = "source_dataset_attrs"
"""Pyramid-wide attributes of a multiscales datatree (root group only)."""

OWNED_GLOBAL_ATTRS: Final = frozenset(
    {
        "Conventions",
        METADATA_VERSION_ATTR,
        VERSION_ATTR,
        METHOD_ATTR,
        COARSENED_FROM_ATTR,
        COARSEN_MODE_ATTR,
        MIN_VALID_FRACTION_ATTR,
        SPARSE_ATTR,
        IMPLICIT_COORDS_ATTR,
    }
)
"""Global attributes written by [`HealpixMetadata.apply`][grid_doctor.metadata.HealpixMetadata.apply].

``Conventions`` is listed because it is merged, not copied, so it must
not be passed on verbatim from another dataset."""
