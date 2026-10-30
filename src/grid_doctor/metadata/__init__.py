"""Metadata of grid-doctor datasets, written and read in one place.

Every attribute, coordinate name and grid mapping grid-doctor stores is
defined in this package.  Code that produces data builds a
[`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata] and calls
[`apply`][grid_doctor.metadata.HealpixMetadata.apply]; code that reads
data calls
[`HealpixMetadata.from_dataset`][grid_doctor.metadata.HealpixMetadata.from_dataset].
Changing the stored layout therefore never touches the write sites.

- ``_grid``: [`HealpixGrid`][grid_doctor.metadata.HealpixGrid], the
  grid itself (level, indexing scheme, figure of the earth).
- ``_provenance``: [`Provenance`][grid_doctor.metadata.Provenance],
  how the values were produced.
- ``_cf``: the CF-1.13 ``healpix`` grid mapping spelling of a grid.
- ``_dataset``: [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata],
  which combines the above and applies it to a dataset.
- ``_keys``: every stored name.

The written layout is CF-1.13: a scalar ``crs`` grid mapping variable
(``grid_mapping_name = "healpix"``, ``indexing_scheme``,
``refinement_level``), an integer ``cell`` index coordinate with
``standard_name = "healpix_index"``, and ``grid_mapping = "crs"`` on
every variable along ``cell``.  ``xdggs.decode(ds, convention="cf")``
reads it.
"""

from ._cf import grid_mapping_attrs, merge_conventions
from ._dataset import HealpixMetadata, has_healpix_metadata
from ._grid import INDEXING_SCHEMES, MAX_LEVEL, HealpixGrid, IndexingScheme
from ._keys import (
    CELL_DIM,
    CF_VERSION,
    CRS_VAR,
    INDEX_STANDARD_NAME,
    LAT_VAR,
    LON_VAR,
    METADATA_VERSION,
)
from ._provenance import Provenance, variable_method, variable_method_attrs

__all__ = [
    "CELL_DIM",
    "CF_VERSION",
    "CRS_VAR",
    "INDEXING_SCHEMES",
    "INDEX_STANDARD_NAME",
    "LAT_VAR",
    "LON_VAR",
    "MAX_LEVEL",
    "METADATA_VERSION",
    "HealpixGrid",
    "HealpixMetadata",
    "IndexingScheme",
    "Provenance",
    "grid_mapping_attrs",
    "has_healpix_metadata",
    "merge_conventions",
    "variable_method",
    "variable_method_attrs",
]
