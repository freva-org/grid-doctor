# Metadata

Every attribute, coordinate name and grid mapping that grid-doctor
stores is defined in `grid_doctor.metadata`.  Code that produces data
builds a [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata] and
calls `apply`; code that reads data calls `from_dataset`.  The stored
layout follows the CF-1.13 `healpix` grid mapping; see
[Technical decisions](../technical-decisions.md#output-metadata-and-crs-convention)
for the full description.

```python
from grid_doctor.metadata import HealpixGrid, HealpixMetadata, Provenance

meta = HealpixMetadata(HealpixGrid(level=8), Provenance(method="conservative"))
ds = meta.apply(ds)

HealpixMetadata.from_dataset(ds).grid.level  # 8
```

::: grid_doctor.metadata
    options:
      show_root_full_path: true
      members: false

::: grid_doctor.metadata.HealpixMetadata

::: grid_doctor.metadata.HealpixGrid

::: grid_doctor.metadata.Provenance

::: grid_doctor.metadata.PyramidMetadata

::: grid_doctor.metadata.PyramidLevel

## Helpers

::: grid_doctor.metadata.has_healpix_metadata

::: grid_doctor.metadata.variable_method

::: grid_doctor.metadata.variable_method_attrs

::: grid_doctor.metadata.grid_mapping_attrs

::: grid_doctor.metadata.merge_conventions

::: grid_doctor.metadata.level_group

::: grid_doctor.metadata.IndexingScheme
