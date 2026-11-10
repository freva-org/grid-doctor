# Metadata API

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
      show_root_heading: true
      show_root_full_path: true
      members: false

::: grid_doctor.metadata.HealpixMetadata
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.metadata.HealpixGrid
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.metadata.Provenance
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.metadata.PyramidMetadata
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.metadata.PyramidLevel
    options:
      show_root_heading: true
      show_root_full_path: false
