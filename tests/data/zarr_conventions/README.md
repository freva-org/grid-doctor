# Vendored Zarr convention schemas

JSON schemas used by `tests/test_multiscales.py` to validate the root
attributes grid-doctor writes for multiscales pyramids.  They are copied
verbatim from the convention repositories (Apache-2.0 licensed) so the
tests run offline and notice when we drift from a pinned version.

| File | Source |
|:-----|:-------|
| `multiscales-v0.1.schema.json` | [zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales), tag `v0.1` (`9b78efa`) |
| `dggs-v1.schema.json` | [zarr-conventions/dggs](https://github.com/zarr-conventions/dggs), tag `v1` (`b8afc83`) |
| `dggs-pr25.schema.json` | [zarr-conventions/dggs#25](https://github.com/zarr-conventions/dggs/pull/25) (`ba7f43b`), not merged: adds the `multiscales` composition |

When a convention publishes a new version, replace the file, update the
pinned constants in `grid_doctor/metadata/_multiscales.py` and rerun the
tests.
