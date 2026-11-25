# API reference

Everything listed here is imported from `grid_doctor` (`import grid_doctor
as gd`), unless the page names a submodule.

| Page | What it covers |
|------|----------------|
| [Remapping](remap.md) | `regrid_to_healpix`, `compute_healpix_weights`, `apply_weight_file` |
| [Weights store](weights.md) | `cached_weights`, `WeightsStore` and how weight files are named and shared |
| [Pyramids](pyramid.md) | `create_healpix_pyramid`, `coarsen_healpix`, resolution helpers |
| [Point binning](swath.md) | `bin_to_healpix`, `sparse_to_dense` |
| [Writing stores](io.md) | `save_pyramid`, `get_s3_options`, `chunk_for_target_store_size` |
| [Parallel writes](parallel.md) | `init_pyramid_store`, `write_pyramid_task`, `check_pyramid_store` and the manifest |
| [Region selection](select.md) | `select_bbox`, `select_cone`, `select_cells`, `attach_cell_coords` |
| [Metadata](metadata.md) | `grid_doctor.metadata`: the CF attributes and HEALPix metadata of the outputs |
| [Configuration](config.md) | `grid_doctor.config`: ellipsoid and weights store settings |
| [Types](types.md) | `grid_doctor.types`: type aliases for annotating your own code |
| [Command line](cli.md) | `grid_doctor.cli` and `setup_logging` |
