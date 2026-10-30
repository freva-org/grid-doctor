"""Reading and writing HEALPix data (internal, work in progress).

- ``_zarr``: single-pass Zarr writes for whole pyramids.

The public entry point
[`save_pyramid`][grid_doctor.helpers.save_pyramid] still lives in
[`grid_doctor.helpers`][grid_doctor.helpers].
"""

from ._xarray import (
    get_common_coordinates,
    maybe_drop_coordinates,
    prepare_region_dataset,
    prepare_template,
    pyramid_to_multiscales_tree,
)
from ._zarr import (
    DeferredWrite,
    deferred_writes,
    encode_zarr_variable,
    get_filesystem,
    get_store,
    get_zarr_options,
    store_all,
)

__all__ = [
    "DeferredWrite",
    "deferred_writes",
    "encode_zarr_variable",
    "get_common_coordinates",
    "get_filesystem",
    "get_store",
    "get_zarr_options",
    "maybe_drop_coordinates",
    "prepare_region_dataset",
    "prepare_template",
    "pyramid_to_multiscales_tree",
    "store_all",
]
