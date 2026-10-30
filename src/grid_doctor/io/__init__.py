"""Reading and writing HEALPix data (internal, work in progress).

- ``_zarr``: single-pass Zarr writes for whole pyramids.
- ``_xarray``: preparing templates, regions and multiscales datatrees
  for writing.
- ``_manifest``: serialisable description of an initialised pyramid
  store and its chunk-aligned work units (pure data, no I/O).
- ``_fanout``: init-time manifest persistence, per-task worker writes
  and the completeness check.

The public entry points
[`save_pyramid`][grid_doctor.helpers.save_pyramid] and
[`init_pyramid_store`][grid_doctor.init_pyramid_store] still
live in [`grid_doctor.helpers`][grid_doctor.helpers]; the fan-out names
are re-exported from ``grid_doctor``.
"""

from ._fanout import (
    IncompleteStoreError,
    StaleTaskError,
    StoreExistsError,
    StoreReport,
    check_pyramid_store,
    is_current,
    is_task_done,
    load_manifest,
    write_pyramid_task,
)
from ._manifest import (
    MANIFEST_SCHEMA_VERSION,
    ArraySpec,
    ArraySpecDict,
    LevelSpec,
    LevelSpecDict,
    ManifestDict,
    ManifestError,
    PyramidManifest,
    TaskEntryDict,
    WriteTask,
    WriteTaskDict,
)
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
    "MANIFEST_SCHEMA_VERSION",
    "ArraySpec",
    "ArraySpecDict",
    "DeferredWrite",
    "IncompleteStoreError",
    "LevelSpec",
    "LevelSpecDict",
    "ManifestDict",
    "ManifestError",
    "PyramidManifest",
    "StaleTaskError",
    "StoreExistsError",
    "StoreReport",
    "TaskEntryDict",
    "WriteTask",
    "WriteTaskDict",
    "check_pyramid_store",
    "deferred_writes",
    "encode_zarr_variable",
    "get_common_coordinates",
    "get_filesystem",
    "get_store",
    "get_zarr_options",
    "is_current",
    "is_task_done",
    "load_manifest",
    "maybe_drop_coordinates",
    "prepare_region_dataset",
    "prepare_template",
    "pyramid_to_multiscales_tree",
    "store_all",
    "write_pyramid_task",
]
