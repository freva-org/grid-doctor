"""Reading and writing HEALPix data (internal, work in progress).

- ``_zarr``: single-pass Zarr writes for whole pyramids.
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
    LevelSpec,
    ManifestError,
    PyramidManifest,
    WriteTask,
)
from ._zarr import (
    DeferredWrite,
    deferred_writes,
    encode_zarr_variable,
    store_all,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ArraySpec",
    "DeferredWrite",
    "IncompleteStoreError",
    "LevelSpec",
    "ManifestError",
    "PyramidManifest",
    "StaleTaskError",
    "StoreExistsError",
    "StoreReport",
    "WriteTask",
    "check_pyramid_store",
    "deferred_writes",
    "encode_zarr_variable",
    "is_current",
    "is_task_done",
    "load_manifest",
    "store_all",
    "write_pyramid_task",
]
