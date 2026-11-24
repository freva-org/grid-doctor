"""Write HEALPix pyramids to Zarr stores, in one pass or as parallel tasks.

- ``_save``: [`save_pyramid`][grid_doctor.save_pyramid] and
  [`init_pyramid_store`][grid_doctor.init_pyramid_store].
- ``_options``: S3 options and chunk sizes for HEALPix stores.
- ``_zarr``: single-pass Zarr writes for whole pyramids.
- ``_xarray``: preparing templates, regions and multiscales datatrees
  for writing.
- ``_manifest``: serialisable description of an initialised pyramid
  store and its chunk-aligned work units (pure data, no I/O).
- ``_fanout``: init-time manifest persistence, per-task worker writes
  and the completeness check.

The public names are re-exported from ``grid_doctor``.
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
from ._options import chunk_for_target_store_size, get_s3_options
from ._save import init_pyramid_store, save_pyramid

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ArraySpec",
    "ArraySpecDict",
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
    "chunk_for_target_store_size",
    "get_s3_options",
    "init_pyramid_store",
    "is_current",
    "is_task_done",
    "load_manifest",
    "save_pyramid",
    "write_pyramid_task",
]
