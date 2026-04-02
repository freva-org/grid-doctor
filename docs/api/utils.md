# Helper utilities

This section provides supporting utilities for storage access, caching,
and chunk-size selection. These helpers are mainly intended to support
efficient IO and scalable data layout decisions.

## S3 Output handling

Once a pyramid has been created, it can be written to object storage for
cloud-native access and downstream analysis.

::: grid_doctor.save_pyramid_to_s3
    options:
      show_root_heading: true
      show_root_full_path: false

## Storage configuration

These utilities simplify access to S3-compatible object storage by
collecting and normalizing the options needed by downstream code.

::: grid_doctor.get_s3_options
    options:
      show_root_heading: true
      show_root_full_path: false

## Chunking

Choosing appropriate chunk sizes is important for balancing storage
efficiency and access performance. The function below helps determine a
chunk layout for a desired target store size.

::: grid_doctor.chunk_for_target_store_size
    options:
      show_root_heading: true
      show_root_full_path: false

## Caching

These helpers provide lightweight caching for repeatedly used datasets
and weight files to avoid unnecessary reopen or recomputation overhead.

::: grid_doctor.cached_open_dataset
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.cached_weights
    options:
      show_root_heading: true
      show_root_full_path: false
