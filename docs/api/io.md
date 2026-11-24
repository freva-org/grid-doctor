# Writing stores

Write HEALPix pyramids to Zarr stores on S3 or local disk, and choose
storage options and chunk sizes for them.

## S3 Output handling

Once a pyramid has been created, it can be written to object storage for
cloud-native access and downstream analysis.

::: grid_doctor.save_pyramid
    options:
      show_root_heading: true
      show_root_full_path: false

To split the write across many jobs (one array-job element per time
range, say), see [Parallel writes](parallel.md).

## Storage configuration

These utilities simplify access to S3-compatible object storage by
collecting and normalising the options needed by downstream code.

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
