# Writing stores

Write HEALPix pyramids to Zarr stores on local disk or S3, and choose
storage options and chunk sizes for them (`grid_doctor.io`).

## Writing a pyramid

::: grid_doctor.save_pyramid

To split the write across many jobs (one array-job element per time
range, say), see [Parallel writes](parallel.md).

## S3 options

::: grid_doctor.get_s3_options

## Chunking

Chunk sizes balance the number of objects in a store against the amount
of data read for a small selection.

::: grid_doctor.chunk_for_target_store_size
