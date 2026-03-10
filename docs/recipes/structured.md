# Structured Grids (ERA5, CMIP)

This recipe converts a regular lat/lon dataset (e.g. ERA5 or CMIP6
output) to a HEALPix pyramid and uploads it to S3.

## Minimal Example

```python
import grid_doctor as gd

# 1. Open
ds = gd.cached_open_dataset(["era5_2m_temperature_*.nc"])

# 2. Convert
pyramid = gd.latlon_to_healpix_pyramid(ds)

# 3. Upload
gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/era5/2t.zarr",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

grid-doctor automatically detects the grid resolution and picks a
matching HEALPix level.  To override:

```python
pyramid = gd.latlon_to_healpix_pyramid(ds, max_level=7)
```

## Using the Data-Portal Rechunker

For production uploads you typically want optimised chunk sizes.  The
`data_portal_worker.rechunker.ChunkOptimizer` can be applied to each
level before writing:

```python
from data_portal_worker.rechunker import ChunkOptimizer

opt = ChunkOptimizer()
chunked = {lvl: opt.apply(ds) for lvl, ds in pyramid.items()}
gd.save_pyramid_to_s3(chunked, ...)
```

## Full CLI Script

Scripts under `scripts/<name>/convert.py` use the shared CLI parser:

```python
import os
import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("era5", "Convert ERA5 to HEALPix.")
parser.add_argument("files", nargs="+")
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = gd.cached_open_dataset(args.files)
pyramid = gd.latlon_to_healpix_pyramid(ds)
gd.save_pyramid_to_s3(
    pyramid,
    f"s3://{args.s3_bucket}/era5.zarr",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

Run it:

```console
python scripts/era5/convert.py my-bucket /pool/data/era5/*.nc -vv
```
