# Structured grids (ERA5, CMIP)

This recipe converts data on a regular or curvilinear lat/lon grid, such
as ERA5 or CMIP6 output, to a HEALPix pyramid and writes it to S3.

## Minimal example

Open the data, get the weights for its grid and build the pyramid with
*conservative* remapping, the default:

```python
import xarray as xr

import grid_doctor as gd

# 1. Open
ds = xr.open_mfdataset("era5_2m_temperature_*.nc")

# 2. Weights: computed once, then taken from the weights store
level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
weights_file = gd.cached_weights(
    ds,
    level=level,
    store="/path/to/weights",
    nproc=4,
    prefer_offline=True,
)

# 3. Convert
pyramid = ds.gd.create_healpix_pyramid(max_level=level, weights_path=weights_file)

# 4. Write
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/era5/2t",
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

- `level` is the HEALPix level whose cell spacing matches the grid; pass
  another value to `max_level` (and `level`) to choose it yourself.
- `store` is a local directory, possibly shared with others. Without it,
  the default store is used, see [Weights store](../api/weights.md).
- `prefer_offline=True` computes large conservative weights with the
  parallel `ESMF_RegridWeightGen` (with `nproc` MPI processes) instead
  of in memory.

## Masked fields (ocean, sea ice)

For variables with missing values, store the valid fraction of each cell
so that averages over coarse levels stay consistent with the finest one
(see [Coarsening and averaging masked fields](averaging.md) for how to use them):

```python
pyramid = ds.gd.create_healpix_pyramid(
    max_level=level,
    weights_path=weights_file,
    # land-sea masked SST: one cell-only fraction is enough;
    # sea ice changes over time and needs the full shape
    valid_fraction={"sst": "static", "siconc": True},
)
```

Variables without missing values do not need fractions.

## Command-line script

The same conversion as a script, using the parser from
[`grid_doctor.cli`](../api/cli.md):

```python
import xarray as xr

import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("era5", "Convert ERA5 to HEALPix.")
parser.add_argument("files", nargs="+")
parser.add_argument("--weights-store", help="Local directory of the weights store.")
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = xr.open_mfdataset(args.files)
weights_file = gd.cached_weights(ds, store=args.weights_store, nproc=4, prefer_offline=True)
pyramid = ds.gd.create_healpix_pyramid(weights_path=weights_file)
gd.save_pyramid(
    pyramid,
    f"s3://{args.s3_bucket}/era5",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

```console
python convert.py --s3-bucket my-bucket --s3-endpoint https://s3.example.org \
    --weights-store /path/to/weights era5_*.nc -vv
```
