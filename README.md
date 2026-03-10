# Grid Doctor HEALs your Grids

> [!NOTE]
> This is a scripting solution for a proof of concept. An operational ready
> approach will follow. For adding code for specific datasets please add
> your script solution into the `scripts/<yourname>` folder.


## Installation

```console
git clone git@github.com:freva-org/grid-doctor.git
cd grid-doctor
python -m pip install -e .
```

## Quick Start

### Structured Grids (ERA5, CMIP, …)

```python
import grid_doctor as gd

ds = gd.cached_open_dataset(["path/to/*.nc"])
pyramid = gd.latlon_to_healpix_pyramid(ds)
gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/dataset.zarr",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

### Unstructured Grids (ICON)

For ICON and other triangular-mesh models, pre-compute and cache the
Delaunay interpolation weights:

```python
import grid_doctor as gd

grid_ds = gd.cached_open_dataset(["ICON_grid.nc"])
max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
weights = gd.cached_weights(grid_ds, level=max_level)

ds = gd.cached_open_dataset(["icon_data_*.grb"])
ds = ds.rename_dims({"values": "cell"}).chunk({"cell": -1})

pyramid = gd.latlon_to_healpix_pyramid(ds, max_level=max_level, weights=weights)
gd.save_pyramid_to_s3(pyramid, "s3://my-bucket/icon.zarr", s3_options=...)
```


## Writing a Conversion Script

Create a folder under `scripts/` and add your script:

```console
mkdir -p scripts/<yourname>
```

A minimal script using the built-in CLI helpers:

```python
import grid_doctor as gd
import grid_doctor.cli as gd_cli
from data_portal_worker.rechunker import ChunkOptimizer

parser = gd_cli.get_parser("my-dataset", "Convert my-dataset to HEALPix.")
parser.add_argument("--variables", nargs="*", default=["t_2m"])
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = gd.cached_open_dataset(["path/to/*.nc"])
pyramid = gd.latlon_to_healpix_pyramid(ds)

opt = ChunkOptimizer()
chunked = {lvl: opt.apply(d) for lvl, d in pyramid.items()}

gd.save_pyramid_to_s3(
    chunked,
    f"s3://{args.s3_bucket}/my-dataset.zarr",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

Run with verbosity:

```console
python scripts/my-dataset/convert.py my-bucket -vv
```

> [!IMPORTANT]
> Please add a descriptive README about what your script is trying to achieve.
> Document any problems you ran into.

> [!CAUTION]
> DO NOT commit S3 keys or secrets to this repository. Use environment
> variables or a credentials file.


## Building Documentation

```console
pip install tox
tox -e docs          # build to site/
tox -e docs-serve    # live preview at http://127.0.0.1:8000
```


## Type Checking

```console
tox -e type-check
```


## Issues

As this is still very much work in progress it is very likely that you will
run into problems. Please note any problems in the `README.md` file
for your dataset folder. Feel free to submit PRs if there are any issues
with the `DatasetAggregator` or `ChunkOptimizer` classes. If you don't feel
comfortable with submitting PRs you can file an issue report
[here](https://github.com/freva-org/freva-nextgen/issues).
