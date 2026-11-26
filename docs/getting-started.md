# Getting started

## Installation

grid-doctor is installed from GitHub:

```console
python -m pip install git+https://github.com/freva-org/grid-doctor.git
```

For GPU-accelerated weight application, add the `gpu` extra, which
installs [CuPy](https://cupy.dev/):

```console
python -m pip install "grid-doctor[gpu] @ git+https://github.com/freva-org/grid-doctor.git"
```

### ESMF

Remapping weights are generated with
[ESMF](https://earthsystemmodeling.org/regrid/): in memory through its
Python bindings (ESMPy), or for large grids with the parallel
`ESMF_RegridWeightGen` command under MPI. ESMF is not available from PyPI;
install it from conda-forge:

```console
mamba install -c conda-forge "esmf=*=mpi_openmpi_*" esmpy
```

On an HPC system, pick the MPI build that matches the system's MPI
library. ESMF is only needed to *generate* weights. Applying existing
weight files, binning point data, coarsening, selecting regions and
writing stores work without it.

## Your first pyramid

```python
import xarray as xr

import grid_doctor as gd

ds = xr.open_mfdataset("path/to/*.nc")
pyramid = ds.gd.create_healpix_pyramid()
gd.save_pyramid(pyramid, "/path/to/output", mode="w")
```

The finest level follows from the resolution of the source grid; pass
`max_level=` to choose it yourself. `save_pyramid` writes one Zarr store
per level, `level_0.zarr`, `level_1.zarr`, …, below `/path/to/output`. The weights are computed on
the first run and reused from the [weights store](api/weights.md) on
every later run with the same grid. Set `GRID_DOCTOR_WEIGHTS_STORE` to
share them with others, see [Configuration](api/config.md).

## Writing to S3

Pass an `s3://` URL and the storage options.
[`get_s3_options`][grid_doctor.get_s3_options] reads the access and secret
key from a JSON file with the fields `accessKey` and `secretKey`:

```python
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/my-dataset",
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

!!! warning "Keep credentials out of scripts"
    Read secrets from a credentials file or the environment, and never
    commit them to a repository.

## Conversion scripts

[`grid_doctor.cli.get_parser`][grid_doctor.cli.get_parser] returns an
argument parser with the options `--s3-bucket`, `--s3-endpoint`,
`--s3-credentials-file` and `-v` already defined:

```python
import xarray as xr

import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("my-dataset", "Convert my-dataset to HEALPix.")
parser.add_argument("files", nargs="+")
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = xr.open_mfdataset(args.files)
pyramid = ds.gd.create_healpix_pyramid()
gd.save_pyramid(
    pyramid,
    f"s3://{args.s3_bucket}/my-dataset",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

```console
python convert.py --s3-bucket my-bucket --s3-endpoint https://s3.example.org data/*.nc -vv
```

## Logging

Each `-v` on the command line raises the log level:

| Flags | Level |
|-------|-------|
| *(none)* | WARNING |
| `-v` | INFO |
| `-vv` | DEBUG |

Outside a script, set the level directly:

```python
import grid_doctor as gd

gd.setup_logging(level="DEBUG")  # or verbosity=2
```
