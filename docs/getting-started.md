# Getting Started

## Installation

Install grid-doctor in editable mode from a local clone:

```console
python -m pip install git+https://github.com/freva-org/grid-doctor.git
```

## Dependencies
For very large grids grid-doctor uses
[ESMF](https://earthsystemmodeling.org/regrid/) for parallel offline
remapping. Since ESMF is not pip installable you have to install it via
conda-forge:

```console
mamba install -c conda-forge -y "esmf=*=mpi_openmpi_*" esmpy
```

Note: Check openmpi module versions on levante and pick an appropriate
`mpi_openmpi` version for installation.

## Writing a Conversion Script

Create a folder under `scripts/` with your script and a README:

```console
mkdir -p scripts/my-dataset
```

A minimal script looks like this:

```python
import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("my-dataset", "Convert my-dataset to HEALPix.")
# add script-specific arguments here …
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = gd.cached_open_dataset(["path/to/*.nc"])
weights_file = gd.cached_weights("~/weights/", nproc=4)
pyramid = gd.create_healpix_pyramid(ds, weights_path=weights_path)
gd.save_pyramid_to_s3(
    pyramid,
    f"s3://{args.s3_bucket}/my-dataset.zarr",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

!!! warning "Do not commit S3 credentials"
    Always read secrets from environment variables or a credentials file.

## Logging / Verbosity

Every script that uses `gd_cli.get_parser` automatically gets a `-v` flag.
Each repetition increases log verbosity:

| Flags | Level |
|-------|-------|
| *(none)* | WARNING |
| `-v` | INFO |
| `-vv` | DEBUG |

You can also control logging programmatically:

```python
from grid_doctor.log import set_level, increase, decrease

set_level("DEBUG")
increase()   # one step more verbose
decrease(2)  # two steps less verbose
```
