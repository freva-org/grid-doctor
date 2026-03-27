# Getting Started

## Installation

Install grid-doctor in editable mode from a local clone:

```console
git clone git@github.com:freva-org/grid-doctor.git
cd grid-doctor
python -m pip install -e .
```

### Dependencies

Core dependencies are installed automatically.  For **unstructured grid**
support (ICON, etc.) you also need [easygems](https://github.com/mpimet/easygems):

```console
pip install easygems
```

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
pyramid = gd.latlon_to_healpix_pyramid(ds)
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
