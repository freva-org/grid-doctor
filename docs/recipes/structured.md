# Structured Grids (ERA5, CMIP)

This recipe converts a regular lat/lon dataset (e.g. ERA5 or CMIP6
output) to a HEALPix pyramid and uploads it to S3.

## Minimal Example

A minimal example involves opening the data, creating the weights file for this
data and applying *conservative* remapping:

```python
from getpass import getuser
import grid_doctor as gd

# 1. Open
ds = gd.cached_open_dataset(["era5_2m_temperature_*.nc"])

# 2. Weights file
weights_dir = Path(
    "/scratch/{user[0]}/{user}/healpix-weights".format(user=getuser())
)
weights_file = gd.cached_weights(
    ds,
    weights_path=weights_dir,
    nproc=4,
    prefer_offline=True
)
# 2. Convert
pyramid = gd.create_healpix_pyramid(ds)

# 3. Upload
gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/era5/2t",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

grid-doctor automatically detects the grid resolution and picks a
matching HEALPix level.  To override:

```python
pyramid = gd.create_healpix_pyramid(ds, max_level=7)
```

## Full CLI Script

Scripts under `scripts/<name>/convert.py` use the shared CLI parser:

```python
import os
from pathlib import Path

from getpass import getuser

import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("era5", "Convert ERA5 to HEALPix.")
parser.add_argument("files", nargs="+")
parser.add_argument(
   "weights-dir",
    type=Path,
    help="Path to to store weights."
    defualt="/scratch/{user[0]}/{user}/healpix-weights/".format(user=getuser())
    )
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = gd.cached_open_dataset(args.files)
weights_file = gd.cached_weights(
    ds
    weights_path=args.weights_dir,
    nproc=4,
    prefer_offline=True
)
pyramid = gd.create_healpix_pyramid(ds)
gd.save_pyramid_to_s3(
    pyramid,
    f"s3://{args.s3_bucket}/era5",
    s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
)
```

Run it:

```console
python scripts/era5/convert.py my-bucket /pool/data/era5/*.nc -vv
```
