# Grid Doctor HEALs your Grids
<p align="center">
  <img src="docs/assets/logo-512.png" alt="Logo" width="600"><br>
  <img
      src="https://img.shields.io/badge/grid--doctor-Documentation-green?logo=read-the-docs&amp;logoColor=white"
      alt="Documentation"
    >
</p>

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
For GPU support use

```console
python -m pip install -e .[gpu]
```


For remapping of large grids you should install
[ESMF](https://earthsystemmodeling.org/regrid/) through ocnda-forge.

```console
mamba install -c conda-forge -y "esmf=*=mpi_openmpi_*" esmpy
```

## Quick Start

```python
import grid_doctor as gd

ds = gd.cached_open_dataset(["path/to/*.nc"])
max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
weights_dir="/scratch/{user[0]}/{user}/grid-doctor/weights"\
    .format(user=getuser(), level=level)
gd.cached_weights(
    ds,
    level=max_level,
    prefer_offline=True,
    cache_path=weights_path
)
pyramid = gd.create_healpix_pyramid(
    ds,
    weights_path=weights_dir,
    max_level=max_level
)
gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/dataset.zarr",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

## 🏥 Grid Rehab Progress
How are our patients doing? Every dataset starts broken and leaves HEALed.
If your dataset is still 😢, it needs a doctor — that could be you.
Claim a patient, write a script, and turn that frown into 😎.

| | Meaning |
|:-:|---------|
| 😢 | Not started |
| 🩹 | In treatment |
| 😎 | HEALed |

| Dataset | Uploaded to S3 | Script Submitted |
|---------|:--------------:|:----------------:|
| ICON-DREAM  | 😎 | 😎 |
| EERIE | 😎 | 😎 |
| ERA5 | 😎 | 😎 |
| CMIP6 | 🩹 | 😢 |
| NextGEMS | 😎 | 😎 |
| ICDC     | 😎 | 😢 |
| ORCHESTRA | 😎 | 😎 |
| PalMod | 😢 | 😢 |
| Dyamond| 😎 | 😢 |
> [!TIP]
> To claim a dataset, open a PR adding your script to `scripts/<dataset>/`
> and update this table. See [Getting Started](#writing-a-conversion-script)
> for the template.

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
pyramid = gd.create_healpix_pyramid(ds)
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
