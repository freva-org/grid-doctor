# Grid Doctor HEALs your Grids
<p align="center">
  <img src="docs/assets/logo-512.png" alt="Logo" width="200"><br>
  <a href="https://grid-doctor.org/">
  <img
    src="https://img.shields.io/badge/grid--doctor-Documentation-green?logo=readthedocs&logoColor=white"
    alt="Documentation"
  >
</a>
</p>

This repository contains the code for regridding data onto HEALPix grids.
Example scripts can be found in the `scripts/` directory.
For full documentation, see https://grid-doctor.org/.

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


Generating remapping weights needs
[ESMF](https://earthsystemmodeling.org/regrid/), which is installed
through conda-forge.

```console
mamba install -c conda-forge -y "esmf=*=mpi_openmpi_*" esmpy
```

## Developing the documentation

The documentation is built with [Zensical](https://zensical.org) from the
`docs/` directory and configured in `mkdocs.yml`.

```console
tox -e docs        # spell check and strict build into site/
tox -e docs-serve  # serve on http://localhost:8000 with live reload
```

Add a page as a Markdown file under `docs/` and list it in the `nav` of
`mkdocs.yml`. Images and other static files go under `docs/assets/`.
API reference pages render docstrings with
[mkdocstrings](https://mkdocstrings.github.io/): a line such as
`::: grid_doctor.save_pyramid` inserts the documentation of that object.

### Pages shared with the Waterpark docs

`docs/shared/technical-decisions.md` and `docs/shared/high-level-access.md`
are also published in the Waterpark documentation, which vendors
`docs/shared/` at a pinned commit. Here they are included by the short
pages `docs/technical-decisions.md` and `docs/high-level-access.md`. Edit
the files in `docs/shared/`, and keep them self-contained: no mkdocstrings
cross-references, and absolute `https://grid-doctor.org/...` links for
pages that only exist in this site.

### Publishing

The documentation at <https://grid-doctor.org> is versioned with
[Zensical's fork of mike](https://github.com/squidfunk/mike) and served by GitHub Pages from
the `gh-pages` branch (`.github/workflows/docs.yaml`):

* every push to `main` updates the version `dev`,
* a release tag `vX.Y.Z` deploys the version `X.Y` with the alias
  `latest`; the site root redirects there.  Until the first release,
  `dev` carries the alias `latest`.

To preview the versioned site locally, deploy into your local `gh-pages`
branch without pushing and serve it:

```console
tox -e docs-deploy -- deploy dev
tox -e docs-deploy -- serve
```


## Quick Start
Running the example requires ESMF to be installed, see above.
```python
import xarray as xr

import grid_doctor as gd

ds = xr.open_mfdataset("path/to/*.nc")
max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
weights_file = gd.cached_weights(ds, level=max_level, prefer_offline=True)
pyramid = gd.create_healpix_pyramid(
    ds,
    weights_path=weights_file,
    max_level=max_level
)
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/dataset.zarr",
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

## Swath and Point Data

Satellite Level-2 swaths, station records, and trajectories don't live on a
fixed grid, so the ESMF weight path doesn't apply (no weight reuse, and
nearest-neighbour would smear a swath over the whole globe).  Use point
binning instead — the output carries the standard metadata, so coarsening
and uploading work exactly as above:

```python
finest = gd.bin_to_healpix(
    ds,                                   # per-sample latitude/longitude
    level=11,                             # coarser than the sample spacing!
    agg={"radiance": "mean", "cloud_type": "mode"},
    fill_values={"cloud_type": 255},
    with_counts=True,                     # auditable coverage
)
pyramid = {11: finest}
for level in range(10, -1, -1):
    pyramid[level] = gd.coarsen_healpix(pyramid[level + 1], level)
gd.save_pyramid(pyramid, "s3://my-bucket/my-l2-product.zarr", ...)
```

`mean` is the binning analogue of conservative remapping (valid when the
samples oversample the target cells), `mode` the analogue of
nearest-neighbour for categorical fields.  Cells are defined on the
sphere by default, like remapping, or on the WGS84 ellipsoid
(`ellipsoid="WGS84"`).  Both are equally accurate but they are different
grids, displaced by up to ~14 km, so only compare data that uses the
same definition.  See the point-data
recipe in the documentation for details.

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
| CMIP6 | 🩹 | 😎 |
| NextGEMS | 😎 | 😎 |
| ICDC     | 😎 | 🩹 |
| ORCHESTRA | 😎 | 😎 |
| PalMod | 😢 | 😢 |
| Dyamond| 😎 | 😎 |
| EarthCARE | 😢 | 🩹 |
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
import xarray as xr

import grid_doctor as gd
import grid_doctor.cli as gd_cli

parser = gd_cli.get_parser("my-dataset", "Convert my-dataset to HEALPix.")
parser.add_argument("--variables", nargs="*", default=["t_2m"])
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = xr.open_mfdataset("path/to/*.nc")
pyramid = gd.create_healpix_pyramid(ds)
gd.save_pyramid(
    pyramid,
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



## Development checks

All checks run through the tox envs that CI uses. After cloning, install the
git hooks once so they run automatically:

```console
pip install -e .[dev]   # provides pre-commit and tox
pre-commit install
```

This installs two hooks (see `.pre-commit-config.yaml`):

* **pre-commit** – quick checks on the staged files only: lint on staged
  `src/` files, mypy when `src/` changed, and spelling on staged `docs/` files.
* **pre-push** – the full CI suite on the whole code: `lint`, `types`, `docs`
  and `test`.

To run the envs by hand:

```console
tox -e lint      # ruff and codespell
tox -e types     # mypy (strict)
tox -e docs      # spelling + mkdocs build --strict
tox -e test      # pytest with coverage
```


## Issues

As this is still very much work in progress it is very likely that you will
run into problems. Please note any problems in the `README.md` file
for your dataset folder. Pull requests are welcome; if you would rather
not submit one, file an issue report
[here](https://github.com/freva-org/grid-doctor/issues).
