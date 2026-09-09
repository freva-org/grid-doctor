# Grid Doctor HEALs your Grids
<p align="center">
  <img src="docs/assets/logo-512.png" alt="Logo" width="200"><br>
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
[ESMF](https://earthsystemmodeling.org/regrid/) through conda-forge.

```console
mamba install -c conda-forge -y "esmf=*=mpi_openmpi_*" esmpy
```

## Developing the documentation

This repository builds *two* documentation sites from the same `docs/` source tree:

* **Technical documentation**: built with `mkdocs.tech.yml`
* **Waterpark/data documentation**: built with `mkdocs.data.yml`

Both sites share common assets and selected shared Markdown files, but are published independently.

### Documentation layout

The documentation sources are organised as follows:

```text
docs/
├── technical/   # technical Grid Doctor documentation
├── data/        # Waterpark/data-oriented documentation
├── shared/      # pages shared by both documentation sites
└── assets/      # images, CSS, JavaScript, logos, etc.
```

During the build, the relevant documentation tree is staged into `.build/`:

```text
.build/
├── tech/
└── data/
```

This allows both documentation sites to behave as if their own content starts at the web root `/`.

### Building the documentation

The default documentation target builds the technical documentation:

```console
tox -e docs
```

This is equivalent to:

```console
tox -e docs -- tech
```

To build the Waterpark/data documentation:

```console
tox -e docs -- data
```

The generated output directories are:

```text
site-tech/   # technical documentation
site-data/   # Waterpark/data documentation
```

### Serving the documentation locally for easy and quick development

To serve the technical documentation locally:

```console
tox -e docs-serve
```

or explicitly:

```console
tox -e docs-serve -- tech
```

To serve the Waterpark/data documentation locally:

```console
tox -e docs-serve -- data
```

The documentation is then served at:

```text
http://localhost:8000
```

The local serve target uses symlinks in `.build/`, so changes made under `docs/technical/`, `docs/data/`, `docs/shared/`, or `docs/assets/` can be picked up while developing.

### Adding new pages

Add technical documentation pages under:

```text
docs/technical/
```

Add Waterpark/data documentation pages under:

```text
docs/data/
```

Add pages that should be available to both sites under:

```text
docs/shared/
```

After adding a page, include it in the appropriate MkDocs navigation file:

```text
mkdocs.tech.yml
mkdocs.data.yml
```

For example:

```yaml
nav:
  - Overview: index.md
  - Getting started: getting-started.md
```

### Adding images and other assets

Shared images, CSS, JavaScript, logos, and other static assets should go under:

```text
docs/assets/
```

You can reference them from Markdown like this:

```markdown
![Example image](assets/example.png)
```

or, when a root-relative link is more appropriate:

```markdown
![Example image](/assets/example.png)
```

### Shared pages

Files in `docs/shared/` are staged into both documentation builds. This is useful for pages such as common concepts, terminology, license notes, or explanations that apply to both the technical and data-facing documentation.

For example, a shared file:

```text
docs/shared/technical-decisions.md
```

can be referenced in either MkDocs config as:

```yaml
nav:
  - Technical decisions: technical-decisions.md
```

### Deployment overview

The two documentation sites are deployed differently:

* `site-tech/` is deployed via the GitHub Pages action.
* `site-data/` is pushed to the `gh-pages` branch and served independently from the custom Waterpark web server.

This allows both sites to use `/` as their root URL while still keeping all documentation sources in a single repository.


```console
pip install tox
tox -e docs          # build to site/
tox -e docs-serve    # live preview at http://127.0.0.1:8000
```


## Quick Start
Running the example will likely require ESMF to be installed, see above.
```python
import grid_doctor as gd

ds = gd.cached_open_dataset(["path/to/*.nc"])
max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
weights_dir="/scratch/{user[0]}/{user}/grid-doctor/weights"\
    .format(user=getuser(), level=level)
weights_file = gd.cached_weights(
    ds,
    level=max_level,
    prefer_offline=True,
    cache_path=weights_dir
)
pyramid = gd.create_healpix_pyramid(
    ds,
    weights_path=weights_file,
    max_level=max_level
)
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/dataset.zarr",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
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
nearest-neighbour for categorical fields.  All cell geometry stays on the
perfect sphere — do not index satellite data on the WGS84 ellipsoid, or it
will be misregistered against every other dataset in the hub.  See the
point-data recipe in the documentation for details.

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
import grid_doctor as gd
import grid_doctor.cli as gd_cli
from data_portal_worker.rechunker import ChunkOptimizer

parser = gd_cli.get_parser("my-dataset", "Convert my-dataset to HEALPix.")
parser.add_argument("--variables", nargs="*", default=["t_2m"])
args = parser.parse_args()
gd_cli.setup_logging_from_args(args)

ds = gd.cached_open_dataset(["path/to/*.nc"])
pyramid = gd.create_healpix_pyramid(ds)
gd.save_pyramid(
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
