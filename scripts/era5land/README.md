# ERA5-Land

## Current approach

TBD

## Installation

ERA5-Land uses the normal `grid_doctor` environment plus a local dependency
overlay and a local copy of the ERA5 CMOR tables.

Create a mamba environment in your local home conda installation:

```console
mamba create -n grid_doctor -c conda-forge python=3.11 pip
```

Activate it:

```console
mamba activate grid_doctor
```

From the repository root, install the main package and dependencies from
`pyproject.toml` into that active environment:

```console
python3 -m pip install -e .
```

Then layer the ERA5-Land-specific packages into the same active conda/mamba
environment:

```console
mamba env update -p "$CONDA_PREFIX" -f scripts/era5land/requirements.yml
```

Finally, install or refresh the local ERA5 CMOR tables:

```console
make -C scripts/era5land download
```

At this point, keep the environment active while working with ERA5-Land:

```console
mamba activate grid_doctor
python3 scripts/era5land/convert.py --help
```

The `requirements.yml` file is intentionally local to this script folder and
only includes the extra reader/workflow packages:

- `cfgrib`
- `eccodes`
- `kerchunk`
- `pyarrow`
- `python-dotenv`
- `reflow-hpc`

The Makefile is intentionally limited to local table assets. It does not manage
the Python environment.

## Running

TBD

## Development checks

To confine the dev checks to ERA5-Land only:

Run tests with:

```console
python3 -m pytest tests/test_era5land_convert.py
```

Lint the code with:

```console
python3 -m ruff check scripts/era5land/convert.py tests/test_era5land_convert.py
```

Apply auto-fixable Ruff checks with:

```console
python3 -m ruff check --fix scripts/era5land/convert.py tests/test_era5land_convert.py
```

Format the code with black _or_ ruff as you might end up in a format ping-pong:

```console
python3 -m black scripts/era5land/convert.py tests/test_era5land_convert.py
# or
python3 -m ruff format scripts/era5land/convert.py tests/test_era5land_convert.py
```


-------

### Issues with GRIB Forecast files,

variables such `2tm, tp` (`tas, pr`) are forecast products of `EL` (`EL/sf/fc/{1H,1D,1M}/{167,228}`)

there are 2 major problems here:

1. time dimension
2. spatial dimension: gaussian reduced

that is:

```python
ds_el = xr.open_dataset("/work/bk1099/data/EL/sf/fc/1H/167/ELsf12_1H_2026-04-16_167.grb", engine="cfgrib", backend_kwargs={"indexpath":""})
In [97]: ds_el
Out[97]:
<xarray.Dataset> Size: 1GB
Dimensions:     (time: 2, step: 24, values: 6599680)
Coordinates:
  * time        (time) datetime64[ns] 16B 2026-04-15 2026-04-16
  * step        (step) timedelta64[ns] 192B 01:00:00 ... 1 days 00:00:00
    valid_time  (time, step) datetime64[ns] 384B 2026-04-15T01:00:00 ... 2026...
    latitude    (values) float64 53MB ...
    longitude   (values) float64 53MB 0.0 18.0 36.0 54.0 ... 306.0 324.0 342.0
    number      int64 8B ...
    surface     float64 8B ...
Dimensions without coordinates: values
Data variables:
    t2m         (time, step, values) float32 1GB ...
Attributes:
    GRIB_edition:            1
    GRIB_centre:             ecmf
    GRIB_centreDescription:  European Centre for Medium-Range Weather Forecasts
    GRIB_subCentre:          0
    Conventions:             CF-1.7
    institution:             European Centre for Medium-Range Weather Forecasts
    history:                 2026-04-23T16:29 GRIB to CDM+CF via cfgrib-0.9.1...
```

an attempt with cdo to transform it to a netcdf file with minimal processing (only varname was chnaged):
```
╰─$ cdo -f nc4 -z zip_9 chname,var167,tas /work/bk1099/data/EL/sf/fc/1H/167/ELsf12_1H_2026-04-16_167.grb tas_orig.nc
```
when opened via xarray:
```python
In [94]: ds
Out[94]:
<xarray.Dataset> Size: 634MB
Dimensions:         (time: 24, rgrid: 6599680, reduced_points: 2560, lat: 2560)
Coordinates:
  * time            (time) datetime64[ns] 192B 2026-04-16 ... 2026-04-16T23:0...
  * reduced_points  (reduced_points) int32 10kB 20 24 28 32 36 ... 32 28 24 20
  * lat             (lat) float64 20kB 3.081e+218 -1.176e+210 ... 1.171e-240
Dimensions without coordinates: rgrid
Data variables:
    tas             (time, rgrid) float32 634MB ...
Attributes:
    CDI:          Climate Data Interface version 2.2.4 (https://mpimet.mpg.de...
    Conventions:  CF-1.6
    institution:  European Centre for Medium-Range Weather Forecasts
    history:      Thu Apr 23 17:10:26 2026: cdo -f nc4 -z zip_9 chname,var167...
    CDO:          Climate Data Operators version 2.2.2 (https://mpimet.mpg.de...

```
