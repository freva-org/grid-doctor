# ERA5-Land

## Current approach

`convert.py` is now config-driven and expects an adjacent JSON config file plus a
variable-conversion table:

- `config.json`
- `variable_map.json`

The default setup reads GRIB files through `cfgrib`, renames source variables to
target names, and applies unit conversions defined outside the code.

If an adjacent `.env` file exists, `convert.py` loads it before resolving S3
credentials.

## Inputs

The source config supports both GRIB and kerchunk in the same config file. The
active source can be selected from `source.default_kind` or overridden on the
command line with `--use-grib` or `--use-kerchunk`.

The sample `config.json` uses:

- `source.kind.grib.path` for direct GRIB access through `cfgrib`
- `source.kind.kerchunk.path` for parquet-backed kerchunk references

The path templates are metadata-driven and can expand fields such as:

- `{dataset}`
- `{stream}`
- `{type}`
- `{time_freq}`
- `{parameter}`

## Conversion table

`variable_map.json` stores dataset-specific logic outside the script. Each entry is keyed
by the source variable name and can define:

- `target`
- `source_units`
- `target_units`
- `factor`
- `offset`
- metadata such as `standard_name`, `long_name`, `cell_methods`, and extra `attrs`

## Dependencies

First install the repository itself from the project root in the normal grid-doctor
environment:

```console
python3 -m pip install -e .
```

Then layer the ERA5-Land-specific dependencies into that same active conda
environment:

```console
conda env update -p "$CONDA_PREFIX" -f scripts/era5land/requirements.yml
```

or, from `scripts/era5land`:

```console
make env
```

The ERA5-Land environment file is intentionally local to this script folder and
only includes the extra reader/workflow dependencies:

- `cfgrib`
- `eccodes`
- `kerchunk`
- `pyarrow`
- `python-dotenv`
- `reflow-hpc`

## Running

```console
python3 scripts/era5land/convert.py --config scripts/era5land/config.json --init
python3 scripts/era5land/convert.py --config scripts/era5land/config.json --slice-size 1440
```

With the sample config, each run writes a local preview first under
`{scratch}/era5land-preview`, where `{scratch}` resolves via
`grid_doctor.utils.cache_dir()`, and then writes the same layout to the configured S3
destination.

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

## Notes

- The sample `config.json` includes both GRIB and kerchunk source definitions.
- In kerchunk mode, the code first looks for a matching parquet reference and can fall back to GRIB when no matching reference exists.
- S3 secrets should live in `.env`, for example:

```dotenv
access_key=...
secret_key=...
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
