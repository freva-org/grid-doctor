# Configuration

`grid_doctor.config` holds library-wide settings. A setting is resolved in
this order, later ones winning:

1. the built-in default,
2. the environment variable `GRID_DOCTOR_<SETTING>` (read at import),
3. assignment at runtime.

| Setting | Default | Values | Environment variable |
|---|---|---|---|
| `ellipsoid` | `"sphere"` | `"sphere"`, `"WGS84"` | `GRID_DOCTOR_ELLIPSOID` |
| `weights_store` | `None` (`$XDG_CACHE_HOME/grid-doctor/weights`) | local path, `file://` or `memory://` URL | `GRID_DOCTOR_WEIGHTS_STORE` |

```python
import grid_doctor as gd

gd.config.ellipsoid = "WGS84"  # HEALPix geometry on WGS84

with gd.config.set(ellipsoid="sphere", weights_store="/work/ab1234/weights"):
    ...  # restored after the block
```

Settings are process-wide: a `with gd.config.set(...)` block also affects
other threads while it is open, and dask workers in other processes only see
the environment variable.

`ellipsoid` applies to grids being created (regridding, binning, weight
generation). The ellipsoid is recorded in the output (CF
`reference_ellipsoid_name`, `semi_major_axis` and `inverse_flattening` for
WGS84) and in weight files; reading, selecting from or coarsening existing
data uses the recorded ellipsoid. HEALPix metadata or a weight file without
these attributes is spherical. A dataset with no HEALPix metadata at all gets
this setting when selecting from it with an explicit `level=`; coarsening
requires the HEALPix metadata.

`weights_store` is where [`cached_weights`][grid_doctor.cached_weights] and
[`regrid_to_healpix`][grid_doctor.regrid_to_healpix] look up and store weight
files; see [Weights Store](weights.md). grid-doctor never guesses a scratch
or project directory: point it at one yourself, for example on an HPC system
where the home directory has a small quota.

::: grid_doctor.config
    options:
      show_root_heading: true
      members_order: source
      members: [ellipsoid, weights_store, set, env_var, SETTINGS]
      filters: ["!^_"]
