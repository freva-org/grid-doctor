# Configuration

`grid_doctor.config` holds library-wide settings. A setting is resolved in
this order, later ones winning:

1. the built-in default,
2. the environment variable `GRID_DOCTOR_<NAMESPACE>__<KEY>` (read at import),
3. assignment at runtime.

| Setting | Default | Environment variable |
|---|---|---|
| `healpix.ellipsoid` | `"sphere"` | `GRID_DOCTOR_HEALPIX__ELLIPSOID` |

```python
import grid_doctor as gd

gd.config.healpix.ellipsoid = "WGS84"  # HEALPix geometry on WGS84

with gd.config.healpix.set(ellipsoid="sphere"):  # restored after the block
    ...

with gd.config.set(healpix={"ellipsoid": "sphere"}):  # several namespaces at once
    ...
```

Passing a whole namespace object, e.g. `gd.config.set(healpix=HealpixConfig())`,
applies all of its settings, defaults included.

::: grid_doctor.config
    options:
      show_root_heading: true
      members_order: source
      members: [set, HealpixConfig, healpix]
      show_bases: false
      inherited_members: true
      filters: ["!^_"]
