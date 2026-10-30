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

gd.config.healpix.ellipsoid = "WGS84"
```

::: grid_doctor.config
    options:
      show_root_heading: true
      members_order: source
      members: [HealpixConfig, healpix]
      show_bases: false
