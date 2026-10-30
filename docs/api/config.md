# Configuration

`grid_doctor.config` holds library-wide settings. A setting is resolved in
this order, later ones winning:

1. the built-in default,
2. the environment variable `GRID_DOCTOR_<NAMESPACE>__<KEY>` (read at import),
3. assignment at runtime,
4. an explicit function argument.

| Setting | Default | Environment variable |
|---|---|---|
| `remap.method` | `"nearest"` | `GRID_DOCTOR_REMAP__METHOD` |

```python
import grid_doctor as gd

gd.config.remap.method = "conservative"
gd.regrid_to_healpix(ds, 8)                    # conservative
gd.regrid_to_healpix(ds, 8, method="nearest")  # explicit argument wins
```

::: grid_doctor.config
    options:
      show_root_heading: true
      members_order: source
