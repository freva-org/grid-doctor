# Configuration

`grid_doctor.config` holds library-wide settings. A setting is resolved in
this order, later ones winning:

1. the built-in default,
2. the environment variable `GRID_DOCTOR_<NAMESPACE>__<KEY>` (read at import),
3. assignment at runtime,
4. an explicit function argument.

| Setting | Default | Environment variable |
|---|---|---|
| `remap.method` | `"conservative"` | `GRID_DOCTOR_REMAP__METHOD` |

```python
import grid_doctor as gd

gd.config.remap.method = "nearest"
gd.regrid_to_healpix(ds, 8)                         # nearest
gd.regrid_to_healpix(ds, 8, method="conservative")  # explicit argument wins

with gd.config.remap.set(method="conservative"):  # restored after the block
    gd.regrid_to_healpix(ds, 8)                    # conservative

with gd.config.set(remap={"method": "nearest"}):  # several namespaces at once
    ...
```

Passing a whole namespace object, e.g. `gd.config.set(remap=RemapConfig())`,
applies all of its settings, defaults included.

::: grid_doctor.config
    options:
      show_root_heading: true
      members_order: source
      members: [set, RemapConfig, remap]
      show_bases: false
      inherited_members: true
      filters: ["!^_"]
