# Types

Shared type aliases used in grid-doctor's signatures.  Use them to
annotate your own code, e.g. an encoding you build up before passing it
to [`save_pyramid`][grid_doctor.save_pyramid]:

```python
from grid_doctor.types import PyramidEncoding

encoding: PyramidEncoding = {
    level: {"ta": {"dtype": "int16", "scale_factor": 0.01}}
    for level in pyramid
}
```

::: grid_doctor.types
    options:
      show_root_heading: false
      show_source: false
      members_order: source
