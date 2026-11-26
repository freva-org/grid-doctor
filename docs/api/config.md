# Configuration

`grid_doctor.config` holds library-wide settings. A setting is resolved in
this order, later ones winning:

1. the built-in default,
2. the environment's config file, `<sys.prefix>/etc/grid-doctor/config.toml`
   (in a conda environment `$CONDA_PREFIX/etc/grid-doctor/config.toml`),
   for everyone using that environment,
3. your config file, `config.toml` in your user config directory as given
   by [platformdirs](https://platformdirs.readthedocs.io):
   `~/.config/grid-doctor/config.toml` on Linux,
   `~/Library/Application Support/grid-doctor/config.toml` on macOS. The file
   named by `GRID_DOCTOR_CONFIG` replaces it,
4. the environment variable `GRID_DOCTOR_<SETTING>`,
5. changes at runtime, including config files loaded with `set`.

Files and environment variables are read at import. Every setting is
optional.

| Setting | Default | Values | Environment variable |
|---|---|---|---|
| `ellipsoid` | `"sphere"` | `"sphere"`, `"WGS84"` | `GRID_DOCTOR_ELLIPSOID` |
| `weights_store` | `None` (`~/.cache/grid-doctor/weights` on Linux) | local path, `file://` or `memory://` URL | `GRID_DOCTOR_WEIGHTS_STORE` |
| `esmf_regrid_weightgen` | `None` (`ESMF_RegridWeightGen` from `PATH`) | name or path of the executable | `GRID_DOCTOR_ESMF_REGRID_WEIGHTGEN` |
| `scratch_dir` | `None` (`$TMPDIR`, usually `/tmp`) | directory | `GRID_DOCTOR_SCRATCH_DIR` |

## Changing settings at runtime

[`set`][grid_doctor.config.set] changes settings from a config file, one by
one, or both (the file first). Called on its own, the change stays until
changed again; used with `with`, it is restored after the block:

```python
import grid_doctor as gd

gd.config.set("project.toml")                         # a file's settings
gd.config.set(scratch_dir="/scratch/ab1234")          # single settings
gd.config.set("project.toml", ellipsoid="WGS84")      # the file, then these

with gd.config.set(weights_store="/work/ab1234/weights"):
    ...  # restored after the block

gd.config.get("weights_store")   # one setting
gd.config.get()                  # all settings, as a dict
gd.config.loaded_files           # the config files in effect
gd.config.refresh()              # read the default files and variables again
```

A file loaded with `set` is a change at runtime, so it overrides
environment variables. Only the settings it names change. The file and the
values are checked first: if anything is invalid, nothing changes.
Assigning a setting (`gd.config.ellipsoid = "WGS84"`) works too and is
checked the same way. [`refresh`][grid_doctor.config.refresh] resets every
setting and reads the default files and environment variables again, as at
import; use it after editing a config file.

Settings are process-wide: a `with gd.config.set(...)` block also affects
other threads while it is open.

!!! note "Multi-process dask"
    Dask workers in other processes read the default config files and
    environment variables themselves, but do not see changes made at runtime;
    put settings they need in a default config file, the file named by
    `GRID_DOCTOR_CONFIG`, or an environment variable.

## Config files

Config files are [TOML](https://toml.io) with one key per setting. Write a
commented one to start from, with every setting at its default, or set
single entries:

```console
python -m grid_doctor.config.template                    # your config file
python -m grid_doctor.config.template -o "$CONDA_PREFIX/etc/grid-doctor"
python -m grid_doctor.config.template --stdout
python -m grid_doctor.config.template --set weights_store=/work/ab1234/weights
```

Without `-o`, the command writes your config file
([`user_config_file()`][grid_doctor.config.user_config_file]). `-o` takes a
directory (created if needed, the file is `config.toml`) or a file ending in
`.toml`. `--set KEY=VALUE` (repeatable) sets a setting: in a new file it is
written as an active line, and an existing file is updated in place, keeping
everything else. Without `--set`, an existing file is only replaced with
`--force`. A file could look like this:

```toml
# $CONDA_PREFIX/etc/grid-doctor/config.toml, shared by a project
weights_store = "/work/ab1234/grid-doctor/weights"
esmf_regrid_weightgen = "/work/ab1234/esmf/bin/ESMF_RegridWeightGen"
scratch_dir = "/scratch/a/ab1234/grid-doctor"
```

A key in a later file wins; an empty string (`scratch_dir = ""`) unsets a
setting given by a file read before. `~` is expanded in paths.
`GRID_DOCTOR_CONFIG=""` skips your own file, e.g. in tests, and
[`loaded_files`][grid_doctor.config.loaded_files] lists the files that were
read.

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

`esmf_regrid_weightgen` and `scratch_dir` apply to offline weight
generation (`prefer_offline=True`). `scratch_dir` holds a temporary working
directory per run for the intermediate mesh files, which get large for fine
grids; it is removed after the run. The `workdir=` argument instead names a
directory that is kept.

::: grid_doctor.config
    options:
      show_root_full_path: true
      members_order: source
      members: [ellipsoid, weights_store, esmf_regrid_weightgen, scratch_dir, set, get, refresh, loaded_files, user_config_file, environment_config_file, config_files, env_var, SETTINGS, CONFIG_ENV_VAR, ConfigWarning]
      filters: ["!^_"]
