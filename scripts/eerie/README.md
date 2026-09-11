# EERIE

## Goal

This folder contains a first draft EERIE workflow in
[convert.py](/home/k/k202181/Git/grid-doctor/scripts/eerie/convert.py).
It is a baseline script that uses the current `grid_doctor` functionality
without the older custom EERIE remapping helpers.

The goal of this draft is to turn the earlier exploratory EERIE work into a
reviewable and reproducible conversion path that follows the repository style:
a script, a documented workflow, and explicit notes about what is still missing
in `grid_doctor`.

## Approach

The current script does the following:

1. Open the EERIE STAC / kerchunk dataset with `xarray`.
2. Open the ICON grid file and use its cell centers for the current
   `grid_doctor` Delaunay workflow.
3. Remap once to the finest HEALPix level with `grid_doctor`.
4. Build lower zoom levels from that finest output.
5. Write a local multiscales Zarr tree at
   `<out-root>.zarr/multiscales/zoom_<N>`.
6. Keep rechunking and S3 publication as a separate rechunker step.

## How to run

1. Install the project into your remapping venv:

```console
/work/k20200/k202181/venv-regrid/bin/python -m pip install -r scripts/eerie/requirements.txt
```

2. Run the draft script to create a local multiscales store, for example:

```console
/work/k20200/k202181/venv-regrid/bin/python scripts/eerie/convert.py \
  --dataset hist-1950-v20240618 \
  --gridfile /pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc \
  --out-root /scratch/$USER/remap_eerie/eerie-hist-1950-v20240618 \
  --variables pr \
  -vv
```

Another preset:

```console
/work/k20200/k202181/venv-regrid/bin/python scripts/eerie/convert.py \
  --dataset future-ssp245-v20240618 \
  --out-root /scratch/$USER/remap_eerie/eerie-future-ssp245-v20240618 \
  --variables ts pr \
  -vv
```

Useful options:

- `--item-url` to override the preset STAC item
- `--max-level` to force the HEALPix target level
- `--out-root` to choose the local output root
- `--overwrite-output` to replace an existing local store
- `--apply-chunk-optimizer` if you want the in-process `ChunkOptimizer` step

## Issues Encountered

The main limitations found while doing this are:

- The standard `grid_doctor` write path does not yet produce the EERIE-style
  multiscales layout directly, so this draft adds that layout in the dataset
  script.
- Rechunking and publishing still need to stay in the separate rechunker flow.

## Conclusions

This draft is good enough to serve as a reproducible `grid_doctor`-based
baseline for EERIE.

It now covers the parts that can already be done with the shared library:
opening the source data, computing weights, remapping, creating the pyramid,
and writing a local multiscales store.

The remaining work is mainly library work rather than dataset-script work:

- use ICON grid connectivity directly
- use Delaunay only as a fallback
- make the multiscales output path part of the standard library workflow
