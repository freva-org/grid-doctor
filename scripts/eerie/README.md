# EERIE

## What is EERIE?

This folder contains a first draft EERIE workflow in
[convert.py](/home/k/k202181/Git/grid-doctor/scripts/eerie/convert.py).
It is a baseline script that uses the current `grid_doctor` functionality
without the older custom EERIE remapping helpers.

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

## Current status

This draft now writes a local multiscales Zarr tree at
`<out-root>.zarr/multiscales/zoom_<N>`.

Current workflow:

1. use `grid_doctor` for opening the data, computing the weights, remapping,
   and creating the pyramid
2. Rechunker-based flow for final chunking and publishing to S3

So the script does **not** upload to S3 directly. 
The rechunker calls and configs are stored in this project too.

