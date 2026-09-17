"""First-draft EERIE conversion workflow using the current grid-doctor API.

This script intentionally stays close to the existing ICON-DREAM workflow:

1. Open an EERIE STAC item / kerchunk reference with xarray.
2. Open the ICON grid file as an xarray dataset.
3. Compute or load the current grid-doctor Delaunay weights.
4. Regrid to a HEALPix pyramid with grid-doctor.
5. Write a local multiscales Zarr tree for the rechunker-based publish step.

Current limitation: the grid file is only used to supply lat/lon centers for
the Delaunay weight computation. Native ICON connectivity is not used here.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr
import zarr

import grid_doctor as gd
import grid_doctor.cli as gd_cli

logger = logging.getLogger(__name__)

_UNSTRUCTURED_DIMS: tuple[str, ...] = ("cell", "ncells", "ncell", "nCells", "values")
_LAT_NAMES: tuple[str, ...] = ("clat", "lat", "latitude")
_LON_NAMES: tuple[str, ...] = ("clon", "lon", "longitude")


@dataclass(frozen=True)
class EerieDatasetConfig:
    """Metadata for a supported EERIE source dataset."""

    dataset_id: str
    item_url: str
    frequency_tag: str = "P1M"
    stat_tag: str = "mean"
    default_variables: tuple[str, ...] = ()


EERIE_PRESETS: dict[str, EerieDatasetConfig] = {
    "hist-1950-v20240618": EerieDatasetConfig(
        dataset_id="eerie-hist-1950-v20240618",
        item_url=(
            "https://stac2.cloud.dkrz.de/fastapi/collections/"
            "eerie-eerie-mpi-m-icon-esm-er-hist-1950-v20240618/items/"
            "eerie-eerie-mpi-m-icon-esm-er-hist-1950-v20240618-"
            "disk.model-output.icon-esm-er.hist-1950.v20240618."
            "atmos.native.2d_monthly_mean-zarr-kerchunk"
        ),
        default_variables=("pr",),
    ),
    "future-ssp245-v20240618": EerieDatasetConfig(
        dataset_id="eerie-future-ssp245-v20240618",
        item_url=(
            "https://stac2.cloud.dkrz.de/fastapi/collections/"
            "eerie-eerie-mpi-m-icon-esm-er-highres-future-ssp245-v20240618/items/"
            "eerie-eerie-mpi-m-icon-esm-er-highres-future-ssp245-v20240618-"
            "disk.model-output.icon-esm-er.highres-future-ssp245.v20240618."
            "atmos.native.2d_monthly_mean-zarr-kerchunk"
        ),
        default_variables=("ts", "pr"),
    ),
}


def _find_coord_name(ds: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in ds.coords or name in ds.data_vars:
            return name
    return None


def _looks_like_radians(values: np.ndarray) -> bool:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.nanmax(np.abs(finite)) <= np.pi + 0.01)


def _get_spatial_dim(ds: xr.Dataset) -> str:
    for dim in _UNSTRUCTURED_DIMS:
        if dim in ds.dims:
            return dim
    raise ValueError(
        "Could not determine unstructured spatial dimension. "
        f"Found dims: {dict(ds.sizes)}"
    )


def normalize_unstructured_dims(ds: xr.Dataset) -> xr.Dataset:
    """Normalize common ICON-like dimension names for grid-doctor."""
    if "values" in ds.dims and "cell" not in ds.dims:
        ds = ds.rename_dims({"values": "cell"})
    return ds


def _assign_array(
    ds: xr.Dataset,
    name: str,
    dim: str,
    values: np.ndarray,
    attrs: dict[str, object],
) -> xr.Dataset:
    array = xr.DataArray(values, dims=(dim,), attrs=attrs)
    if name in ds.coords:
        return ds.assign_coords({name: array})
    return ds.assign({name: array})


def ensure_dataset_coords_in_degrees(ds: xr.Dataset) -> xr.Dataset:
    """Convert a recognised 1-D lat/lon pair from radians to degrees if needed."""
    lat_name = _find_coord_name(ds, _LAT_NAMES)
    lon_name = _find_coord_name(ds, _LON_NAMES)
    if lat_name is None or lon_name is None:
        return ds

    lat = np.asarray(ds[lat_name].values)
    lon = np.asarray(ds[lon_name].values)
    if not (_looks_like_radians(lat) and _looks_like_radians(lon)):
        return ds

    dim = str(ds[lat_name].dims[0])
    ds = _assign_array(
        ds,
        lat_name,
        dim,
        np.rad2deg(lat),
        dict(ds[lat_name].attrs),
    )
    ds = _assign_array(
        ds,
        lon_name,
        dim,
        np.rad2deg(lon),
        dict(ds[lon_name].attrs),
    )
    logger.info("Converted %s/%s from radians to degrees", lat_name, lon_name)
    return ds


def ensure_source_coords_from_grid(
    source_ds: xr.Dataset,
    grid_ds: xr.Dataset,
) -> xr.Dataset:
    """Attach 1-D grid-file lat/lon coordinates when the source lacks them."""
    if _find_coord_name(source_ds, _LAT_NAMES) and _find_coord_name(
        source_ds, _LON_NAMES
    ):
        return ensure_dataset_coords_in_degrees(source_ds)

    grid_lat_name = _find_coord_name(grid_ds, _LAT_NAMES)
    grid_lon_name = _find_coord_name(grid_ds, _LON_NAMES)
    if grid_lat_name is None or grid_lon_name is None:
        raise ValueError("Grid dataset does not expose a recognised lat/lon pair")

    source_dim = _get_spatial_dim(source_ds)
    grid_lat = np.asarray(grid_ds[grid_lat_name].values)
    grid_lon = np.asarray(grid_ds[grid_lon_name].values)
    if _looks_like_radians(grid_lat) and _looks_like_radians(grid_lon):
        grid_lat = np.rad2deg(grid_lat)
        grid_lon = np.rad2deg(grid_lon)

    if grid_lat.size != source_ds.sizes[source_dim]:
        raise ValueError(
            "Grid-file coordinates do not match source spatial dimension size: "
            f"{grid_lat.size} != {source_ds.sizes[source_dim]}"
        )

    source_ds = source_ds.assign_coords(
        {
            "clat": (source_dim, grid_lat),
            "clon": (source_dim, grid_lon),
        }
    )
    logger.info(
        "Attached clat/clon from the grid file to source dimension %s", source_dim
    )
    return source_ds


def select_variables(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    """Keep a specific set of variables and fail loudly on typos."""
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Variables not found in source dataset: {missing}. "
            f"Available variables: {sorted(map(str, ds.data_vars))}"
        )
    return ds[list(variables)]


def build_output_root(
    *,
    config: EerieDatasetConfig,
    out_root: Optional[str] = None,
) -> Path:
    """Build the local output root used for the multiscales Zarr store."""
    if out_root is not None:
        return Path(out_root)
    return gd_cli.get_scratch("remap_eerie", config.dataset_id)


def make_multiscales_root_path(out_root: str | Path) -> Path:
    out_root = Path(out_root)
    if str(out_root).endswith(".zarr"):
        return out_root
    return Path(f"{out_root}.zarr")


def _multiscales_layout(max_level: int, min_level: int) -> list[dict[str, object]]:
    layout: list[dict[str, object]] = []
    previous_asset: Optional[str] = None

    for level in range(max_level, min_level - 1, -1):
        asset = f"zoom_{level}"
        entry: dict[str, object] = {"asset": asset}
        if previous_asset is not None:
            entry["derived_from"] = previous_asset
            entry["transform"] = {"scale": [2.0]}
        layout.append(entry)
        previous_asset = asset

    return layout


def _zarr_encoding_with_slash_separator(ds: xr.Dataset) -> dict[str, dict[str, object]]:
    """Force Zarr v2 chunk keys to use '/' separators like the older EERIE stores."""
    encoding: dict[str, dict[str, object]] = {}

    for name, variable in ds.variables.items():
        if variable.ndim == 0:
            continue
        encoding[name] = {"dimension_separator": "/"}

    return encoding


def write_pyramid_to_multiscales_zarr(
    *,
    pyramid: dict[int, xr.Dataset],
    source_ds: xr.Dataset,
    out_root: str | Path,
    config: EerieDatasetConfig,
    overwrite: bool = False,
    consolidated: bool = True,
) -> Path:
    """Write the pyramid to a local multiscales Zarr tree."""
    root_path = make_multiscales_root_path(out_root)
    if root_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {root_path}. "
                "Use --overwrite-output to replace it."
            )
        shutil.rmtree(root_path)

    max_level = max(map(int, pyramid))
    min_level = min(map(int, pyramid))

    root = zarr.open_group(root_path, mode="w")
    root.attrs.update(
        {
            "dataset_id": config.dataset_id,
            "frequency": config.frequency_tag,
            "statistic": config.stat_tag,
            "healpix_zoom_min": int(min_level),
            "healpix_zoom_max": int(max_level),
            "source_dataset_attrs": {
                str(key): str(value) for key, value in source_ds.attrs.items()
            },
        }
    )

    multiscales_group = zarr.open_group(root_path / "multiscales", mode="a")
    multiscales_group.attrs.update(
        {
            "dataset_id": config.dataset_id,
            "frequency": config.frequency_tag,
            "statistic": config.stat_tag,
            "zarr_conventions": [
                {
                    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
                    "name": "multiscales",
                    "schema_url": (
                        "https://raw.githubusercontent.com/zarr-conventions/"
                        "multiscales/refs/tags/v1/schema.json"
                    ),
                    "spec_url": (
                        "https://github.com/zarr-conventions/multiscales/blob/v1/README.md"
                    ),
                }
            ],
            "multiscales": {
                "resampling_method": "mean",
                "layout": _multiscales_layout(max_level=max_level, min_level=min_level),
            },
        }
    )

    for level in sorted(pyramid, reverse=True):
        ds = pyramid[level]
        ds = ds.assign_attrs(
            dataset_id=config.dataset_id,
            frequency=config.frequency_tag,
            statistic=config.stat_tag,
            healpix_zoom=int(level),
        )
        store_path = root_path / "multiscales" / f"zoom_{level}"
        logger.info("Writing zoom %d to %s", level, store_path)
        ds.to_zarr(
            store_path,
            mode="w",
            consolidated=consolidated,
            encoding=_zarr_encoding_with_slash_separator(ds),
        )

    if consolidated:
        zarr.consolidate_metadata(root_path)

    return root_path


def resolve_dataset_config(args_dataset: str, args: object) -> EerieDatasetConfig:
    config = EERIE_PRESETS[args_dataset]

    item_url = getattr(args, "item_url", None)
    if item_url:
        config = replace(config, item_url=item_url)

    frequency_tag = getattr(args, "frequency_tag", None)
    if frequency_tag:
        config = replace(config, frequency_tag=frequency_tag)

    stat_tag = getattr(args, "stat_tag", None)
    if stat_tag:
        config = replace(config, stat_tag=stat_tag)

    return config


def open_stac_dataset(item_url: str, asset: str) -> xr.Dataset:
    """Open a kerchunk-backed dataset described by a STAC item."""
    with gd_cli.AutoRaiseSession() as session:
        response = session.get(item_url, timeout=60)
        response.raise_for_status()
        item = response.json()

    try:
        asset_info = item["assets"][asset]
    except KeyError as error:
        raise KeyError(
            f"Asset {asset!r} not found in STAC item. "
            f"Available assets: {sorted(item.get('assets', {}))}"
        ) from error

    return xr.open_dataset(
        asset_info["href"],
        **asset_info["xarray:open_kwargs"],
        storage_options=asset_info.get("xarray:storage_options"),
    )


def maybe_rechunk_pyramid(
    pyramid: dict[int, xr.Dataset],
    apply_chunk_optimizer: bool,
) -> dict[int, xr.Dataset]:
    """Apply the project ChunkOptimizer unless explicitly disabled."""
    if not apply_chunk_optimizer:
        return pyramid

    from data_portal_worker.rechunker import ChunkOptimizer

    optimizer = ChunkOptimizer()
    return {level: optimizer.apply(ds) for level, ds in pyramid.items()}


def run_pipeline(args: object) -> None:
    config = resolve_dataset_config(getattr(args, "dataset"), args)

    logger.info("Opening grid file: %s", getattr(args, "gridfile"))
    grid_ds = gd.cached_open_dataset([str(getattr(args, "gridfile"))])
    grid_ds = normalize_unstructured_dims(ensure_dataset_coords_in_degrees(grid_ds))

    max_level = getattr(args, "max_level")
    if max_level is None:
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))

    weights_cache = getattr(args, "weights_cache")
    if weights_cache is None:
        weights_cache = gd_cli.get_scratch("remap_eerie", "weights")

    logger.info("Preparing weights for HEALPix level %s", max_level)
    weights = gd.cached_weights(
        grid_ds,
        level=max_level,
        cache_path=weights_cache,
    )

    logger.info("Opening EERIE source dataset: %s", config.item_url)
    source_ds = open_stac_dataset(config.item_url, getattr(args, "asset"))
    source_ds = normalize_unstructured_dims(source_ds)

    variables = tuple(getattr(args, "variables") or config.default_variables)
    if variables:
        source_ds = select_variables(source_ds, variables)
    source_ds = ensure_source_coords_from_grid(source_ds, grid_ds)

    time_chunk = getattr(args, "time_chunk")
    if time_chunk is not None and "time" in source_ds.dims:
        source_ds = source_ds.chunk({"time": time_chunk})

    source_dim = _get_spatial_dim(source_ds)
    source_ds = source_ds.chunk({source_dim: -1})

    logger.info("Building HEALPix pyramid for variables: %s", list(source_ds.data_vars))
    healpix_pyramid = gd.latlon_to_healpix_pyramid(
        source_ds,
        min_level=getattr(args, "min_level"),
        max_level=max_level,
        weights=weights,
    )

    healpix_pyramid = maybe_rechunk_pyramid(
        healpix_pyramid,
        apply_chunk_optimizer=getattr(args, "apply_chunk_optimizer"),
    )

    output_root = build_output_root(
        config=config,
        out_root=getattr(args, "out_root"),
    )
    output_path = write_pyramid_to_multiscales_zarr(
        pyramid=healpix_pyramid,
        source_ds=source_ds,
        out_root=output_root,
        config=config,
        overwrite=getattr(args, "overwrite_output"),
        consolidated=not getattr(args, "skip_consolidate"),
    )
    logger.info("Local multiscales output written to %s", output_path)


def parse_args(name: str, argv: Optional[Sequence[str]] = None) -> object:
    parser = argparse.ArgumentParser(
        prog=name,
        description=(
            "Convert EERIE ICON output to a local multiscales HEALPix Zarr "
            "with the current grid-doctor workflow."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="hist-1950-v20240618",
        choices=sorted(EERIE_PRESETS),
        help="Named EERIE dataset preset.",
    )
    parser.add_argument(
        "--item-url",
        help="Override the preset STAC item URL.",
    )
    parser.add_argument(
        "--asset",
        default="dkrz-disk",
        help="Asset name inside the STAC item.",
    )
    parser.add_argument(
        "--gridfile",
        default="/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc",
        help="ICON grid file used for the current Delaunay weight computation.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        help="Variables to convert. Defaults to the preset-specific first draft set.",
    )
    parser.add_argument(
        "--min-level",
        type=int,
        default=0,
        help="Minimum HEALPix level to write.",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        help="Maximum HEALPix level. Defaults to the level derived from the grid file resolution.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=12,
        help="Optional time chunk size before regridding.",
    )
    parser.add_argument(
        "--weights-cache",
        help="Optional cache file or directory for Delaunay weights.",
    )
    parser.add_argument(
        "--out-root",
        help=(
            "Local output root. The script writes <out-root>.zarr with a "
            "multiscales/zoom_<N> layout."
        ),
    )
    parser.add_argument(
        "--frequency-tag",
        help="Override the preset frequency tag used in the output path.",
    )
    parser.add_argument(
        "--stat-tag",
        help="Override the preset statistic tag used in the output path.",
    )
    parser.add_argument(
        "--apply-chunk-optimizer",
        action="store_true",
        help=(
            "Apply ChunkOptimizer before writing. Disabled by default because "
            "the rechunker flow is expected to handle the final chunking."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite an existing local output store.",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip Zarr metadata consolidation after writing.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for more: -v, -vv, -vvv).",
    )
    args = parser.parse_args(argv)
    gd.setup_logging(verbosity=args.verbose)
    return args


def main(name: str, argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(name, argv)
    run_pipeline(args)


if __name__ == "__main__":
    main(os.path.basename(sys.argv[0]), sys.argv[1:])
