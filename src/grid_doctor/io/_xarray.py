"""Helper functions for preparing HEALPix pyramids for writing to zarr."""

from typing import Literal

import dask
import xarray as xr

from ..metadata import (
    CELL_DIM,
    CRS_VAR,
    LAT_VAR,
    LON_VAR,
    HealpixMetadata,
    has_healpix_metadata,
    level_group,
)
from ..metadata._pyramid import pyramid_root_attrs
from ..types import EncodingDict, IndexSlice, PyramidEncoding


def maybe_drop_coordinates(
    dataset: xr.Dataset,
    level: int,
    write_coords: bool | Literal["auto"] = "auto",
    max_level_to_auto_write: int = 10,
) -> xr.Dataset:
    """Drop the coordinate arrays implied by the cell index, if requested.

    Only dense datasets are affected: position along ``cell`` equals the
    HEALPix index, so ``cell``, ``latitude`` and ``longitude`` carry no
    information and are dropped, and the dataset is marked as having
    implicit coordinates.  Sparse datasets keep everything, since their
    ``cell`` coordinate is the only record of which cells they hold.
    Datasets without HEALPix metadata just lose the coordinates.
    """
    include_coords = (
        write_coords
        if isinstance(write_coords, bool)
        else level <= max_level_to_auto_write
    )
    if include_coords:
        return dataset
    coords = [CELL_DIM, LAT_VAR, LON_VAR]
    if not has_healpix_metadata(dataset):
        return dataset.drop_vars(coords, errors="ignore")
    metadata = HealpixMetadata.from_dataset(dataset)
    if metadata.sparse:
        return dataset
    dataset = dataset.drop_vars(coords, errors="ignore")
    return metadata.replace(implicit_coords=True).apply(dataset)


def _raise_template_error() -> None:
    raise ValueError("cannot compute array values")


def get_common_coordinates(
    pyramid: dict[int, xr.Dataset],
) -> list[str]:
    """Find coordinates that are identical amongst given datasets.

    The ``crs`` grid mapping is never shared: its value is the same on
    every level, but its attributes describe each level's grid.
    """
    common_coords = []
    coordinates = [ds.coords for ds in pyramid.values()]
    for c in coordinates[0]:
        if c == CRS_VAR:
            continue
        if all(c in cs and coordinates[0][c].equals(cs[c]) for cs in coordinates):
            common_coords.append(str(c))
    return common_coords


def pyramid_to_multiscales_tree(
    pyramid: dict[int, xr.Dataset],
    common_coords: list[str] | None = None,
) -> xr.DataTree:
    """Create datatree from pyramid."""
    min_level = min(map(int, pyramid))
    if common_coords:
        root = pyramid[min_level][common_coords]
        # Selecting variables drags scalar coordinates (``crs``) along.
        root = root.drop_vars([name for name in root.variables if name not in common_coords])
    else:
        root = xr.Dataset()
    root.attrs = pyramid_root_attrs(pyramid)
    tree = {"/": root}
    for level, ds in pyramid.items():
        tree[level_group(level)] = ds

    dt = xr.DataTree.from_dict(tree)
    return dt


def prepare_template(
    dataset: xr.Dataset,
    level: int,
    write_coords: bool | Literal["auto"] = "auto",
    max_level_to_auto_write_coords: int = 10,
    encoding: PyramidEncoding | None = None,
    materialise: list[str] | None = None,
) -> tuple[xr.Dataset, EncodingDict]:
    """Create empty template based on dataset.

    Parameters listed under `materialise` will be written,
    all other left empty.
    """
    dataset = maybe_drop_coordinates(
        dataset,
        level,
        write_coords,
        max_level_to_auto_write=max_level_to_auto_write_coords,
    )
    lazy_vars = set(dataset.keys())
    if materialise:
        lazy_vars = {v for v in lazy_vars if v not in materialise}

    template = dataset.copy()
    template.update(dataset.drop_vars(lazy_vars).compute())

    delayed = dask.delayed(_raise_template_error)()  # type: ignore[attr-defined]
    for k, v in dataset.variables.items():
        if k in lazy_vars:
            # names of dask arrays are used for keeping track of results, so arrays
            # with the same name cannot have different shape or dtype
            name = f"template_{'x'.join(map(str, v.shape))}_{v.dtype}"
            template[k].data = dask.array.from_delayed(
                delayed, v.shape, v.dtype, name=name
            )

    # Remove old chunk encoding.
    for var in template.variables.values():
        if "chunks" in var.encoding:
            del var.encoding["chunks"]
    # Create regular chunks
    zarr_chunks: dict[str, tuple[int, ...]] = {}
    for var_name, variable in dataset.variables.items():
        chunksizes = variable.chunksizes
        zarr_chunks[str(var_name)] = tuple(c[0] for c in chunksizes.values())

    level_encoding = (encoding or {}).get(level, {})
    combined_encoding: EncodingDict = {}
    for k in template.variables:
        name = str(k)
        var_encoding = dict(level_encoding.get(name, {}))
        if name in zarr_chunks:
            var_encoding["chunks"] = zarr_chunks[name]
        combined_encoding[name] = var_encoding
    return template, combined_encoding


def prepare_region_dataset(dataset: xr.Dataset, region: dict[str, IndexSlice]) -> xr.Dataset:
    """Select given reqion from dataset.

    Variables that do not depend on the region keys are dropped.
    """
    region_keys = set(region)
    to_drop = (
        {
            name
            for name, var in dataset.data_vars.items()
            if region_keys.isdisjoint(set(var.dims))
        }
        | {str(dim) for dim in dataset.dims if str(dim) not in region_keys}
        | {str(coord) for coord in dataset.coords if str(coord) not in region_keys}
    )

    return dataset.drop_vars(to_drop, errors="ignore").isel(region)
