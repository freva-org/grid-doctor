"""Helper functions for preparing HEALPix pyramids for writing to zarr."""

from typing import Literal

import dask
import xarray as xr

from ..types import EncodingDict, IndexSlice, PyramidEncoding


def maybe_drop_coordinates(
    dataset: xr.Dataset,
    level: int,
    write_coords: bool | Literal["auto"] = "auto",
    max_level_to_auto_write: int = 10,
) -> xr.Dataset:
    """Drop coordinates from dataset if requested."""
    include_coords = (
        write_coords
        if isinstance(write_coords, bool)
        else level <= max_level_to_auto_write
    )
    if not include_coords:
        dataset = dataset.drop_vars(["latitude", "longitude", "cell"], errors="ignore")
        dataset.attrs["grid_doctor_implicit_coords"] = 1
    return dataset


def _raise_template_error() -> None:
    raise ValueError("cannot compute array values")


def get_common_coordinates(
    pyramid: dict[int, xr.Dataset],
) -> list[str]:
    """Find coordinates that are identical amongst given datasets."""
    common_coords = []
    coordinates = [ds.coords for ds in pyramid.values()]
    for c in coordinates[0]:
        if all(c in cs and coordinates[0][c].equals(cs[c]) for cs in coordinates):
            common_coords.append(str(c))
    return common_coords


def pyramid_to_multiscales_tree(
    pyramid: dict[int, xr.Dataset],
    common_coords: list[str] | None = None,
) -> xr.DataTree:
    """Create datatree from pyramid."""
    max_level = max(map(int, pyramid))
    min_level = min(map(int, pyramid))
    if common_coords:
        root = pyramid[min_level][common_coords]
    else:
        root = xr.Dataset()
    root.attrs.update(
        {
            "healpix_level_min": int(min_level),
            "healpix_level_max": int(max_level),
            "source_dataset_attrs": {
                str(key): str(value)
                for key, value in pyramid[min_level].attrs.items()
                if "healpix" not in key
            },
        }
    )
    tree = {
        "/": root,
    }
    for level, ds in pyramid.items():
        ds_mod = ds.assign_attrs(
            healpix_level=int(level),
        )
        tree[f"level_{level}"] = ds_mod

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
