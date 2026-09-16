"""Functionality related to multiscale zarr trees."""

import logging
import shutil
from pathlib import Path
from typing import Any

import xarray as xr
import zarr

__all__ = [
    "write_pyramid_to_multiscales_zarr",
]


logger = logging.getLogger(__name__)


def write_pyramid_to_multiscales_zarr(
    pyramid: dict[int, xr.Dataset],
    out_root: str | Path,
    root_attrs: dict[str, Any] | None = None,
    additional_attrs: dict[str, Any] | None = None,
    overwrite: bool = True,
    consolidated: bool = True,
) -> None:
    """Write the pyramid to a local multiscales Zarr tree.

    Args:
        pyramid: Healpix levels and data
        out_root: .zarr output store path
        root_attrs: Attributes for root group
        additional_attrs: Attributes added to all datasets
        overwrite: If out_root exists, replace, otherwise raise exception
        consolidated: Consolidate zarr metadata

    Raises:
        FileExistsError if overwrite is False and out_root exists.

    """
    root_path = Path(out_root)
    if root_path.exists():
        if not overwrite:
            raise FileExistsError(f"multiscales zarr exists: {root_path}. Skipping creation.")
        shutil.rmtree(root_path)

    max_level = max(map(int, pyramid))
    min_level = min(map(int, pyramid))

    root_attrs = root_attrs or {}
    additional_attrs = additional_attrs or {}

    root = zarr.open_group(root_path, mode="w")
    root.attrs.update(
        {
            "healpix_zoom_min": int(min_level),
            "healpix_zoom_max": int(max_level),
            "source_dataset_attrs": {str(key): str(value) for key, value in root_attrs.items()},
            **additional_attrs,
        }
    )

    multiscales_group = zarr.open_group(root_path / "multiscales", mode="a")
    multiscales_group.attrs.update(
        additional_attrs,
    )

    for level in sorted(pyramid, reverse=True):
        ds = pyramid[level]
        ds = ds.assign_attrs(
            healpix_zoom=int(level),
        )
        ds.attrs.update(additional_attrs)
        store_path = root_path / "multiscales" / f"zoom_{level}"
        logger.info("Writing zoom %d to %s", level, store_path)
        ds.to_zarr(
            store_path,
            mode="w",
            consolidated=consolidated,
        )

    if consolidated:
        zarr.consolidate_metadata(root_path)
