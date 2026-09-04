"""Cleanup helpers for ERA5/ERA5-Land HEALPix outputs."""

import gc
import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import xarray as xr
import zarr

from .formatter import dataset_output_root, existing_destinations_for_frequency
from .zarr_publisher import _rewrite_dataset_via_temp

LOGGER = logging.getLogger(__name__)
LEVEL_RE = re.compile(r"level_(?P<level>\d+)\.zarr$")


def existing_level_destinations(
    dataset: str,
    frequency: str,
    *,
    output_path: str | Path | None = None,
) -> list[tuple[int, Path]]:
    """Return existing HEALPix Zarr stores for one frequency with parsed levels."""

    destinations: list[tuple[int, Path]] = []
    for destination in existing_destinations_for_frequency(
        dataset,
        frequency,
        output_path=output_path,
    ):
        match = LEVEL_RE.search(destination)
        if match is None:
            continue
        destinations.append((int(match.group("level")), Path(destination)))
    return sorted(destinations, reverse=True)


def selected_level_destinations(
    dataset: str,
    frequency: str,
    *,
    levels: tuple[int, ...] | None,
    output_path: str | Path | None = None,
) -> list[tuple[int, Path]]:
    """Return the existing level stores selected for one frequency."""

    destinations = existing_level_destinations(
        dataset,
        frequency,
        output_path=output_path,
    )
    if levels is None:
        return destinations
    requested = set(levels)
    return [(level, path) for level, path in destinations if level in requested]


def _time_axis_info(zarr_array) -> tuple[int, int] | None:
    """Return the time-axis position and chunk length for one Zarr array."""

    dims = tuple(str(dim) for dim in zarr_array.attrs.get("_ARRAY_DIMENSIONS", ()))
    if "time" not in dims:
        return None

    axis = dims.index("time")
    return axis, int(zarr_array.chunks[axis])


def _shrink_time_arrays_in_place(destination: str, keep_size: int) -> bool:
    """Shrink all time-bearing arrays in one store without rewriting kept chunks.

    Zarr's in-place ``resize`` removes chunks that fall completely outside the
    new shape, which makes tail truncation much cheaper than rebuilding the
    entire retained dataset. Boundary chunks are intentionally left intact by
    Zarr when they straddle the new array edge; those now-inaccessible values
    remain hidden unless the store is later expanded again.
    """

    root = zarr.open_group(destination, mode="a")
    changed = False

    for name in root.array_keys():
        zarr_array = cast(zarr.Array[Any], root[name])
        time_axis = _time_axis_info(zarr_array)
        if time_axis is None:
            continue

        axis, _time_chunk = time_axis
        old_size = int(zarr_array.shape[axis])
        if keep_size >= old_size:
            continue

        new_shape = list(zarr_array.shape)
        new_shape[axis] = keep_size
        zarr_array.resize(tuple(new_shape))
        changed = True

    if changed:
        zarr.consolidate_metadata(destination)
    return changed


def truncate_zarr_store_after(
    destination: str,
    *,
    cutoff: str,
    zarr_format: int,
) -> bool:
    """Remove timestamps strictly after ``cutoff`` from one existing Zarr store.

    Parameters
    ----------
    destination:
        Path to the target Zarr store.
    cutoff:
        Inclusive ISO date string used as the upper bound of the retained time
        selection.
    zarr_format:
        Output Zarr format version used when opening the truncated store.

    Returns
    -------
    bool
        ``True`` when the store was rewritten, else ``False``.
    """

    path = Path(destination)
    if not path.exists():
        return False

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))
    try:
        if "time" not in existing.dims:
            return False

        original_size = existing.sizes.get("time", 0)
        retained_time = existing["time"].sel(time=slice(None, cutoff))
        truncated_size = retained_time.sizes.get("time", 0)
        if truncated_size == original_size:
            return False

        LOGGER.info(
            "✂️ Truncating existing Zarr store %s after %s (time: %s -> %s)",
            destination,
            cutoff,
            original_size,
            truncated_size,
        )
        return _shrink_time_arrays_in_place(destination, truncated_size)
    finally:
        existing.close()
        gc.collect()


def drop_variables_from_zarr_store(
    destination: str,
    *,
    variable_names: Iterable[str],
    zarr_format: int = 2,
) -> tuple[bool, tuple[str, ...], bool]:
    """Remove named data variables from one existing Zarr store.

    Parameters
    ----------
    destination:
        Path to the target Zarr store.
    variable_names:
        Variable names to remove from the store.
    zarr_format:
        Output Zarr format version used when opening dataset metadata.

    Returns
    -------
    tuple[bool, tuple[str, ...], bool]
        A tuple containing ``(changed, removed_names, deleted_store)``.
    """

    path = Path(destination)
    if not path.exists():
        return False, (), False

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2), chunks=None)
    try:
        requested = {str(name) for name in variable_names}
        removable = tuple(name for name in existing.data_vars if name in requested)
        if not removable:
            return False, (), False

        if len(removable) == len(existing.data_vars):
            existing.close()
            existing = None
            shutil.rmtree(path)
            return True, removable, True

        root = zarr.open_group(destination, mode="a")
        for name in removable:
            if name in root:
                del root[name]
        zarr.consolidate_metadata(destination)
        return True, removable, False
    finally:
        if existing is not None:
            existing.close()
        gc.collect()


def _drop_pressure_levels_from_zarr_store(
    *,
    destination: str,
    variable_names: tuple[str, ...] | None,
    pressure_levels: tuple[int, ...] | None,
    zarr_format: int = 2,
) -> bool:
    """Remove selected hPa levels from one store's pressure-level variables."""

    path = Path(destination)
    if not path.exists():
        return False

    requested_variables = set(variable_names) if variable_names is not None else None
    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2), chunks=None)
    try:
        pressure_variables = [
            name
            for name, data in existing.data_vars.items()
            if "plev" in data.dims and (requested_variables is None or name in requested_variables)
        ]
        if not pressure_variables or "plev" not in existing.coords:
            return False

        selected_levels = set(pressure_levels or ())
        available_levels = {int(value) for value in existing["plev"].values}
        if pressure_levels is not None and not selected_levels.intersection(available_levels):
            return False
        remaining = (
            []
            if pressure_levels is None
            else [value for value in existing["plev"].values if int(value) not in selected_levels]
        )
        if pressure_levels is None or not remaining:
            cleaned = existing.drop_vars(pressure_variables)
        else:
            cleaned = existing.sel(plev=remaining)

        if not any("plev" in data.dims for data in cleaned.data_vars.values()):
            cleaned = cleaned.drop_vars("plev", errors="ignore")
        _rewrite_dataset_via_temp(
            cleaned,
            destination,
            zarr_format=zarr_format,
        )
        return True
    finally:
        existing.close()
        gc.collect()


def clean_frequency_stores(
    *,
    dataset: str,
    frequency: str,
    variable_names: tuple[str, ...] | None,
    pressure_levels: tuple[int, ...] | None = None,
    levels: tuple[int, ...] | None,
    zarr_format: int = 2,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Clean named variables or pressure levels from selected stores for one frequency."""

    if variable_names is None and pressure_levels is None:
        raise ValueError("A variable or pressure-level selection is required to clean frequency stores.")

    actions: list[str] = []
    for level, destination in selected_level_destinations(
        dataset,
        frequency,
        levels=levels,
        output_path=output_path,
    ):
        if pressure_levels is not None:
            selection = ",".join(map(str, pressure_levels))
            scope = "all pressure-level variables" if variable_names is None else ",".join(variable_names)
            if dry_run:
                actions.append(
                    f"would remove pressure levels {selection} from {scope} in {destination} (level {level})"
                )
            elif _drop_pressure_levels_from_zarr_store(
                destination=str(destination),
                variable_names=variable_names,
                pressure_levels=pressure_levels,
                zarr_format=zarr_format,
            ):
                actions.append(f"❌ removed pressure levels {selection} from {destination} (level {level})")
            continue

        assert variable_names is not None
        if dry_run:
            actions.append(f"would remove variables {','.join(variable_names)} from {destination} (level {level})")
            continue

        changed, removed, deleted_store = drop_variables_from_zarr_store(
            str(destination),
            variable_names=variable_names,
            zarr_format=zarr_format,
        )
        if not changed:
            continue
        if deleted_store:
            actions.append(f"❌ deleted {destination} after removing all variables: {','.join(removed)}")
        else:
            actions.append(f"❌ removed variables {','.join(removed)} from {destination} (level {level})")
    return actions


def truncate_frequency_destinations(
    *,
    dataset: str,
    frequency: str,
    zarr_format: int = 2,
    cutoff: str,
    highest_level_only: bool = False,
    output_path: str | Path | None = None,
) -> int:
    """Truncate the selected existing destinations for one output frequency."""

    destinations = existing_level_destinations(
        dataset,
        frequency,
        output_path=output_path,
    )
    if highest_level_only and destinations:
        destinations = destinations[:1]

    truncated_count = 0
    for _level, destination in destinations:
        if truncate_zarr_store_after(
            str(destination),
            cutoff=cutoff,
            zarr_format=zarr_format,
        ):
            truncated_count += 1
    return truncated_count


def truncate_existing_healpix_stores(
    *,
    dataset: str,
    frequencies: tuple[str, ...],
    cutoff: str,
    zarr_format: int = 2,
    highest_level_only: bool = False,
    output_path: str | Path | None = None,
) -> int:
    """Truncate existing time-based HEALPix Zarr stores before a rerun.

    The truncation selection follows the CLI write targets. When
    ``highest_level_only`` is enabled, only the highest existing zoom level for
    each frequency is truncated.
    """

    truncated_count = 0
    for frequency in frequencies:
        truncated_count += truncate_frequency_destinations(
            dataset=dataset,
            frequency=frequency,
            zarr_format=zarr_format,
            cutoff=cutoff,
            highest_level_only=highest_level_only,
            output_path=output_path,
        )
    return truncated_count


def delete_frequency_level_stores(
    *,
    dataset: str,
    frequency: str,
    levels: tuple[int, ...],
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Delete selected whole level stores for one output frequency."""

    actions: list[str] = []
    for level, destination in selected_level_destinations(
        dataset,
        frequency,
        levels=levels,
        output_path=output_path,
    ):
        if dry_run:
            actions.append(f"would delete {destination} (level {level})")
            continue
        shutil.rmtree(destination)
        actions.append(f"❌ deleted {destination} (level {level})")
    return actions


def delete_frequency_directory(
    *,
    dataset: str,
    frequency: str,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Delete one whole published output-frequency directory if it exists."""

    destinations = existing_level_destinations(
        dataset,
        frequency,
        output_path=output_path,
    )
    if not destinations:
        return []

    frequency_dir = destinations[0][1].parent
    if dry_run:
        return [f"would delete frequency directory {frequency_dir}"]

    shutil.rmtree(frequency_dir)
    return [f"❌ deleted frequency directory {frequency_dir}"]


def delete_dataset_root(
    *,
    dataset: str,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Delete the whole dataset output root if it exists."""

    root_path = dataset_output_root(dataset, output_path=output_path)
    if not root_path.exists():
        return []
    if dry_run:
        return [f"would delete dataset output root {root_path}"]
    shutil.rmtree(root_path)
    return [f"❌ deleted dataset output root {root_path}"]
