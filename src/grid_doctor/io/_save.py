"""Write HEALPix pyramids to Zarr stores, in one go or as parallel tasks."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal, overload

import s3fs
import xarray as xr

from ..metadata import level_group
from ..types import PyramidEncoding, Region, S3Options
from ._fanout import (
    IfExists,
    StoreExistsError,
    begin_init,
    describe_level,
    finish_init,
    inspect_target,
    resume_init,
)
from ._manifest import LevelSpec, PyramidManifest
from ._xarray import (
    get_common_coordinates,
    maybe_drop_coordinates,
    prepare_region_dataset,
    prepare_template,
    pyramid_to_multiscales_tree,
)
from ._zarr import (
    DeferredWrite,
    deferred_writes,
    get_filesystem,
    get_store,
    get_zarr_options,
    store_all,
)

logger = logging.getLogger(__name__)


WRITE_COORDS_MAX_LEVEL = 10
"""Highest level for which ``save_pyramid(write_coords="auto")``
materialises coordinate arrays.  At level 10 the two float64 coordinate
arrays cost ~200 MB per store; one level up they double, and by level 16
they would reach hundreds of GB while carrying no information that is not
already implied by the cell index."""


# ===================================================================
# S3 / Zarr output
# ===================================================================


@overload
def init_pyramid_store(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: S3Options | None = None,
    *,
    zarr_format: Literal[2, 3] = 2,
    encoding: PyramidEncoding | None = None,
    write_coords: bool | Literal["auto"] = "auto",
    materialise: list[str] | None = None,
    structure: Literal["flat"] = "flat",
    split: Mapping[str, int] | None = None,
    variables_per_task: int | None = None,
    if_exists: IfExists = "error",
) -> PyramidManifest: ...


@overload
def init_pyramid_store(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: S3Options | None = None,
    *,
    zarr_format: Literal[2, 3] = 2,
    encoding: PyramidEncoding | None = None,
    write_coords: bool | Literal["auto"] = "auto",
    materialise: list[str] | None = None,
    structure: Literal["multiscales"],
    split: Mapping[str, int] | None = None,
    variables_per_task: int | None = None,
    if_exists: IfExists = "error",
) -> None: ...


def init_pyramid_store(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: S3Options | None = None,
    *,
    zarr_format: Literal[2, 3] = 2,
    encoding: PyramidEncoding | None = None,
    write_coords: bool | Literal["auto"] = "auto",
    materialise: list[str] | None = None,
    structure: Literal["flat", "multiscales"] = "flat",
    split: Mapping[str, int] | None = None,
    variables_per_task: int | None = None,
    if_exists: IfExists = "error",
) -> PyramidManifest | None:
    """Create empty pyramid stores and plan the work to fill them.

    This is step one of a fan-out write.  It writes, for every level,
    ``"<path>/level_<level>.zarr"`` with all metadata, encodings and
    coordinates, but none of the (lazy) data variables, and returns a
    [`PyramidManifest`][grid_doctor.PyramidManifest] that splits the
    remaining work into independent
    [`WriteTask`][grid_doctor.WriteTask] units.  Hand each task to one
    worker, which calls
    [`write_pyramid_task`][grid_doctor.write_pyramid_task]; afterwards
    [`check_pyramid_store`][grid_doctor.check_pyramid_store] tells
    whether everything was written.  The manifest is also stored as
    ``"<path>/_grid_doctor/manifest.json"``.

    No data is computed (apart from coordinates and *materialise*), so
    the init is cheap even for stores with hundreds of thousands of
    chunks.  The pyramid only needs the right shapes, dtypes, chunks and
    attributes; it can be built lazily from the full source.

    What happens when *path* already holds a pyramid is controlled by
    *if_exists*; by default the init refuses, so a second run cannot
    destroy the stores of a first one by accident.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset, typically the lazy result
        of [`create_healpix_pyramid`][grid_doctor.create_healpix_pyramid].
    path:
        Target prefix.  An ``"s3://bucket/pyramid"`` URL writes to S3; any
        other value is treated as a local directory path.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.  Only used for S3
        targets; ignored (and optional) for local paths.  Never stored in
        the manifest.
    zarr_format:
        Zarr format version.
    encoding:
        Per-level encoding dictionaries.  Chunking always follows the
        dask chunks of the pyramid (first chunk along each dimension).
    write_coords:
        Whether to materialise the ``cell``/``latitude``/``longitude``
        coordinate arrays in the store.  HEALPix coordinates are a pure
        function of the cell index, and above roughly level 10 the
        arrays dwarf regional payloads (hundreds of GB at level 16, and
        they are never fill values, so chunk elision cannot help).
        ``"auto"`` (default) writes coordinates for levels up to
        :data:`WRITE_COORDS_MAX_LEVEL` and omits them above; ``True`` /
        ``False`` force either behaviour for all levels.  Coordinate-
        less stores keep the ``crs`` grid mapping, are marked as having
        implicit coordinates (see
        [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata]), and are
        meant to be accessed through the region selectors
        ([`select_bbox`][grid_doctor.select_bbox],
        [`select_cells`][grid_doctor.select_cells]), which reconstruct
        coordinates for exactly the cells they return.
    materialise:
        Data variables to compute and write completely during the init,
        e.g. small static fields.  They are not part of any task.
    structure:
        ``"flat"`` (default) writes one store per level,
        ``"<path>/level_<level>.zarr"``, and plans the fan-out tasks.
        ``"multiscales"`` writes the template of a single datatree store
        at *path* (one group per level) and returns ``None``: fan-out
        tasks are not supported for it yet, so *split*,
        *variables_per_task* and ``if_exists="resume"`` are rejected, and
        an existing store at *path* is overwritten.  Fill it with
        [`save_pyramid`][grid_doctor.save_pyramid] and
        ``structure="multiscales"``.
    split:
        Task size per dimension in index units, e.g. ``{"time": 24}``
        for tasks of 24 time steps.  Sizes are rounded **up** to a
        multiple of the zarr chunk (or shard) size along that dimension,
        taken over all variables and levels, so no two tasks ever write
        to the same chunk.  Several dimensions give the cartesian
        product of their regions.  Splitting along ``cell`` is not
        supported.  ``None`` (default) plans one task per group of
        variables, covering the whole store.
    variables_per_task:
        Maximum number of variables per task.  ``None`` (default) keeps
        all variables that share split dimensions and levels in one
        task, so the regridding they share is computed once per region.
    if_exists:
        What to do if *path* already has a manifest, level stores, or the
        remains of an interrupted init:

        - ``"error"`` (default): raise
          [`StoreExistsError`][grid_doctor.io.StoreExistsError].
        - ``"resume"``: if the stored manifest was planned exactly as
          this call would plan it (same path, levels, array layouts,
          split and tasks), return it unchanged, with its
          ``manifest_id``, and write nothing.  Tasks that already
          finished keep their markers, so re-submitting a conversion
          after a time-out only redoes the missing tasks.  Raises
          [`StoreExistsError`][grid_doctor.io.StoreExistsError] listing
          the differences otherwise, or if there is no readable
          manifest to resume (e.g. the previous init was killed).
          Without anything at *path* it initialises normally.
        - ``"overwrite"``: re-initialise.  All data is discarded, the
          result gets a new ``manifest_id``, and tasks and markers of the
          previous manifest become invalid; workers still running for
          it fail with ``StaleTaskError``.  Only overwrite when no
          worker of a previous run may still be writing: grid-doctor
          does not lock the store, and a worker that is already writing
          can put its data into the new stores before it notices.

    Returns
    -------
    PyramidManifest or None
        Store layout and tasks.  JSON-serialisable via
        ``manifest.to_json()`` and accepted directly as a reflow task
        result.  ``None`` for ``structure="multiscales"``.

    Examples
    --------
    ```python
    pyramid = create_healpix_pyramid(ds, max_level=9)
    manifest = init_pyramid_store(pyramid, "/scratch/out", split={"time": 24})
    for task in manifest.tasks:  # one array-job element each
        write_pyramid_task(pyramid, task)
    check_pyramid_store(manifest).raise_if_incomplete()
    ```
    """
    if if_exists not in ("error", "resume", "overwrite"):
        raise ValueError(f"if_exists must be 'error', 'resume' or 'overwrite', got {if_exists!r}")
    if structure not in ("flat", "multiscales"):
        raise ValueError(f"structure must be 'flat' or 'multiscales', got {structure!r}")
    materialise = list(materialise or [])
    if structure == "multiscales":
        unsupported = [
            name
            for name, given in (
                ("split", split is not None),
                ("variables_per_task", variables_per_task is not None),
                ("if_exists='resume'", if_exists == "resume"),
            )
            if given
        ]
        if unsupported:
            raise NotImplementedError(
                f"{', '.join(unsupported)} not supported for "
                "structure='multiscales' yet; use structure='flat' for fan-out "
                "writes"
            )
        _init_multiscales(
            pyramid,
            path,
            s3_options,
            zarr_format=zarr_format,
            encoding=encoding,
            write_coords=write_coords,
            materialise=materialise,
        )
        return None

    # Workers rebuild store paths from the manifest; keep them identical.
    path = path.rstrip("/")
    existing = inspect_target(path, s3_options, [f"level_{level}.zarr" for level in pyramid])
    if existing.exists and if_exists == "error":
        raise StoreExistsError(
            f"{path} already holds a pyramid ({existing.describe()}).  Pass "
            "if_exists='resume' to continue it or if_exists='overwrite' to "
            "start over."
        )
    if existing.exists and if_exists == "resume":
        if existing.manifest is None:
            raise StoreExistsError(
                f"cannot resume {path}: there is no readable manifest "
                f"({existing.describe()}).  Use if_exists='overwrite'."
            )
        templates = {
            level: maybe_drop_coordinates(ds, level, write_coords, WRITE_COORDS_MAX_LEVEL)
            for level, ds in pyramid.items()
        }
        return resume_init(
            existing.manifest,
            path,
            templates,
            {level: [str(k) for k in ds.data_vars if str(k) not in materialise] for level, ds in templates.items()},
            s3_options,
            zarr_format=zarr_format,
            split=split,
            variables_per_task=variables_per_task,
            materialised=[m for m in materialise if m in {str(v) for ds in pyramid.values() for v in ds}],
        )
    fs = get_filesystem(path, s3_options)
    begin_init(path, s3_options)
    level_specs: dict[int, LevelSpec] = {}
    for level, dataset in pyramid.items():
        template, combined_encoding = prepare_template(
            dataset,
            level,
            write_coords=write_coords,
            max_level_to_auto_write_coords=WRITE_COORDS_MAX_LEVEL,
            encoding=encoding,
            materialise=materialise,
        )
        task_vars = [str(k) for k in template.data_vars if str(k) not in materialise]
        store_name = f"level_{level}.zarr"
        root_path = f"{path}/{store_name}"
        store = get_store(root_path, fs)
        logger.info("Writing HEALPix to %s", root_path)
        zarr_options = get_zarr_options("w", zarr_format, combined_encoding)
        template.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
        level_specs[level] = describe_level(
            level,
            store_name,
            store,
            template,
            task_vars,
            zarr_format=zarr_format,
        )

    unknown = sorted(set(materialise) - {str(v) for ds in pyramid.values() for v in ds})
    if unknown:
        logger.warning("materialise names unknown variables: %s", ", ".join(unknown))
    return finish_init(
        path,
        s3_options,
        [level_specs[lvl] for lvl in sorted(level_specs, reverse=True)],
        zarr_format=zarr_format,
        split=split,
        variables_per_task=variables_per_task,
        materialised=[m for m in materialise if m not in unknown],
    )


def _init_multiscales(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: S3Options | None,
    *,
    zarr_format: Literal[2, 3],
    encoding: PyramidEncoding | None,
    write_coords: bool | Literal["auto"],
    materialise: list[str],
) -> None:
    """Write the template of a multiscales datatree store (no data)."""
    fs = get_filesystem(path, s3_options)
    templates = {}
    encodings = {}
    common_coords = get_common_coordinates(pyramid)
    for level, dataset in pyramid.items():
        template, combined_encoding = prepare_template(
            dataset,
            level,
            write_coords=write_coords,
            max_level_to_auto_write_coords=WRITE_COORDS_MAX_LEVEL,
            encoding=encoding,
            materialise=materialise,
        )
        templates[level] = template
        encodings[f"/{level_group(level)}"] = combined_encoding

    # Remove duplicate encoding for common variables
    common_encodings = {}
    for enc in encodings.values():
        for c in common_coords:
            common_encodings[c] = enc.pop(c)
    encodings.update({"/": common_encodings})

    tree_template = pyramid_to_multiscales_tree(templates, common_coords=common_coords)
    zarr_options = get_zarr_options("w", zarr_format, encodings)
    store = get_store(path, fs)
    logger.info("Writing HEALPix to %s", store)
    tree_template.to_zarr(store, **zarr_options)  # type: ignore[call-overload]


def save_pyramid(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: S3Options | None = None,
    *,
    mode: Literal["a", "w", "r+"] = "a",
    region: Literal["auto"] | Region = "auto",
    zarr_format: Literal[2, 3] = 2,
    encoding: PyramidEncoding | None = None,
    write_coords: bool | Literal["auto"] = "auto",
    structure: Literal["flat", "multiscales"] = "flat",
) -> None:
    """Write a HEALPix pyramid to Zarr stores on S3 or local disk.

    Each level is stored below ``"<path>/level_<level>.zarr"``.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset.
    path:
        Target prefix.  An ``"s3://bucket/pyramid"`` URL writes to S3; any
        other value is treated as a local directory path.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.  Only used for S3
        targets; ignored (and optional) for local paths.
    mode:
        Zarr write mode.
    region:
        Region writes for partial updates.
    zarr_format:
        Zarr format version.
    encoding:
        Per-level encoding dictionaries.
    write_coords:
        Whether to materialise the ``cell``/``latitude``/``longitude``
        coordinate arrays in the store.  HEALPix coordinates are a pure
        function of the cell index, and above roughly level 10 the
        arrays dwarf regional payloads (hundreds of GB at level 16, and
        they are never fill values, so chunk elision cannot help).
        ``"auto"`` (default) writes coordinates for levels up to
        :data:`WRITE_COORDS_MAX_LEVEL` and omits them above; ``True`` /
        ``False`` force either behaviour for all levels.  Coordinate-
        less stores keep the ``crs`` grid mapping, are marked as having
        implicit coordinates (see
        [`HealpixMetadata`][grid_doctor.metadata.HealpixMetadata]), and are
        meant to be accessed through the region selectors
        ([`select_bbox`][grid_doctor.select_bbox],
        [`select_cells`][grid_doctor.select_cells]), which reconstruct
        coordinates for exactly the cells they return.
    structure: Specify whether to write the datasets in the pyramid to
        separate stores ('flat'), or combine them into a datatree
        ('multiscales').
    """
    fs = get_filesystem(path, s3_options)

    if structure == "multiscales":
        _save_multiscale(
            pyramid,
            path,
            fs,
            mode,
            region,
            zarr_format,
            encoding,
            write_coords,
        )
        return
    # Stores are initialised per level; the data of all levels is written
    # in one pass (see grid_doctor.io._zarr for why).
    deferred: list[DeferredWrite] = []
    encoding = encoding or {}
    for level, dataset in pyramid.items():
        dataset = maybe_drop_coordinates(dataset, level, write_coords, WRITE_COORDS_MAX_LEVEL)
        store_name = f"level_{level}"
        root_path = f"{path}/{store_name}.zarr"
        store = get_store(root_path, fs)
        logger.info("Writing HEALPix to %s", store)
        zarr_options = get_zarr_options(mode, zarr_format, encoding.get(level, None))

        if region == "auto":
            dataset.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
            written, write_region = dataset, None
        else:
            # Fill in the region of data.
            written = prepare_region_dataset(dataset, region)
            written.to_zarr(  # type: ignore[call-overload]
                store,
                region=region,
                **zarr_options,
            )
            write_region = region
        deferred.extend(
            deferred_writes(
                written,
                store,
                encoding=zarr_options.get("encoding"),
                region=write_region,
                zarr_format=zarr_format,
            )
        )
    store_all(deferred)


def _save_multiscale(
    pyramid: dict[int, xr.Dataset],
    path: str,
    fs: s3fs.S3FileSystem | None,
    mode: Literal["a", "w", "r+"],
    region: Literal["auto"] | Region,
    zarr_format: Literal[2, 3],
    encoding: PyramidEncoding | None,
    write_coords: bool | Literal["auto"],
) -> None:
    store = get_store(path, fs)
    encoding = encoding or {}

    dropped_pyramid = {
        lev: maybe_drop_coordinates(ds, lev, write_coords, WRITE_COORDS_MAX_LEVEL) for lev, ds in pyramid.items()
    }
    if region == "auto":
        template = pyramid_to_multiscales_tree(dropped_pyramid)
        write_region = None
    else:
        # Fill in the region of data.
        templates = {}
        for level, dataset in dropped_pyramid.items():
            templates[level] = prepare_region_dataset(dataset, region)

        template = pyramid_to_multiscales_tree(templates, common_coords=list(region.keys()))
        write_region = region

    encodings = {f"/{level_group(level)}": enc for level, enc in encoding.items()}
    zarr_options = get_zarr_options(mode, zarr_format, encodings)
    # Deferred writes not yet implemented for multiscales regions
    zarr_options["compute"] = True
    logger.info("Writing HEALPix to %s", store)
    template.to_zarr(  # type: ignore[call-overload]
        store,
        region=write_region,
        **zarr_options,
    )
