from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import xarray as xr
import uxarray as ux
import zarr
from numcodecs import Blosc
from tqdm import tqdm


# =============================================================================
# low-level helpers
# =============================================================================

def _blosc_lz4():
    return Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)


def _fill_value_for_dtype(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return None


# =============================================================================
# time encode/decode helpers
# =============================================================================

def _decode_time_like(da: xr.DataArray) -> np.ndarray:
    """
    Return time values as decoded datetime64[ns] where possible.
    Falls back to raw values if CF decoding is not possible.
    """
    vals = np.asarray(da.values)

    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]")

    try:
        tmp = xr.decode_cf(xr.Dataset(coords={"time": da}))
        vals2 = np.asarray(tmp["time"].values)
        if np.issubdtype(vals2.dtype, np.datetime64):
            return vals2.astype("datetime64[ns]")
        return vals2
    except Exception:
        return vals


def _prepare_time_storage(time_coord: xr.DataArray):
    """
    Prepare values/attrs for storing time in a way that xarray can naturally decode.

    Strategy:
      - if source time is datetime64, store as int64 offsets since Unix epoch
        using either seconds or nanoseconds depending on alignment
      - if source time is already numeric, store as-is and preserve attrs/encoding
    """
    decoded = _decode_time_like(time_coord)

    attrs_in = dict(time_coord.attrs)
    enc_in = dict(getattr(time_coord, "encoding", {}) or {})

    if np.issubdtype(np.asarray(decoded).dtype, np.datetime64):
        ns = np.asarray(decoded).astype("datetime64[ns]").astype(np.int64)

        if np.all(ns % 1_000_000_000 == 0):
            values = (ns // 1_000_000_000).astype(np.int64)
            units = "seconds since 1970-01-01 00:00:00"
        else:
            values = ns.astype(np.int64)
            units = "nanoseconds since 1970-01-01 00:00:00"

        calendar = (
            enc_in.get("calendar")
            or attrs_in.get("calendar")
            or "proleptic_gregorian"
        )

        attrs = {
            "_ARRAY_DIMENSIONS": ["time"],
            "units": str(units),
            "calendar": str(calendar),
            "axis": "T",
        }
        return values, np.int64, attrs

    # numeric fallback
    raw = np.asarray(time_coord.values)
    attrs = {"_ARRAY_DIMENSIONS": ["time"], "axis": "T"}
    for key in ["units", "calendar", "long_name", "standard_name"]:
        if key in attrs_in:
            attrs[key] = attrs_in[key]
        elif key in enc_in:
            attrs[key] = enc_in[key]

    return raw, raw.dtype, attrs


def _write_time_coordinate(store_path: str | Path, time_coord: xr.DataArray) -> None:
    """
    Create/overwrite the time coordinate manually with zarr-python.
    """
    store_path = Path(store_path)
    root = zarr.open_group(store_path, mode="a")

    if "time" in root:
        del root["time"]

    values, dtype, attrs = _prepare_time_storage(time_coord)
    values = np.asarray(values)

    arr = root.create_dataset(
        "time",
        shape=(int(values.size),),
        chunks=(max(1, int(values.size)),),
        dtype=dtype,
        compressor=_blosc_lz4(),
        fill_value=None,
        order="C",
        overwrite=False,
        dimension_separator="/",
    )
    arr[:] = values
    arr.attrs.update(attrs)


def _read_existing_time_decoded(store_path: str | Path) -> np.ndarray:
    """
    Read existing store time and decode robustly.
    """
    ds_existing = xr.open_zarr(store_path, consolidated=False)
    if "time" not in ds_existing.coords:
        raise RuntimeError(f"Existing store has no 'time' coordinate: {store_path}")
    return _decode_time_like(ds_existing["time"])


# =============================================================================
# naming
# =============================================================================

def make_zoom_store_path(
    *,
    out_root: str | Path,
    dataset_id: str,
    frequency_tag: str,
    stat_tag: str,
    zoom: int,
) -> str:
    return str(Path(out_root) / f"{dataset_id}_{frequency_tag}_{stat_tag}_{zoom}.zarr")


# =============================================================================
# metadata helpers
# =============================================================================

def _crs_attrs_for_healpix(*, zoom: int, nest: bool) -> dict:
    return {
        "grid_mapping_name": "healpix",
        "healpix_nside": int(2**zoom),
        "healpix_order": "nest" if bool(nest) else "ring",
    }


def _append_cell_mean_to_cell_methods(attrs: dict) -> dict:
    attrs = dict(attrs or {})
    cm = attrs.get("cell_methods")
    if cm is None:
        attrs["cell_methods"] = "cell: mean"
        return attrs

    cm_str = str(cm)
    if "cell:" not in cm_str:
        attrs["cell_methods"] = cm_str.strip() + " cell: mean"
    return attrs


def _normalize_var_attrs(
    source_var: xr.DataArray,
    *,
    prepared,
    zoom: int,
    nest: bool,
    pyramid_source_zoom: int,
    pyramid_aggregation: str,
) -> dict:
    attrs = dict(source_var.attrs)
    attrs = _append_cell_mean_to_cell_methods(attrs)

    attrs["grid_mapping"] = "crs"
    attrs["healpix_zoom"] = int(zoom)
    attrs["healpix_nest"] = bool(nest)
    attrs["src_grid_hash"] = prepared.src_grid_hash
    attrs["dst_grid_hash"] = prepared.dst_grid_hash if zoom == prepared.zoom else attrs.get("dst_grid_hash", "")
    attrs["pair_hash"] = prepared.pair_hash
    attrs["weight_file"] = prepared.weight_file
    attrs["source_mode"] = prepared.source_info.get("source_mode", "unknown")
    attrs["source_path_used"] = prepared.source_info.get("source_path_used", "unknown")
    attrs["pyramid_source_zoom"] = int(pyramid_source_zoom)
    attrs["pyramid_aggregation"] = str(pyramid_aggregation)
    return attrs


# =============================================================================
# progress helpers
# =============================================================================

def _pyramid_progress_path(max_store_path: str | Path, var_name: str, min_zoom: int) -> Path:
    return Path(max_store_path) / f".pyramid_progress_{var_name}_to_{min_zoom}.json"


def _load_pyramid_progress(max_store_path: str | Path, var_name: str, min_zoom: int) -> dict:
    p = _pyramid_progress_path(max_store_path, var_name, min_zoom)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"next_time_index": 0, "completed": False}


def _save_pyramid_progress(
    max_store_path: str | Path,
    var_name: str,
    min_zoom: int,
    next_time_index: int,
    completed: bool,
) -> None:
    p = _pyramid_progress_path(max_store_path, var_name, min_zoom)
    with open(p, "w") as f:
        json.dump(
            {
                "next_time_index": int(next_time_index),
                "completed": bool(completed),
            },
            f,
            indent=2,
        )


# =============================================================================
# store initialization
# =============================================================================

def _ensure_store_base(
    *,
    store_path: str | Path,
    source_ds: xr.Dataset,
    zoom: int,
    nest: bool,
    dataset_id: str | None = None,
    frequency_tag: str | None = None,
    stat_tag: str | None = None,
) -> None:
    """
    Create base store with:
      - time coordinate (manual CF-style encoding)
      - crs variable (manual with Zarr v2 '/' separator)
    If store exists, validate decoded time and add crs if missing.
    """
    store_path = Path(store_path)
    time_coord = source_ds["time"]

    root_attrs = {
        "healpix_zoom": int(zoom),
        "healpix_nest": bool(nest),
        "healpix_nside": int(2**zoom),
    }
    if dataset_id is not None:
        root_attrs["dataset_id"] = dataset_id
    if frequency_tag is not None:
        root_attrs["frequency"] = frequency_tag
    if stat_tag is not None:
        root_attrs["statistic"] = stat_tag

    mode = "w" if not store_path.exists() else "a"
    root = zarr.open_group(store_path, mode=mode)
    root.attrs.update(root_attrs)

    source_time = _decode_time_like(time_coord)

    # ensure time coordinate exists and matches
    if "time" not in root:
        _write_time_coordinate(store_path, time_coord)
    else:
        existing_time = _read_existing_time_decoded(store_path)
        if existing_time.shape != source_time.shape or not np.array_equal(existing_time, source_time):
            raise RuntimeError(
                "Existing store time coordinate does not match source dataset time axis.\n"
                f"store : len={existing_time.size}, first={existing_time[0]}, last={existing_time[-1]}\n"
                f"source: len={source_time.size}, first={source_time[0]}, last={source_time[-1]}"
            )

    # ensure crs exists
    root = zarr.open_group(store_path, mode="a")
    if "crs" not in root:
        arr = root.create_dataset(
            "crs",
            shape=(1,),
            chunks=(1,),
            dtype=np.float32,
            compressor=_blosc_lz4(),
            fill_value=np.nan,
            order="C",
            overwrite=False,
            dimension_separator="/",
        )
        arr[:] = np.array([0.0], dtype=np.float32)
        arr.attrs["_ARRAY_DIMENSIONS"] = ["crs"]
        arr.attrs.update(_crs_attrs_for_healpix(zoom=zoom, nest=nest))


def _ensure_var_array_exists(
    *,
    store_path: str | Path,
    var_name: str,
    ntime: int,
    ncell: int,
    dtype,
    attrs: dict,
    cell_chunk: int = 65536,
) -> None:
    """
    Create an empty 2D array (time, cell) manually with zarr-python.
    """
    store_path = Path(store_path)
    root = zarr.open_group(store_path, mode="a")
    if var_name in root.array_keys():
        return

    dtype = np.dtype(dtype)
    cell_chunk = min(int(cell_chunk), int(ncell))

    arr = root.create_dataset(
        var_name,
        shape=(int(ntime), int(ncell)),
        chunks=(1, cell_chunk),
        dtype=dtype,
        compressor=_blosc_lz4(),
        fill_value=_fill_value_for_dtype(dtype),
        order="C",
        overwrite=False,
        dimension_separator="/",
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "cell"]
    arr.attrs.update(attrs)


# =============================================================================
# pyramid aggregation
# =============================================================================

def _nanmean_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Quiet nanmean:
      - ignores NaNs
      - returns NaN if all children are NaN
    """
    blocks = np.asarray(blocks)
    mask = np.isfinite(blocks)
    count = mask.sum(axis=1)

    out = np.full(blocks.shape[0], np.nan, dtype=np.result_type(blocks.dtype, np.float64))
    good = count > 0
    if np.any(good):
        summed = np.where(mask, blocks, 0).sum(axis=1)
        out[good] = summed[good] / count[good]
    return out


def _resolve_pyramid_aggregator(agg_strategy) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    if callable(agg_strategy):
        name = getattr(agg_strategy, "__name__", "callable")

        def _wrapped(blocks):
            out = agg_strategy(blocks)
            out = np.asarray(out)
            if out.ndim != 1 or out.shape[0] != blocks.shape[0]:
                raise ValueError(
                    "Custom pyramid aggregator must return shape (n_out,) "
                    f"for input blocks of shape {blocks.shape}, got {out.shape}"
                )
            return out

        return _wrapped, name

    if agg_strategy == "mean":
        return lambda blocks: blocks.mean(axis=1), "mean"
    if agg_strategy == "sum":
        return lambda blocks: blocks.sum(axis=1), "sum"
    if agg_strategy == "min":
        return lambda blocks: blocks.min(axis=1), "min"
    if agg_strategy == "max":
        return lambda blocks: blocks.max(axis=1), "max"
    if agg_strategy == "nanmean":
        return _nanmean_blocks, "nanmean"
    if agg_strategy == "nansum":
        return lambda blocks: np.nansum(blocks, axis=1), "nansum"
    if agg_strategy == "nanmin":
        return lambda blocks: np.nanmin(blocks, axis=1), "nanmin"
    if agg_strategy == "nanmax":
        return lambda blocks: np.nanmax(blocks, axis=1), "nanmax"
    if agg_strategy == "first":
        return lambda blocks: blocks[:, 0], "first"

    raise ValueError(
        "Unsupported pyramid aggregation strategy. "
        "Use one of: 'mean', 'sum', 'min', 'max', 'nanmean', 'nansum', "
        "'nanmin', 'nanmax', 'first', or pass a callable."
    )


def degrade_healpix_nested_with_aggregation(
    values,
    *,
    zoom_in: int,
    zoom_out: int,
    agg_strategy="mean",
):
    values = np.asarray(values)

    if zoom_out > zoom_in:
        raise ValueError("zoom_out must be <= zoom_in")

    if zoom_out == zoom_in:
        return values.copy()

    npix_in = 12 * (4**zoom_in)
    if values.size != npix_in:
        raise ValueError(
            f"Expected {npix_in} values for zoom {zoom_in}, got {values.size}"
        )

    agg, _ = _resolve_pyramid_aggregator(agg_strategy)

    out = values
    z = zoom_in
    while z > zoom_out:
        if out.size % 4 != 0:
            raise ValueError(f"Cannot degrade zoom {z} values of length {out.size} by factor 4")
        blocks = out.reshape(out.size // 4, 4)
        out = agg(blocks)
        z -= 1

    return np.asarray(out)


# =============================================================================
# native prepare convenience
# =============================================================================

def prepare_native_regridder_for_zoom(
    *,
    zoom: int,
    gridfile: str | Path,
    cache_root: str | Path,
    rnh_module,
    nest: bool = True,
    use_memmap: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    rebuild_source_cache: bool = False,
    rebuild_target_cache: bool = False,
    rebuild_weights: bool = False,
):
    g_native = ux.open_grid(gridfile, chunks=-1)

    prepared = rnh_module.prepare_healpix_regridder(
        zoom=zoom,
        src_grid=g_native,
        cache_root=cache_root,
        nest=nest,
        show_progress=show_progress,
        verbose=verbose,
        use_memmap=use_memmap,
        rebuild_source_cache=rebuild_source_cache,
        rebuild_target_cache=rebuild_target_cache,
        rebuild_weights=rebuild_weights,
    )
    return prepared


# =============================================================================
# single-variable pyramid writer
# =============================================================================

def remap_variable_to_zoom_pyramid_stores(
    *,
    source_ds: xr.Dataset,
    prepared_max_zoom,
    rnh_module,
    store_paths: dict[int, str],
    var_name: str,
    max_zoom: int,
    min_zoom: int = 0,
    dataset_id: str | None = None,
    frequency_tag: str | None = None,
    stat_tag: str | None = None,
    cell_chunk: int = 65536,
    pyramid_agg="mean",
    resume: bool = True,
    overwrite: bool = False,
    check_conservation_first_step: bool = False,
    consolidate_at_end: bool = True,
) -> None:
    if not bool(prepared_max_zoom.nest):
        raise ValueError(
            "Pyramid construction by child aggregation requires nested HEALPix ordering "
            "(nest=True)."
        )

    if var_name not in source_ds:
        raise KeyError(f"{var_name!r} not found in source dataset")

    da_src = source_ds[var_name]
    if "time" not in da_src.dims:
        raise ValueError(f"{var_name!r} must have a 'time' dimension")

    other_dims = [d for d in da_src.dims if d != "time"]
    if len(other_dims) != 1:
        raise ValueError(
            f"{var_name!r} currently only supported for 2D variables (time, ncells). "
            f"Got dims={da_src.dims}"
        )

    ntime = int(da_src.sizes["time"])
    out_dtype = np.dtype(da_src.dtype)

    agg_fn, agg_name = _resolve_pyramid_aggregator(pyramid_agg)
    zooms = list(range(int(max_zoom), int(min_zoom) - 1, -1))

    for zoom in zooms:
        store_path = store_paths[zoom]
        ncell = int(12 * (4**zoom))

        _ensure_store_base(
            store_path=store_path,
            source_ds=source_ds,
            zoom=zoom,
            nest=True,
            dataset_id=dataset_id,
            frequency_tag=frequency_tag,
            stat_tag=stat_tag,
        )

        _ensure_var_array_exists(
            store_path=store_path,
            var_name=var_name,
            ntime=ntime,
            ncell=ncell,
            dtype=out_dtype,
            attrs=_normalize_var_attrs(
                da_src,
                prepared=prepared_max_zoom,
                zoom=zoom,
                nest=True,
                pyramid_source_zoom=max_zoom,
                pyramid_aggregation=agg_name,
            ),
            cell_chunk=cell_chunk,
        )

    max_store_path = store_paths[max_zoom]

    if overwrite:
        for zoom in zooms:
            root = zarr.open_group(store_paths[zoom], mode="a")
            if var_name in root.array_keys():
                del root[var_name]
            _ensure_var_array_exists(
                store_path=store_paths[zoom],
                var_name=var_name,
                ntime=ntime,
                ncell=int(12 * (4**zoom)),
                dtype=out_dtype,
                attrs=_normalize_var_attrs(
                    da_src,
                    prepared=prepared_max_zoom,
                    zoom=zoom,
                    nest=True,
                    pyramid_source_zoom=max_zoom,
                    pyramid_aggregation=agg_name,
                ),
                cell_chunk=cell_chunk,
            )

        p = _pyramid_progress_path(max_store_path, var_name, min_zoom)
        if p.exists():
            p.unlink()

    progress = _load_pyramid_progress(max_store_path, var_name, min_zoom)
    start_idx = int(progress["next_time_index"]) if resume else 0

    if progress.get("completed", False) and not overwrite:
        print(f"{Path(max_store_path).name} :: {var_name}: already completed")
        return

    max_ncell = int(12 * (4**max_zoom))
    print(
        f"{Path(max_store_path).name} :: {var_name}: start at timestep "
        f"{start_idx}/{ntime}"
    )
    print(
        f"{Path(max_store_path).name} :: {var_name}: output dtype = {out_dtype}, "
        f"chunks=(1, {min(cell_chunk, max_ncell)}), pyramid={max_zoom}->{min_zoom}, agg={agg_name}"
    )

    for i in range(start_idx, ntime):
        da_i = da_src.isel(time=i).squeeze(drop=True).compute()

        res_hi = rnh_module.apply_prepared_healpix_regridder(
            prepared_max_zoom,
            da_i,
            output_name=var_name,
            return_uxda=False,
            check_conservation=(check_conservation_first_step and i == start_idx),
            verbose=(i == start_idx),
        )

        vals_by_zoom = {max_zoom: np.asarray(res_hi["values"])}

        for zoom_hi in range(max_zoom, min_zoom, -1):
            zoom_lo = zoom_hi - 1
            blocks = vals_by_zoom[zoom_hi].reshape(-1, 4)
            vals_by_zoom[zoom_lo] = np.asarray(agg_fn(blocks))

        for zoom in zooms:
            vals_i = vals_by_zoom[zoom]

            ds_i = xr.Dataset(
                {
                    var_name: xr.DataArray(
                        vals_i[None, :],
                        dims=("time", "cell"),
                    )
                }
            )

            ds_i.to_zarr(
                store_paths[zoom],
                mode="r+",
                region={"time": slice(i, i + 1)},
                zarr_format=2,
                consolidated=False,
            )

        _save_pyramid_progress(
            max_store_path,
            var_name,
            min_zoom,
            next_time_index=i + 1,
            completed=(i + 1 == ntime),
        )

        print(
            f"{Path(max_store_path).name} :: {var_name}: wrote {i+1}/{ntime} "
            f"to zooms {max_zoom}->{min_zoom}",
            flush=True,
        )

    if consolidate_at_end:
        for zoom in zooms:
            zarr.consolidate_metadata(store_paths[zoom])

    print(f"{Path(max_store_path).name} :: {var_name}: done")


# =============================================================================
# orchestration across zoom levels
# =============================================================================

def remap_variables_to_zoom_pyramid_stores(
    *,
    source_ds: xr.Dataset,
    variables: Iterable[str],
    max_zoom: int,
    min_zoom: int = 0,
    dataset_id: str,
    frequency_tag: str,
    stat_tag: str,
    out_root: str | Path,
    cache_root: str | Path,
    gridfile: str | Path,
    rnh_module,
    cell_chunk: int = 65536,
    pyramid_agg="mean",
    nest: bool = True,
    use_memmap: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    rebuild_source_cache: bool = False,
    rebuild_target_cache: bool = False,
    rebuild_weights: bool = False,
    rebuild_stores: bool = False,
    resume: bool = True,
    overwrite_variables: bool = False,
    check_conservation_first_step: bool = False,
    consolidate_at_end: bool = True,
):
    if not nest:
        raise ValueError(
            "Pyramid construction by child aggregation requires nest=True."
        )

    zooms = list(range(int(max_zoom), int(min_zoom) - 1, -1))
    store_paths = {
        zoom: make_zoom_store_path(
            out_root=out_root,
            dataset_id=dataset_id,
            frequency_tag=frequency_tag,
            stat_tag=stat_tag,
            zoom=zoom,
        )
        for zoom in zooms
    }

    if rebuild_stores:
        for zoom in zooms:
            p = Path(store_paths[zoom])
            if p.exists():
                shutil.rmtree(p)

    print(f"\n=== PREPARE MAX ZOOM {max_zoom} ===")
    prepared_max = prepare_native_regridder_for_zoom(
        zoom=int(max_zoom),
        gridfile=gridfile,
        cache_root=cache_root,
        rnh_module=rnh_module,
        nest=nest,
        use_memmap=use_memmap,
        show_progress=show_progress,
        verbose=verbose,
        rebuild_source_cache=rebuild_source_cache,
        rebuild_target_cache=rebuild_target_cache,
        rebuild_weights=rebuild_weights,
    )

    for var_name in variables:
        print(f"\n=== REMAP {var_name} TO PYRAMID {max_zoom}->{min_zoom} ===")
        remap_variable_to_zoom_pyramid_stores(
            source_ds=source_ds,
            prepared_max_zoom=prepared_max,
            rnh_module=rnh_module,
            store_paths=store_paths,
            var_name=var_name,
            max_zoom=int(max_zoom),
            min_zoom=int(min_zoom),
            dataset_id=dataset_id,
            frequency_tag=frequency_tag,
            stat_tag=stat_tag,
            cell_chunk=cell_chunk,
            pyramid_agg=pyramid_agg,
            resume=resume,
            overwrite=overwrite_variables,
            check_conservation_first_step=check_conservation_first_step,
            consolidate_at_end=consolidate_at_end,
        )

    return prepared_max, store_paths


def remap_variables_to_all_zoom_stores(
    *,
    source_ds: xr.Dataset,
    variables: Iterable[str],
    zooms: Iterable[int],
    dataset_id: str,
    frequency_tag: str,
    stat_tag: str,
    out_root: str | Path,
    cache_root: str | Path,
    gridfile: str | Path,
    rnh_module,
    cell_chunk: int = 65536,
    pyramid_agg="mean",
    min_zoom: int = 0,
    nest: bool = True,
    use_memmap: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    rebuild_source_cache: bool = False,
    rebuild_target_cache: bool = False,
    rebuild_weights: bool = False,
    rebuild_stores: bool = False,
    resume: bool = True,
    overwrite_variables: bool = False,
    check_conservation_first_step: bool = False,
    consolidate_at_end: bool = True,
):
    zooms = list(zooms)
    if len(zooms) == 0:
        raise ValueError("zooms must not be empty")

    max_zoom = max(int(z) for z in zooms)

    return remap_variables_to_zoom_pyramid_stores(
        source_ds=source_ds,
        variables=variables,
        max_zoom=max_zoom,
        min_zoom=int(min_zoom),
        dataset_id=dataset_id,
        frequency_tag=frequency_tag,
        stat_tag=stat_tag,
        out_root=out_root,
        cache_root=cache_root,
        gridfile=gridfile,
        rnh_module=rnh_module,
        cell_chunk=cell_chunk,
        pyramid_agg=pyramid_agg,
        nest=nest,
        use_memmap=use_memmap,
        show_progress=show_progress,
        verbose=verbose,
        rebuild_source_cache=rebuild_source_cache,
        rebuild_target_cache=rebuild_target_cache,
        rebuild_weights=rebuild_weights,
        rebuild_stores=rebuild_stores,
        resume=resume,
        overwrite_variables=overwrite_variables,
        check_conservation_first_step=check_conservation_first_step,
        consolidate_at_end=consolidate_at_end,
    )