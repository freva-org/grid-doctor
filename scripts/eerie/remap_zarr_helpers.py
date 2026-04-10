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


def _to_json_attr_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_json_attr_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_json_attr_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_attr_value(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _source_dataset_attrs_json(source_ds: xr.Dataset) -> dict:
    return {str(k): _to_json_attr_value(v) for k, v in dict(source_ds.attrs).items()}


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

def make_multiscales_root_path(*, out_root: str | Path) -> str:
    out_root = Path(out_root)
    out_root_str = str(out_root)
    if out_root_str.endswith(".zarr"):
        return out_root_str
    return f"{out_root_str}.zarr"


def make_zoom_store_path(
    *,
    out_root: str | Path,
    dataset_id: str,
    frequency_tag: str,
    stat_tag: str,
    zoom: int,
) -> str:
    # dataset_id/frequency_tag/stat_tag kept in signature for API compatibility
    del dataset_id, frequency_tag, stat_tag
    return str(Path(make_multiscales_root_path(out_root=out_root)) / "multiscales" / f"zoom_{zoom}")


# =============================================================================
# multiscales / GeoZarr helpers
# =============================================================================

def _multiscales_convention_descriptor() -> list[dict]:
    return [
        {
            "uuid": "d35379db-88df-4056-af3a-620245f8e347",
            "name": "multiscales",
            "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
            "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
        }
    ]


def _build_multiscales_layout(
    *,
    max_zoom: int,
    min_zoom: int,
) -> list[dict]:
    layout = []
    previous_asset = None

    for zoom in range(int(max_zoom), int(min_zoom) - 1, -1):
        asset = f"zoom_{zoom}"
        entry = {"asset": asset}
        if previous_asset is not None:
            entry["derived_from"] = previous_asset
            entry["transform"] = {"scale": [2.0]}
        layout.append(entry)
        previous_asset = asset

    return layout


def _ensure_multiscales_root_group(
    *,
    root_store_path: str | Path,
    source_ds: xr.Dataset,
    max_zoom: int,
    min_zoom: int,
    pyramid_agg: str,
    dataset_id: str | None = None,
    frequency_tag: str | None = None,
    stat_tag: str | None = None,
    nest: bool = True,
) -> None:
    root_store_path = Path(root_store_path)
    source_attrs_json = _source_dataset_attrs_json(source_ds)

    root = zarr.open_group(root_store_path, mode="a")
    root_attrs = {
        "dataset_id": dataset_id,
        "frequency": frequency_tag,
        "statistic": stat_tag,
        "healpix_nest": bool(nest),
        "healpix_zoom_min": int(min_zoom),
        "healpix_zoom_max": int(max_zoom),
        "source_dataset_attrs": source_attrs_json,
    }
    root.attrs.update({k: v for k, v in root_attrs.items() if v is not None})

    multiscales_group_path = root_store_path / "multiscales"
    multiscales_group = zarr.open_group(multiscales_group_path, mode="a")

    multiscales_attrs = {
        "dataset_id": dataset_id,
        "frequency": frequency_tag,
        "statistic": stat_tag,
        "healpix_nest": bool(nest),
        "healpix_zoom_min": int(min_zoom),
        "healpix_zoom_max": int(max_zoom),
        "source_dataset_attrs": source_attrs_json,
        "zarr_conventions": _multiscales_convention_descriptor(),
        "multiscales": {
            "resampling_method": str(pyramid_agg),
            "layout": _build_multiscales_layout(max_zoom=max_zoom, min_zoom=min_zoom),
        },
    }
    multiscales_group.attrs.update({k: v for k, v in multiscales_attrs.items() if v is not None})


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


def _ensure_1d_coord_array_exists(
    *,
    store_path: str | Path,
    coord_name: str,
    values,
    attrs: dict | None = None,
) -> None:
    """
    Create a 1D coordinate array manually with zarr-python if it does not exist yet.
    Intended for passthrough singleton dims like height_2.
    """
    store_path = Path(store_path)
    root = zarr.open_group(store_path, mode="a")

    if coord_name in root:
        return

    values = np.asarray(values)
    arr = root.create_dataset(
        coord_name,
        shape=values.shape,
        chunks=values.shape,
        dtype=values.dtype,
        compressor=_blosc_lz4(),
        fill_value=None,
        order="C",
        overwrite=False,
        dimension_separator="/",
    )
    arr[:] = values
    arr.attrs["_ARRAY_DIMENSIONS"] = [coord_name]
    if attrs:
        arr.attrs.update(dict(attrs))


def _ensure_var_array_exists(
    *,
    store_path: str | Path,
    var_name: str,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    dtype,
    attrs: dict,
    cell_chunk: int = 65536,
) -> None:
    """
    Create an empty N-D variable manually with zarr-python.

    Expected use here:
      - (time, cell)
      - (time, singleton_dim, cell)
      - more singleton passthrough dims are okay
    """
    store_path = Path(store_path)
    root = zarr.open_group(store_path, mode="a")
    if var_name in root.array_keys():
        return

    dtype = np.dtype(dtype)
    shape = tuple(int(s) for s in shape)
    dims = tuple(dims)

    if len(shape) != len(dims):
        raise ValueError(f"shape/dims mismatch for {var_name!r}: shape={shape}, dims={dims}")

    chunks = list(shape)
    chunks[0] = 1
    chunks[-1] = min(int(cell_chunk), int(shape[-1]))

    arr = root.create_dataset(
        var_name,
        shape=shape,
        chunks=tuple(chunks),
        dtype=dtype,
        compressor=_blosc_lz4(),
        fill_value=_fill_value_for_dtype(dtype),
        order="C",
        overwrite=False,
        dimension_separator="/",
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    arr.attrs.update(attrs)


# =============================================================================
# dimension detection helpers
# =============================================================================

def _detect_spatial_and_passthrough_dims(da_src: xr.DataArray, src_n_face: int):
    """
    Detect the source spatial dimension from the prepared source-face count.

    Rules:
      - 'time' must exist
      - exactly one dimension must have size == src_n_face -> source face dim
      - all remaining non-time, non-spatial dims are passthrough dims
      - passthrough dims must currently be singleton (size == 1)

    Returns
    -------
    src_face_dim : str
    passthrough_dims : list[str]
    passthrough_sizes : dict[str, int]
    """
    if "time" not in da_src.dims:
        raise ValueError(f"{da_src.name!r} must have a 'time' dimension")

    candidate_face_dims = [
        d for d in da_src.dims
        if d != "time" and int(da_src.sizes[d]) == int(src_n_face)
    ]
    if len(candidate_face_dims) != 1:
        raise ValueError(
            f"{da_src.name!r}: could not uniquely identify source face dimension from "
            f"dims={da_src.dims} with sizes={dict(da_src.sizes)} and src_n_face={src_n_face}. "
            f"Expected exactly one non-time dim with size == src_n_face."
        )

    src_face_dim = candidate_face_dims[0]
    passthrough_dims = [d for d in da_src.dims if d not in ("time", src_face_dim)]
    passthrough_sizes = {d: int(da_src.sizes[d]) for d in passthrough_dims}

    bad = {d: n for d, n in passthrough_sizes.items() if n != 1}
    if bad:
        raise ValueError(
            f"{da_src.name!r}: only singleton passthrough dims are currently supported. "
            f"Found non-singleton extra dims: {bad}. "
            f"Supported examples: (time, height_2, ncells) with height_2=1."
        )

    return src_face_dim, passthrough_dims, passthrough_sizes


def _ensure_passthrough_coords_exist(
    *,
    store_path: str | Path,
    source_ds: xr.Dataset,
    passthrough_dims: list[str],
) -> None:
    """
    Ensure singleton passthrough coordinates exist in the output store.
    """
    for d in passthrough_dims:
        if d in source_ds.coords:
            coord_da = source_ds.coords[d]
            values = np.asarray(coord_da.values)
            attrs = dict(coord_da.attrs)
        else:
            values = np.arange(int(source_ds.sizes[d]), dtype=np.int64)
            attrs = {}

        _ensure_1d_coord_array_exists(
            store_path=store_path,
            coord_name=d,
            values=values,
            attrs=attrs,
        )


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
    consolidate_at_end: bool = False,
) -> None:
    if not bool(prepared_max_zoom.nest):
        raise ValueError(
            "Pyramid construction by child aggregation requires nested HEALPix ordering "
            "(nest=True)."
        )

    if not bool(prepared_max_zoom.nest):
        raise ValueError(
            "Pyramid construction by child aggregation requires nested HEALPix ordering "
            "(nest=True)."
        )

    if var_name not in source_ds:
        raise KeyError(f"{var_name!r} not found in source dataset")

    da_src = source_ds[var_name]
    src_face_dim, passthrough_dims, passthrough_sizes = _detect_spatial_and_passthrough_dims(
        da_src, prepared_max_zoom.src_n_face
    )

    ntime = int(da_src.sizes["time"])
    out_dtype = np.dtype(da_src.dtype)

    agg_fn, agg_name = _resolve_pyramid_aggregator(pyramid_agg)
    zooms = list(range(int(max_zoom), int(min_zoom) - 1, -1))

    out_dims = ("time", *passthrough_dims, "cell")

    for zoom in zooms:
        store_path = store_paths[zoom]
        ncell = int(12 * (4**zoom))
        out_shape = (ntime, *[passthrough_sizes[d] for d in passthrough_dims], ncell)

        _ensure_store_base(
            store_path=store_path,
            source_ds=source_ds,
            zoom=zoom,
            nest=True,
            dataset_id=dataset_id,
            frequency_tag=frequency_tag,
            stat_tag=stat_tag,
        )

        _ensure_passthrough_coords_exist(
            store_path=store_path,
            source_ds=source_ds,
            passthrough_dims=passthrough_dims,
        )

        _ensure_var_array_exists(
            store_path=store_path,
            var_name=var_name,
            shape=out_shape,
            dims=out_dims,
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

            ncell = int(12 * (4**zoom))
            out_shape = (ntime, *[passthrough_sizes[d] for d in passthrough_dims], ncell)

            _ensure_var_array_exists(
                store_path=store_paths[zoom],
                var_name=var_name,
                shape=out_shape,
                dims=out_dims,
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
        f"{Path(max_store_path).name} :: {var_name}: source dims={da_src.dims}, "
        f"src_face_dim={src_face_dim}, passthrough_dims={passthrough_dims}"
    )
    print(
        f"{Path(max_store_path).name} :: {var_name}: output dtype = {out_dtype}, "
        f"chunks=(1, ..., {min(cell_chunk, max_ncell)}), pyramid={max_zoom}->{min_zoom}, agg={agg_name}"
    )

    squeeze_indexers = {d: 0 for d in passthrough_dims}

    pbar = tqdm(
        total=ntime,
        desc=f"{Path(max_store_path).name} :: {var_name}",
        initial=start_idx,
        dynamic_ncols=True,
        leave=True,
    )

    for i in range(start_idx, ntime):
        da_i = da_src.isel(time=i, **squeeze_indexers).squeeze(drop=True).compute()

        if da_i.dims != (src_face_dim,):
            raise ValueError(
                f"{var_name!r}: expected 1D field after squeezing singleton passthrough dims, "
                f"got dims={da_i.dims}, shape={da_i.shape}"
            )

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
            arr_shape = (1, *[passthrough_sizes[d] for d in passthrough_dims], vals_i.size)
            vals_out = vals_i.reshape(arr_shape)

            ds_i = xr.Dataset(
                {
                    var_name: xr.DataArray(
                        vals_out,
                        dims=out_dims,
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

        pbar.update(1)
        pbar.set_postfix_str(f"time={i+1}/{ntime}")

    pbar.close()

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

    multiscales_root_path = make_multiscales_root_path(out_root=out_root)

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
        p = Path(multiscales_root_path)
        if p.exists():
            shutil.rmtree(p)

    _ensure_multiscales_root_group(
        root_store_path=multiscales_root_path,
        source_ds=source_ds,
        max_zoom=int(max_zoom),
        min_zoom=int(min_zoom),
        pyramid_agg=str(pyramid_agg),
        dataset_id=dataset_id,
        frequency_tag=frequency_tag,
        stat_tag=stat_tag,
        nest=nest,
    )

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
            consolidate_at_end=False,
        )

    if consolidate_at_end:
        zarr.consolidate_metadata(multiscales_root_path)

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