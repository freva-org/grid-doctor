#! python3

import logging
from argparse import ArgumentParser
from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import xarray as xr

from config import (
    get_source_cfg,
    load_config,
    load_conversion_rules,
    override_source_kind,
    resolve_requested_rules,
    resolve_source_files,
)
from grid_doctor import (
    cached_open_dataset,
    latlon_to_healpix_pyramid,
    save_pyramid_to_store,
)
from grid_doctor.utils import cache_dir
from mapping import transform_rule_dataset

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def expand_runtime_path(value: str) -> str:
    """Substitute runtime cache placeholders in a path string.

    Parameters
    ----------
    value : str
        Path template that may contain ``{scratch}`` or ``{cache_dir}``.

    Returns
    -------
    str
        Expanded path with any trailing slash removed.
    """
    scratch_path = str(cache_dir())
    return (
        value.replace("{scratch}", scratch_path)
        .replace("{cache_dir}", scratch_path)
        .rstrip("/")
    )


def get_test_target_path(config: Dict[str, Any]) -> str | None:
    """Return the optional preview output path.

    Parameters
    ----------
    config : dict[str, Any]
        Conversion configuration.

    Returns
    -------
    str or None
        Expanded preview path when configured.
    """
    test_destination = config.get("test_destination")
    if test_destination is None:
        return None
    if isinstance(test_destination, str):
        return expand_runtime_path(test_destination)
    if not isinstance(test_destination, dict):
        raise TypeError("'test_destination' must be a string or JSON object.")
    path = test_destination.get("path")
    if not isinstance(path, str):
        raise TypeError("'test_destination.path' must be a string.")
    return expand_runtime_path(path)


def normalise_open_kwargs(
    source_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare kwargs for xarray open calls.

    Parameters
    ----------
    source_cfg : dict[str, Any]
        Source configuration.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Keyword arguments for ``xr.open_dataset`` and cached multi-file
        opening respectively.
    """
    dataset_kwargs = dict(source_cfg.get("dataset_kwargs", {}))
    multi_file_kwargs = dict(source_cfg.get("multi_file_kwargs", {}))

    engine = source_cfg.get("engine")
    if engine is not None:
        dataset_kwargs.setdefault("engine", engine)
        multi_file_kwargs.setdefault("engine", engine)

    backend_kwargs = dict(source_cfg.get("backend_kwargs", {}))
    if backend_kwargs:
        dataset_kwargs["backend_kwargs"] = backend_kwargs
        multi_file_kwargs["backend_kwargs"] = backend_kwargs

    chunks = source_cfg.get("chunks")
    if chunks is not None:
        dataset_kwargs.setdefault("chunks", chunks)
        multi_file_kwargs.setdefault("chunks", chunks)

    multi_file_kwargs.setdefault("combine", "by_coords")
    multi_file_kwargs.setdefault("parallel", False)
    return dataset_kwargs, multi_file_kwargs


def open_kerchunk_dataset(
    reference: str,
    *,
    source_cfg: Dict[str, Any],
) -> xr.Dataset:
    """Open a dataset from a kerchunk reference.

    Parameters
    ----------
    reference : str
        Kerchunk reference path or URL.
    source_cfg : dict[str, Any]
        Source configuration containing kerchunk open options.

    Returns
    -------
    xarray.Dataset
        Dataset opened through the kerchunk backend.
    """
    try:
        import kerchunk  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Kerchunk input was requested but the 'kerchunk' package is not installed."
        ) from exc

    dataset_options = dict(source_cfg.get("kerchunk_open_dataset_options", {}))
    storage_options = dict(source_cfg.get("kerchunk_storage_options", {}))
    return xr.open_dataset(
        reference,
        engine="kerchunk",
        storage_options=storage_options,
        **dataset_options,
    )


def open_rule_dataset(config: Dict[str, Any], rule: Dict[str, Any]) -> xr.Dataset:
    """Open the source dataset for a single conversion rule.

    Parameters
    ----------
    config : dict[str, Any]
        Conversion configuration.
    rule : dict[str, Any]
        Conversion rule.

    Returns
    -------
    xarray.Dataset
        Source dataset for the rule.
    """
    source_cfg = get_source_cfg(config)
    files = resolve_source_files(config, rule)
    if source_cfg.get("kind", "grib") == "kerchunk" and files[0].endswith(".parquet"):
        return open_kerchunk_dataset(files[0], source_cfg=source_cfg)

    dataset_kwargs, multi_file_kwargs = normalise_open_kwargs(source_cfg)
    if len(files) == 1:
        return xr.open_dataset(files[0], **dataset_kwargs)
    return cached_open_dataset(files, **multi_file_kwargs)


def open_source_dataset(config: Dict[str, Any]) -> xr.Dataset:
    """Open, convert, and merge all selected source variables.

    Parameters
    ----------
    config : dict[str, Any]
        Conversion configuration.

    Returns
    -------
    xarray.Dataset
        Merged dataset containing all requested converted variables.
    """
    source_cfg = config["source"]
    conversion_rules = load_conversion_rules(config)
    selected_rules = resolve_requested_rules(config, conversion_rules)
    datasets = [
        transform_rule_dataset(
            open_rule_dataset(config, rule),
            rule=rule,
            source_cfg=source_cfg,
        )
        for rule in selected_rules
    ]
    if not datasets:
        raise ValueError("No variables were selected for conversion.")
    return xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")


def convert(config: Dict[str, Any], *, init: bool, region: Dict[str, slice]) -> None:
    """Convert configured source data to HEALPix and write it out.

    Parameters
    ----------
    config : dict[str, Any]
        Conversion configuration.
    init : bool
        Whether to initialise the output store structure without writing
        data chunks.
    region : dict[str, slice]
        Region selection to write, keyed by dimension name.
    """
    import zarr

    destination = config["destination"]
    zarr.config.set(default_zarr_format=int(destination.get("zarr_format", 2)))
    ds = open_source_dataset(config)

    if "time" in ds.coords and region["time"].start > ds.time.size:
        logging.warning(
            "Region (%s) not overlapping with dataset (%s), skipping!",
            region,
            {"time": slice(0, ds.time.size)},
        )
        return

    time_chunk = int(config.get("time_chunk", 48))
    if "time" in ds.dims:
        ds = ds.chunk({"time": time_chunk})

    logging.info("Converting dataset to HEALPix")
    ds_hp = latlon_to_healpix_pyramid(
        ds,
        source_units=str(config.get("source_units", "auto")),
        method=str(config.get("method", "conservative")),
    )

    target_path = expand_runtime_path(str(destination["path"]))
    test_target_path = get_test_target_path(config)
    storage_options = {
        "endpoint_url": destination["endpoint_url"],
        "key": destination["access_key"],
        "secret": destination["secret_key"]
    }
        
    zarr_format = int(destination.get("zarr_format", 2))

    write_targets: List[Tuple[str, str]] = []
    if test_target_path:
        write_targets.append(("preview", test_target_path))
    write_targets.append(("destination", target_path))

    if init:
        for target_name, path in write_targets:
            logging.info("Initializing %s store %s", target_name, path)
            save_pyramid_to_store(
                ds_hp,
                path,
                storage_options=storage_options if path.startswith("s3://") else None,
                mode="w",
                compute=False,
                zarr_format=zarr_format,
            )
        return

    bounded_region = region
    if "time" in ds.dims:
        bounded_region = {
            key: slice(value.start, min(value.stop, ds.sizes[key]), value.step)
            for key, value in region.items()
        }

    for target_name, path in write_targets:
        logging.info(
            "Writing %s store on region %s to %s",
            target_name,
            bounded_region,
            path,
        )
        save_pyramid_to_store(
            ds_hp,
            path,
            storage_options=storage_options if path.startswith("s3://") else None,
            mode="r+",
            region=bounded_region,
            zarr_format=zarr_format,
        )


def main() -> None:
    """Parse CLI arguments and prepare a conversion run."""
    start_idx = int(getenv("SLURM_ARRAY_TASK_ID", 0))
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--slice-size", default=48, type=int)
    parser.add_argument("--start", default=start_idx, type=int)
    parser.add_argument("--debug", action="store_true")
    source_kind = parser.add_mutually_exclusive_group()
    source_kind.add_argument("--use-grib", action="store_true")
    source_kind.add_argument("--use-kerchunk", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.slice_size % 48:
        raise SystemExit("slice-size must be a multiple of 48 (time chunk)")

    region = {
        "time": slice(args.start * args.slice_size, (args.start + 1) * args.slice_size)
    }
    config = load_config(args.config)
    if args.use_grib:
        config = override_source_kind(config, "grib")
    elif args.use_kerchunk:
        config = override_source_kind(config, "kerchunk")
    logging.info("init=%s region=%s config=%s", args.init, region, args.config)
    convert(config, init=args.init, region=region)


if __name__ == "__main__":
    main()
