"""Run the conversion on a single GH node."""

import logging
from getpass import getuser
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from icon_dream_reflow_helpers.common import DEFAULT_SOURCE_ROOT, target_root

import grid_doctor as gd
import grid_doctor.cli as gd_cli

if TYPE_CHECKING:
    import argparse

SCRATCH_DIR = "/scratch/{u[0]}/{u}/grid-doctor/icon-dream/".format(u=getuser())


logger = logging.getLogger("icon-dream2healpix")


def download_files(
    s3_options: dict[str, str],
    variables: list[str] | None = None,
    frequency: str = "hourly",
    s3_bucket: str = "icon-dream",
    override: bool = False,
    run_dir: Path = Path(SCRATCH_DIR),
    cmor: bool = True,
) -> list[dict[str, str]]:
    """Download files."""

    from icon_dream_reflow_helpers.planning import (IconDreamSource,
                                                    load_existing_target_info,
                                                    parse_datetime,
                                                    resolve_variables)

    variables = resolve_variables(
        variables or ["all"], frequency, DEFAULT_SOURCE_ROOT
    )
    src = IconDreamSource(variables, frequency, ("2010-01-01T00:00", "now"))
    existing_max = None
    existing_variables: set[str] = set()
    if override is False:
        existing = load_existing_target_info(
            target_root(s3_bucket, frequency), s3_options
        )
        existing_max = (
            parse_datetime(existing["max_time"]) if existing["max_time"] else None
        )
        # The update-only check must compare against the variables that
        # are actually in the target store (previously the *requested*
        # variables were passed here, which defeated the check).
        existing_variables = set(existing["variables"])
    items = src.list_items(
        existing_max_time=existing_max,
        existing_variables=existing_variables,
        cmor=cmor,
    )
    for item in items:
        tmp_path = run_dir / "raw-input" / item["variable"] / item["filename"]
        if not tmp_path.is_file():
            logger.info("Downloading %s to %s", item["url"], tmp_path)
            gd_cli.download_file(item["url"], tmp_path.parent)
        else:
            logger.debug("Skipping existing file %s", tmp_path)
        item["raw-path"] = str(tmp_path)
    return items


def remap(
    items: list[dict[str, str]],
    s3_options: dict[str, str],
    run_dir: Path = Path(SCRATCH_DIR),
    frequency: str = "hourly",
    cmor: bool = True,
) -> dict[int, xr.Dataset]:
    """Remap the items."""

    from icon_dream_reflow_helpers.cmor import cmorize_dataset
    from icon_dream_reflow_helpers.common import (DEFAULT_GRID_URL,
                                                  chunk_for_target_store_size)
    from icon_dream_reflow_helpers.transform import \
        prepare_dataset_for_regridding

    variables: dict[str, list[Path]] = {}
    dsets: list[xr.Dataset] = []
    logger.debug("Opening grid")
    grid_ds = xr.open_dataset(
        gd_cli.download_file(DEFAULT_GRID_URL, run_dir / "shared")
    )
    logger.debug("Calculating max healpix level .... ")
    max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
    logger.debug("Calculating max healpix level .... %d", max_level)
    logger.debug("Creating weight matrix ...")
    weight_file = gd.cached_weights(
        grid_ds,
        level=max_level,
        prefer_offline=True,
        nproc=48,
        cache_path="/work/ks1387/healpix-weights",
    )
    logger.debug("Creating weight matrix ... %s ", weight_file)
    for item in items:
        variables.setdefault(item["variable"], [])
        variables[item["variable"]].append(Path(item["raw-path"]))

    for var, files in variables.items():
        logger.debug("Reading %d input files for variable %s", len(files), var)
        ds = xr.open_mfdataset(
            sorted(files),
            combine="nested",
            concat_dim="time",
            parallel=True,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            chunks="auto",
            join="override",  # only if non-time dims really match
            combine_attrs="override",
        )
        logger.debug("Preprocessing files ... ")
        ds = prepare_dataset_for_regridding(ds).drop_duplicates(
            dim="time", keep="first"
        )
        if cmor:
            ds = cmorize_dataset(ds, var, frequency=frequency)
        dsets.append(ds)
        logger.debug("Preprocessing files .... done")
    dset = xr.merge(dsets, join="outer").chunk({"cell": -1})
    chunk = chunk_for_target_store_size(level=max_level)
    logger.debug("Creating healpix pyramid ... ")
    pyramid = gd.create_healpix_pyramid(
        dset, max_level=max_level, weights_path=weight_file, backend="cupy"
    )
    logger.debug("Rechunking ...")
    for key in pyramid:
        pyramid[key] = pyramid[key].chunk(chunk)
    return pyramid


def upload_pyramid(
    pyramid: dict[int, xr.Dataset], s3_options: dict[str, str], s3_path: str
) -> None:
    """Upload the healpix pyramid."""

    logger.debug("Uploading healpix pyramid to %s", s3_path)
    gd.save_pyramid_to_s3(
        pyramid,
        s3_path,
        s3_options,
        mode="w",
        compute=True,
        zarr_format=2,
    )


def cli(argv: list[str] | None = None) -> "argparse.Namespace":
    """Setup the cli."""

    parser = gd_cli.get_parser("convert-icon-dream", description="Convert ICON-DREAM")
    parser.add_argument(
        "--variables",
        default=["all"],
        nargs="+",
        help="Variables to process ('all' expands to every variable "
        "available for the chosen frequency)",
    )
    parser.add_argument(
        "--no-cmor",
        action="store_true",
        help="Keep the original ICON variable names and units.",
    )
    parser.add_argument(
        "--freq", "-f", default="hourly", help="ICON-DREAM data frequency"
    )
    parser.add_argument(
        "--run-dir", type=Path, help="The run directory", default=SCRATCH_DIR
    )
    parser.add_argument(
        "--override", "-o", action="store_true", help="Override existing files."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = cli()
    gd_cli.setup_logging_from_args(args)
    s3_options = gd.get_s3_options(
        args.s3_endpoint, args.s3_credentials_file.expanduser()
    )
    logger.info("Working with s3_options: %s", s3_options)
    s3_path = target_root(f"s3://{args.s3_bucket}", args.freq)
    raw_files = download_files(
        s3_options,
        args.variables,
        args.freq,
        s3_bucket=args.s3_bucket,
        override=args.override,
        run_dir=args.run_dir,
        cmor=not args.no_cmor,
    )
    pyramid = remap(
        raw_files,
        s3_options,
        run_dir=args.run_dir,
        frequency=args.freq,
        cmor=not args.no_cmor,
    )

    upload_pyramid(pyramid, s3_options, s3_path)
