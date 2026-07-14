import argparse
import grid_doctor as gd
import logging
import warnings

logging.basicConfig(level=logging.DEBUG)

_NAME_TEMPLATE = "level_{zoom}.zarr"


def parse_arguments():
    parser = argparse.ArgumentParser(
        "A command line tool to coarsen a HealPIX dataset in Zarr format"
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to dataset to be coarsened. (e.g. file:///path/to/dataset.zarr, https://webserver.example.com/path/to/dataset.zarr)",
    )
    parser.add_argument(
        "destination",
        help="Base path where to write coarsened versions of the input dataset.",
    )

    parser.add_argument(
        "-n",
        "--name-template",
        nargs="?",
        type=str,
        default=_NAME_TEMPLATE,
        help=f"Python format string containing `zoom` to indicate the name of each coarsend dataset (default:{_NAME_TEMPLATE})",
    )
    parser.add_argument(
        "-l",
        "--levels",
        nargs="+",
        type=int,
        help="Levels (whitespace-separated) to which coarsen, must be lower than inputs'(N). Default coarsens from level 0 to N-1.",
    )
    return parser.parse_args()


def open_healpix_zarr(path: str):
    import xarray as xr

    ds = xr.open_zarr(path)
    return ds


def get_region(size):
    import os

    n_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    t_p_job = size // n_jobs + 1
    t_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - int(
        os.environ.get("SLURM_ARRAY_TASK_MIN", 0)
    )
    if t_id * t_p_job > size:
        logging.info("Nothing to do, region outsize dataset")
        return None
    return {"time": slice(t_id * t_p_job, min((t_id + 1) * t_p_job, size))}


def run(args: argparse.Namespace):

    ds = open_healpix_zarr(args.source)
    if (region := get_region(ds.time.size)) is None:
        return

    max_level = ds.attrs.get(
        "healpix_level", int("".join(filter(str.isdigit, args.source)))
    )

    levels = sorted(set(args.levels or [*range(max_level)]))[-1::-1]
    for level in levels:
        if level >= max_level:
            warnings.warn(f"Skipping level {level}")
            continue

        logging.info("Coarsening level to %d", level)
        c_ds = gd.coarsen_healpix(ds, level)

        # Initialize full dataset
        store_path = args.destination + args.name_template.format(zoom=level)

        try:
            gd.utils.init_full_zarr_store(c_ds, store_path)
            logging.info("Initialization complete!")
        except FileExistsError:
            z_ds = open_healpix_zarr(store_path)
            if not c_ds.drop_vars(c_ds.data_vars).identical(
                z_ds.drop_vars(z_ds.data_vars)
            ):
                logging.critical("Found non identical dataset at %s!", store_path)
            logging.info("Found pre-initialized dataset at %s", store_path)

        logging.info(
            "Dataset initialized (%s). Now writing slices from region = %s",
            store_path,
            region,
        )

        c_ds.drop_vars(c_ds.coords).isel(region).to_zarr(
            store_path,
            zarr_format=2,
            mode="r+",
            region=region,
        )


def main():
    args = parse_arguments()
    print(args)
    run(args)
    pass


if __name__ == "__main__":
    main()
