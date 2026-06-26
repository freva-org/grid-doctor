from argparse import ArgumentParser
import logging
from grid_doctor import setup_logging
from imerg import IMERGConfig as config

if __name__ == "__main__":
    # logger.setLevel(logging.INFO)
    setup_logging(logging.DEBUG)

    parser = ArgumentParser(
        "imerg", description="Tool for remaping IMERG to healpix in zarr format"
    )

    parser.add_argument(
        "destination",
        help=f"Base location where to write the dataset (default: file//{config.store_path})",
        type=str,
        nargs="?",
        default=config.store_path,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config(store_path=args.destination).init(overwrite=args.overwrite)
