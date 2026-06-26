from argparse import ArgumentParser
import logging
from grid_doctor import setup_logging
from imerg import IMERGConfig as Config

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_logging(logging.DEBUG)

    parser = ArgumentParser(
        "imerg", description="Tool for remaping IMERG to healpix in zarr format"
    )

    parser.add_argument(
        "action",
        help="Specifies what step to do",
        choices=["init", "write"],
    )

    parser.add_argument(
        "destination",
        help=f"Base location where to write the dataset (default: file//{Config.store_path})",
        type=str,
        nargs="?",
        default=Config.store_path,
    )
    ## Init arguments
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Used to allow overwritting a dataset (only valid with ``init`` action)",
    )

    ## Write arguments
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Used to adjust which time index to start writing from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Used to adjust the width of each regio when writing",
    )
    args = parser.parse_args()

    config = Config(store_path=args.destination)

    if args.action == "init":
        config.init(overwrite=args.overwrite)
    elif args.action == "write":
        config.write(batch_size=args.batch_size)
