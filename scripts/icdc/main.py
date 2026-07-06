from argparse import ArgumentParser
import logging
from grid_doctor import setup_logging


_CONFS = {
    "imerg": ("imerg", "IMERGConfig"),
    "modis-atm-aqua": ("modis_atm_aqua", "MODISAquaAtmConfig"),
}


def loadConfig(name: str):
    from importlib import import_module

    mod_name, cls_name = _CONFS[name]
    mod = import_module(mod_name)
    return getattr(mod, cls_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_logging(logging.DEBUG)

    parser = ArgumentParser(
        description="Tool for remapping datasets to healpix in zarr format"
    )

    parser.add_argument(
        "config",
        choices=_CONFS.keys(),
        help="Dataset configuration to use",
    )

    parser.add_argument(
        "action",
        help="Specifies what step to do",
        choices=["init", "write"],
    )

    parser.add_argument(
        "destination",
        help=f"Parent path/location where to write the dataset",
        type=str,
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

    Config = loadConfig(args.config)

    config = Config(store_path=args.destination)

    if args.action == "init":
        config.init(overwrite=args.overwrite)
    elif args.action == "write":
        config.write(batch_size=args.batch_size)
