#!python3

from glob import glob
from itertools import chain
import sys
from pathlib import Path
import argparse

try:
    from rich_argparse import ArgumentDefaultsRichHelpFormatter as ArgFormatter
except ImportError:
    ArgFormatter = argparse.ArgumentDefaultsHelpFormatter
    
class ICDCCollectionArgumentParser(argparse.ArgumentParser):
        """ArgumentParser with a consistent help formatter."""
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("formatter_class", ArgFormatter)
            super().__init__(*args, **kwargs)

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(
            prog='ICDC',
            description='ICDC healpix converter',
        )
    subp = parser.add_subparsers(
            dest = "_collection",
            required = True,
            parser_class=ICDCCollectionArgumentParser,
        )
    
    from icdc import collections
    for c in collections:
        name = c.__name__.lower()
        _col_p = subp.add_parser(name, description = f'Parser for {name} collection')
        c.configure_parser(_col_p)

    args = parser.parse_args()

if __name__ == "__main__":
    main()
