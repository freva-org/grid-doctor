import xarray as xr
import logging

from typing import Iterable, Mapping
from os import getenv

from grid_doctor import (
    cached_open_dataset,
    save_pyramid_to_s3,
)

Pyramid = Mapping[int, xr.Dataset]

class BaseStructure():
    _open_kargs = {}

    def __init__(self, structure):
        self._structure = structure
        self._open_kwargs = self._open_kwargs

    def items(self):
        for k, v in self._structure.items():
            try:
                ds = cached_open_dataset(v, **self._open_kwargs)
            except Exception as e:
                print(f"ERROR unable to open {k}")
                print(f"{e}")
                continue
            yield k, ds
    
    def __iter__(self) -> Iterable[str, xr.Dataset]:
        for k, v in self._structure.items():
            yield k, v

    def write(self, pyramids:Mapping[str, Pyramid], init:bool = False, region: Mapping[str,slice] = {'time':slice(0,1)}):
        import zarr
        zarr.config.set(default_zarr_format=2)
        opts = {
            "endpoint_url": "https://s3.eu-dkrz-3.dkrz.cloud",
            "key": getenv("S3_KEY"),
            "secret": getenv("S3_SECRET"),
        }
        for dst_url, pyramid in pyramids.items():
            if init:
                logging.info("Writting ONLY metadata to %s", str(region), dst_url)
                save_pyramid_to_s3(pyramid, dst_url, mode="w", compute=False, s3_options=opts)
            else:
                logging.info("Writting region: %s to %s", str(region), dst_url)
                save_pyramid_to_s3(pyramid, dst_url, mode="r+", region=region, s3_options=opts)

