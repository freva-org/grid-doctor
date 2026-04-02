import xarray as xr
import logging
import zarr

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import  Any, Callable, Dict, Iterable, Mapping, Optional, Literal
from os import getenv

from grid_doctor import (
    cached_open_dataset,
    latlon_to_healpix_pyramid,
    save_pyramid_to_s3,
)

Region = dict[str, slice]
Pyramid = Mapping[int, xr.Dataset]
TransformCallable = Callable[[Any], Pyramid]
_ChunkingArg =  str | int | Literal['auto'] | tuple[int, ...] | None
Chunking = _ChunkingArg | Mapping[Any,  _ChunkingArg]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Collection(Enum):
    pass
    
    @classmethod
    def run_pipelines(cls) -> None:
        for collection in cls:
            name, config = collection.name, collection.value
            pipeline = Pipeline(config)
            logger.info("Running %s pipeline", name)
            pipeline.run()
            

@dataclass
class Config:
    dst_s3url: str
    paths: str
    engine: str = "netcdf4"
    parallel: bool = False
    open_kwargs: dict[str, Any] = field(default_factory=dict)
    chunking: Chunking = field(default_factory=dict)
    regrid_function: Callable[[Any], Pyramid] = latlon_to_healpix_pyramid
    init: bool = False
    region: Optional[Region] = None
    zarr_format: int = 2


class Source(ABC):
    def __init__(self, spec: Config):
        self.spec = spec

    def load(self) -> xr.Dataset:
        from glob import glob
        try:
            ds:xr.Dataset = cached_open_dataset(
                glob(self.spec.paths), engine=self.spec.engine, parallel=self.spec.parallel, **self.spec.open_kwargs
            )
        except Exception as e:
            print(f"ERROR unable to open source dataset specified with {self.spec}")
            print(f"{e}")
            raise e
        return ds


class Transform(ABC):
    def __init__(self, function: TransformCallable, chunking: Chunking = None):
        self.function = function
        self.chunking = chunking

    def convert(self, ds: xr.Dataset) -> Pyramid:
        print(ds.chunk(self.chunking))
        print(self.function)
        return self.function(ds.chunk(self.chunking))


class ZarrS3Sink:
    def __init__(self, base_url: str, init: bool, region: Optional[Region] = None):
        self.base_url = base_url
        self.s3_options = {
            "endpoint_url": "https://s3.eu-dkrz-3.dkrz.cloud",
            "key": getenv("S3_KEY"),
            "secret": getenv("S3_SECRET"),
        }
        self.init = (init,)
        self.region = region

    def write(
        self,
        pyramid: Pyramid,
    ) -> None:
        zarr.config.set(default_format=2)
        dst = self.base_url
        if self.init:
            logging.info("Writing metadata %s", dst)
            save_pyramid_to_s3(
                pyramid,
                dst,
                mode="w",
                compute=False,
                s3_options=self.s3_options,
            )
        if self.region:
            logging.info("Writing region %s → %s", self.region, dst)
            save_pyramid_to_s3(
                pyramid,
                dst,
                mode="r+",
                region=self.region,
                s3_options=self.s3_options,
            )


class Pipeline(ABC):
    def __init__(self, config: Config):
        self.source = Source(config)
        self.transform = Transform(config.regrid_function, chunking=config.chunking)
        self.destination = ZarrS3Sink(
            config.dst_s3url, config.init, region=config.region
        )

    def run(self) -> None:
        ds = self.source.load()
        pyramid = self.transform.convert(ds)
        self.destination.write(pyramid)
