import logging

from argparse import ArgumentParser
from icdc.base import BaseStructure
from os import getenv

class HOAPS(BaseStructure):
    _open_kwargs = {'parallel':False, 'decode_timedelta':False} 
    def __init__(self):
        super().__init__({
#            'HOAPS/P1M/' : '/pool/data/ICDC/atmosphere/hoaps/DATA/Precipitation/monthly/PRE*.nc',
            'HOAPS/PT6H/' : '/pool/data/ICDC/atmosphere/hoaps/DATA/Precipitation/6hourly/*/PREic*SCPOS01GL.nc',
        })
    
    @classmethod
    def chunking(cls):
        return {'time':4}

    def convert(self, init: bool = True, region:dict = {'time':slice(0,1)}):
        from grid_doctor import (
            latlon_to_healpix_pyramid,
        )
        dst_url = 's3://icdc/healpix/atmosphere/'
        pyramids = {} 
        for name, ds in self.items():
            ds_hp = latlon_to_healpix_pyramid(ds.chunk(self.chunking()), method='nearest', keep_nans=True)
            print(ds_hp[6])
            pyramids[ dst_url + name ] = ds_hp
        
        self.write(pyramids, init=init, region=region)


def main():
    start_idx = int(getenv("SLURM_ARRAY_TASK_ID", 0))
    parser = ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--slice-size", default=48, type=int)
    parser.add_argument("--start", default=start_idx, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    time_chunk = HOAPS.chunking()['time']

    if args.slice_size % time_chunk:
        print(f"slice-size must be a multiple of {time_chunk} (time chunk)")
        exit(1)

    region = {
        "time": slice(args.start * args.slice_size, (args.start + 1) * args.slice_size)
    }
    print(f"init={args.init}, region={region}")
    ecad = HOAPS()
    ecad.convert(init=args.init, region=region)

    

        
if __name__ == '__main__':
    main()
