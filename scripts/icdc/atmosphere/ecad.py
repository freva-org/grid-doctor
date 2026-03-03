import logging

from argparse import ArgumentParser
from icdc.base import BaseStructure
from os import getenv

class ECAD(BaseStructure):
    _open_kwargs = {'parallel':True, 'engine':'h5netcdf', 'join':'left'} 
    def __init__(self):
        super().__init__({
            'ECAD/mean/' :  f'/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_mean_0.1deg_reg_v31.0e.nc',
            'ECAD/spread/' :  f'/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_spread_0.1deg_reg_v31.0e.nc',

        })
    
    @classmethod
    def chunking(cls):
        return {'time':1}

    def convert(self, init: bool = True, region:dict = {'time':slice(0,1)}):
        from grid_doctor import (
            latlon_to_healpix_pyramid,
        )
        dst_url = 's3://icdc/healpix/atmosphere/'
        pyramids = {} 
        for name, ds in self.items():
            print(ds)
            ds_hp = latlon_to_healpix_pyramid(ds.chunk(self.chunking()), method='nearest', keep_nans=True)
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

    if args.slice_size % ECAD.chunking()['time']:
        print("slice-size must be a multiple of 48 (time chunk)")
        exit(1)

    region = {
        "time": slice(args.start * args.slice_size, (args.start + 1) * args.slice_size)
    }
    print(f"init={args.init}, region={region}")
    ecad = ECAD()
    ecad.convert(init=args.init, region=region)

    

        
if __name__ == '__main__':
    main()
