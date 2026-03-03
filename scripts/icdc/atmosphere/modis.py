from icdc.base import BaseStructure
import logging
from os import getenv


class MODIS(BaseStructure):
    _open_kwargs = {'parallel':False} # otherwise it fails!?
    def __init__(self):
        year = '*'
        super().__init__({
            'MODIS/aqua/P1D/' :  f'/pool/data/ICDC/atmosphere/modis_aqua_watervapor_pwc_temperature/DATA/{year}/MODIS-C6.1__MYD08__daily__watervapor-parameters__[0-9]*__UHAM-ICDC__fv0.1.nc',
            'MODIS/terra/P1D/' : f'/pool/data/ICDC/atmosphere/modis_terra_watervapor_pwc_temperature/DATA/{year}/MODIS-C6.1__MOD08__daily__watervapor-parameters__[0-9]*__UHAM-ICDC__fv0.1.nc',
        })

    def convert(self, init: bool = True, region:dict = {'time':slice(0,1)}):
        from grid_doctor import (
            latlon_to_healpix_pyramid,
        )
        dst_url = 's3://icdc/healpix/atmosphere/'
        pyramids = {} 
        for name, ds in self.items():
            print(ds)
            ds_hp = latlon_to_healpix_pyramid(ds.chunk({'time':1}), method='nearest', keep_nans=True)
            pyramids[ dst_url + name ] = ds_hp
        
        self.write(pyramids, init=init, region=region)

def main():
    from argparse import ArgumentParser
    start_idx = int(getenv("SLURM_ARRAY_TASK_ID", 0))
    parser = ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--slice-size", default=48, type=int)
    parser.add_argument("--start", default=start_idx, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.slice_size % 48:
        print("slice-size must be a multiple of 48 (time chunk)")
        exit(1)

    region = {
        "time": slice(args.start * args.slice_size, (args.start + 1) * args.slice_size)
    }
    print(f"init={args.init}, region={region}")
    modis = MODIS()
    modis.convert(init=args.init, region=region)

        
if __name__ == '__main__':
    main()
