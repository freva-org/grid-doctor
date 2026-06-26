from dataclasses import dataclass
import glob
import zarr
import grid_doctor as gd
from pathlib import Path
import xarray as xr
import logging

logger = logging.getLogger(__name__)


@dataclass
class IMERGConfig:
    pattern: str = "/pool/data/ICDC/atmosphere/imerg/DATA/2025/IMERG_precipitationrate__V07B__halfhourly__0.1degree__*.nc"
    store_path: str = "icdc/healpix/atmosphere/IMERG/PT30M/"  # relative path!
    weights_path: str = "/work/ks1387/healpix_weights/imerg.nc"
    chunks_map = {"time": 12}

    def _open(self):
        return gd.cached_open_dataset(sorted(glob.glob(self.pattern)), decode_cf=True)

    def init(self, overwrite=False):
        """Initializes an empty zarr store with the remaped version of the **opened** dataset(s)."""
        ds = self._open()
        zoom = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
        remap_ds = gd.regrid_to_healpix(ds, zoom, weights_path=self.weights_path)

        from utils import init_full_zarr_store

        store = f"{self.store_path.rstrip('/')}/level_{zoom}.zarr"
        logger.info(
            "%s zoom level %s in %s",
            "Overwriting" if overwrite else "Initializing",
            zoom,
            store,
        )
        init_full_zarr_store(remap_ds, store, overwrite=overwrite)
