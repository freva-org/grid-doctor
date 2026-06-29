from dataclasses import dataclass
import glob
import zarr
import grid_doctor as gd
import xarray as xr
import logging
from os import environ

logger = logging.getLogger(__name__)

import dask

dask.config.set(scheduler="single-threaded")


@dataclass
class IMERGConfig:
    pattern: str = "/pool/data/ICDC/atmosphere/imerg/DATA/*/IMERG_precipitationrate__V07B__halfhourly__0.1degree__*.nc"
    store_path: str = "icdc/healpix/atmosphere/IMERG/PT30M/"  # relative path!
    weights_path: str = "/work/ks1387/healpix_weights/imerg.nc"

    def region_to_files(self, region: dict[str, slice]) -> list[str]:
        if len(region) != 1 or "time" not in region:
            raise (
                ValueError(
                    "The specified region has no or unsupported dimension(s) (only 'time')"
                )
            )

        return self.files[region["time"]]

    @property
    def files(self):
        if not hasattr(self, "_files"):
            self._files = sorted(glob.glob(self.pattern))
        return self._files

    def _open(self):
        if not hasattr(self, "_src_ds"):
            self._src_ds = gd.cached_open_dataset(self.files, decode_cf=True)
        return self._src_ds

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
        self._store = store
        return store

    @property
    def zoom(self):
        if not hasattr(self, "_zoom"):
            _ds = xr.open_dataset(self.files[0])
            self._zoom = gd.resolution_to_healpix_level(gd.get_latlon_resolution(_ds))
        return self._zoom

    @property
    def hp_ds(self, level=None):
        if not hasattr(self, "_hp_ds"):
            store = f"{self.store_path.rstrip('/')}/level_{level or self.zoom}.zarr"
            try:
                self._hp_ds = xr.open_zarr(store)
            except zarr.errors.GroupNotFoundError:
                raise RuntimeError(f"Expected initialized dataset at {store}")
        return self._hp_ds

    def iter_regions(self, start=0, end=-1, size=1):
        assert start >= 0

        t_len = self.hp_ds.time.size
        end = max(end, t_len)
        if end > t_len:
            logging.warning("Restricting regions up until max index (%s)", t_len)
            end = t_len
        logger.info(
            "Iterating over %d regions of size %d", (end - start + size) % size, size
        )
        for i in range(start, end, size):
            yield {"time": slice(i, min(i + size, end))}

    def write(self, start=0, batch_size=1):
        array_id = int(environ.get("SLURM_ARRAY_TASK_ID", -1))
        if array_id > -1:
            if start > 0:
                logging.warning(
                    "Ignoring start argument, as this is array job (start = SLURM_ARRAY_TASK_ID * batch_size)"
                )
            self.write_region(
                region={
                    "time": slice(array_id * batch_size, (array_id + 1) * batch_size)
                }
            )
        else:
            for r in self.iter_regions(start=start, size=batch_size):
                self.write_region(region=r)

    def write_region(self, region=None | dict[str | slice]):
        """Writes a region to the initialized dataset in ``store`` by remapping only the implicated files from the original dataset"""
        region_files = self.region_to_files(region)
        reg_ds = xr.open_mfdataset(region_files)
        src_zoom = gd.resolution_to_healpix_level(gd.get_latlon_resolution(reg_ds))
        store = f"{self.store_path.rstrip('/')}/level_{src_zoom}.zarr"

        hp_reg_ds = xr.open_zarr(store).isel(region)
        logger.info(
            "Writing region %s into store at %s [ %s -> %s ]",
            region,
            store,
            str(hp_reg_ds.time[0].values),
            str(hp_reg_ds.time[-1].values),
        )

        dst_zoom = hp_reg_ds.attrs.get("healpix_level")
        if dst_zoom != src_zoom:
            raise RuntimeError(
                "Unable to safely write region, input resolution (%s) does not match the destinations' (%s)",
                str(src_zoom),
                str(dst_zoom),
            )

        assert hp_reg_ds.time.variable.identical(reg_ds.time.variable)

        remap_reg_ds = gd.regrid_to_healpix(
            reg_ds, dst_zoom, weights_path=self.weights_path
        )
        remap_reg_ds.drop_vars(set(remap_reg_ds.coords)).to_zarr(
            store, mode="r+", region=region
        )
