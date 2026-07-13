import dask
from dataclasses import dataclass
import glob
import zarr
import grid_doctor as gd
import xarray as xr
import logging
from os import environ

logger = logging.getLogger(__name__)


dask.config.set(scheduler="single-threaded")

# /work/uc0928/DATA/atmosphere/modis_aqua_aerosol/DATA/2002/MODIS-C6.1__MYD08__daily__aerosol-parameters__20020704__UHAM-ICDC__fv0.2.nc
# /work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/2002/MODIS-C6.1__MYD08__daily__cloud-Fractions__20020704__UHAM-ICDC__fv0.1.nc
# /work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/2002/MODIS-C6.1__MYD08__daily__cloud-RadiativeProperties__20020704__UHAM-ICDC__fv0.1.nc
# /work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/2002/MODIS-C6.1__MYD08__daily__cloud-TopParameters__20020704__UHAM-ICDC__fv0.1.nc
# /work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/2002/MODIS-C6.1__MYD08__daily__cloud-WaterPaths__20020704__UHAM-ICDC__fv0.1.nc
# /work/uc0928/DATA/atmosphere/modis_aqua_watervapor_pwc_temperature/DATA/2002/MODIS-C6.1__MYD08__daily__watervapor-parameters__20020704__UHAM-ICDC__fv0.1.nc

variants: dict[str | stuple(str)] = {
    #        "histogram": tuple(
    #            "/work/uc0928/DATA/atmosphere/modis_aqua_aerosol/DATA/%Y/MODIS-C6.1__MYD08__daily__aerosol-parameters__histograms__%Y%m%d__UHAM-ICDC__fv0.1.nc",
    #            "/work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/%Y/MODIS-C6.1__MYD08__daily__cloud-parameters__histograms__%Y%m%d__UHAM-ICDC__fv0.1.nc",
    #        ),
    "aerosol": (
        "/work/uc0928/DATA/atmosphere/modis_aqua_aerosol/DATA/%Y/MODIS-C6.1__MYD08__daily__aerosol-parameters__%Y%m%d__UHAM-ICDC__fv0.2.nc",
    ),
    "cloud": (
        "/work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/%Y/MODIS-C6.1__MYD08__daily__cloud-Fractions__%Y%m%d__UHAM-ICDC__fv0.1.nc",
        "/work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/%Y/MODIS-C6.1__MYD08__daily__cloud-RadiativeProperties__%Y%m%d__UHAM-ICDC__fv0.1.nc",
        "/work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/%Y/MODIS-C6.1__MYD08__daily__cloud-TopParameters__%Y%m%d__UHAM-ICDC__fv0.1.nc",
        "/work/uc0928/DATA/atmosphere/modis_aqua_cloud/DATA/%Y/MODIS-C6.1__MYD08__daily__cloud-WaterPaths__%Y%m%d__UHAM-ICDC__fv0.1.nc",
    ),
    "watervapor": (
        "/work/uc0928/DATA/atmosphere/modis_aqua_watervapor_pwc_temperature/DATA/%Y/MODIS-C6.1__MYD08__daily__watervapor-parameters__%Y%m%d__UHAM-ICDC__fv0.1.nc",
    ),
}


@dataclass
class MODISAquaAtmConfig:
    year: int | str = "*"
    store_path: str = "icdc/healpix/atmosphere/MODIS/aqua/P1D/"  # relative path
    weights_path: str = "/work/ks1387/healpix_weights/modis_aqua_atm.nc"

    @property
    def pattern(self) -> str:
        return f"/work/uc0928/DATA/atmosphere/modis_aqua_*/DATA/{self.year}/*daily*.nc"

    def files_from_dataset(self, ds: xr.Dataset) -> list[str]:
        files = []
        for fmts in variants.values():
            for fmt in fmts:
                files.extend(ds.time.dt.strftime(fmt).data)

        logger.debug(
            "Dataset (time: #%d) requires loading %d files: [%s ... %s]",
            ds.time.size,
            len(files),
            files[0],
            files[-1],
        )
        return files

    def _filter_files(self, it):
        return filter(lambda x: "__histograms__" not in x, it)

    @property
    def files(self):
        if not hasattr(self, "_files"):
            self._files = sorted(self._filter_files(glob.glob(self.pattern)))
        return self._files

    def _open(self):
        if not hasattr(self, "_src_ds"):
            self._src_ds = gd.cached_open_dataset(
                self.files,
                decode_cf=True,
                parallel=False,
                compat="no_conflicts",
            )
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

    def write(self, start=0, files_per_batch=1):
        """Writes data in batches of ``files_per_batch``, using the xr.Dataset.to_zarr `region` option.
        - If in a SLURM array job, each index will write its independed slice slice.
        - Otherwise, regions will be writen sequentially"""
        array_id = int(environ.get("SLURM_ARRAY_TASK_ID", -1))
        if array_id > -1:
            if start > 0:
                logging.warning(
                    "Ignoring start argument, as this is array job (start = SLURM_ARRAY_TASK_ID * files_per_batch)"
                )
            self.write_region(
                region={
                    "time": slice(
                        array_id * files_per_batch, (array_id + 1) * files_per_batch
                    )
                }
            )
        else:
            for r in self.iter_regions(start=start, size=files_per_batch):
                self.write_region(region=r)

    def write_region(self, region=None | dict[str | slice], src_zoom: int = 5):
        """Writes a region to the initialized dataset in ``store`` by remapping only the implicated files from the original dataset"""
        store = f"{self.store_path.rstrip('/')}/level_{src_zoom}.zarr"
        hp_reg_ds = xr.open_zarr(store).isel(region)
        region_files = self.files_from_dataset(hp_reg_ds)
        reg_ds = xr.open_mfdataset(region_files, compat="no_conflicts")

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
