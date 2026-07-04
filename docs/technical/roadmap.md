This page documents the open design decisions, areas that require particular
care, and the remaining roadmap for completing the Waterpark project.

Waterpark is more than a collection of converted datasets. It combines data
transformation, storage layout, public access, catalogue integration, and
operational workflows. Some parts of this system are already working as a
prototype, while others still need to be refined before the service can become
reliable, maintainable, and useful for a broader user community.

The goal is to make the remaining decisions explicit, document the reasoning
behind them, and identify where extra caution is needed before
scaling Waterpark from a prototype into a production-ready data hub.

---


## Open decisions

!!! question "Can we define an appropriate naming convention?"

    Suggestion: `<bucket>/healpix/<experiment-compaign>/<model>/<freq>/level_X.zarr"`

    Naming conventions can be quite different for different datasets, but we
    can still aim at having a number **four** directory levels. If those
    *four* levels don't guarantee unique paths the directory names themselves
    can be adjusted to from uniq name patterns such as:

    ```bash
    <bucket>/healpix/<product>/<instrument-level>/<freq>/level_X.zarr
    <bucket>/healpix/<product-experiment>/<model-ensemble>/<freq>/level_X.zarr
    ```

    The output time frequency `freq` should follow ISO 8601 standard.

!!! question "Where to store cached weight files?"


    Weight files are reusable across runs for the same source grid and
    target level. The weight files with their grid signature should be stored
    at the following location:

    ```console
    ls /work/ks1387/healpix-weights

    weights_0ba7e6dca9ba1ae9.nc  weights_3051722bc32a01e5.nc  weights_46fdfc6feb8ea520.nc  weights_d9c2730b22295f4a.nc
    weights_2aff1785f62b0254.nc  weights_3a28f272e1fb6024.nc  weights_901fbfc4a3ce2458.nc
    ```

    To make sure that the weight files are getting reproducible and reusable
    stored use `cache_path=/work/ks1387/healpix-weights` weights generation.

!!! question "Zarr format: v2 or v3?"

    Some clients  (gdal,  zarrita) still require Zarr v2.
    Zarr v3 is the future standard but ecosystem support is still
    catching up.

    **Current decision:** Write Zarr v2 with consolidated metadata
    until gridlook supports v3.

---

## Roadmap

### Completed

- [x] Conservative remapping pipeline (`grid-doctor`) with GPU acceleration
- [x] 1st round of Datasets uploaded


### Next steps

- [ ] Operationalise the remapping pipeline (reproducible batch jobs)
- [ ] Set up shared weight-file cache on Lustre
- [ ] Automatic tape archival and retrieval of Zarr stores
- [ ] STAC catalogue integration for data discoverability
- [ ] Freva databrowser registration
- [ ] Documentation of the full process and methodology
