import logging
import xarray

from typing import Literal, Set, Any


logger = logging.getLogger(__name__)


# This function is useful to speed up writting empty zarr stores for big datasets
# Since xarray.Dataset.to_zarr(mode='w', compute=False) may process huge dask graphs
# This is tacken from https://github.com/pydata/xarray/issues/8343#issuecomment-3364428741
def make_dataset_template(
    dataset: xarray.Dataset,
    lazy_vars: Set[str] | None = None,
    full: bool = False,
) -> xarray.Dataset:
    """Make a lazy Dask xarray.Dataset for use only as a template.

    Lazy variables in an xarray.Dataset can be manipulated with xarray operations,
    but cannot be computed.

    Args:
      dataset: dataset to convert into a template.
      lazy_vars: optional explicit set of variables to make lazy. By default, all
        data variables and coordinates that are not used as an index are made
        lazy, matching xarray.Dataset.chunk.

    Returns:
      Dataset with lazy variables. Lazy variable each use a single Dask chunk.
      Non-lazy variables are loaded in memory as NumPy arrays.
    """
    import dask

    if lazy_vars is None:
        lazy_vars = set(dataset.keys())
        lazy_vars.update(k for k in dataset.coords if k not in dataset.indexes)

    result = dataset.copy()

    # load non-lazy variables into memory
    result.update(dataset.drop_vars(lazy_vars).compute())

    def _raise_template_error():
        raise ValueError(
            "cannot compute array values of xarray.Dataset objects created directly "
            "or indirectly from make_dataset_template()"
        )

    # override the lazy variables
    delayed = dask.delayed(_raise_template_error)()
    for k, v in dataset.variables.items():
        if k in lazy_vars or full:
            # names of dask arrays are used for keeping track of results, so arrays
            # with the same name cannot have different shape or dtype
            name = f"make_dataset_template_{'x'.join(map(str, v.shape))}_{v.dtype}"

            result[k].data = dask.array.from_delayed(
                delayed, v.shape, v.dtype, name=name
            )

    return result


def _get_encoding(ds: xarray.Dataset) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "chunks": ds[k].data.chunksize if ds[k].ndim > 1 else ds[k].size,
        }
        for k in ds.variables  # keys()
    }


def init_full_zarr_store(
    ds: xarray.Dataset,
    store: str,
    indexes=True,
    dims=True,
    coords=True,
    overwrite=False,
    zarr_format: Literal[2, 3] = 2,
    encoding=None,
):
    """Will initialize an empty zarr store from a **full** xarray.Dataset
    Writes only the metadata and by default full single chunked indexes, dimentions and coordinates.

    Assumes `ds` is the full dataset. And later, writes happen in parallel over non-overlapping regions

    It initializes the store as a single task on the dask graph, even if `ds` has fine chunking

    The chunking(and sharding) is inferred from ds
    """

    template_ds = make_dataset_template(ds, lazy_vars=set(ds.keys()))
    # Follow input dataset encoding
    if encoding is None:
        encoding = _get_encoding(ds)
        logger.debug(
            "Encoding not specified, using default method, where 1D elements are put in single chunk: %s",
            encoding,
        )

    if zarr_format == 2:
        from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding

        key_enc = {"chunk_key_encoding": V2ChunkKeyEncoding(separator="/").to_dict()}
        for v in encoding.values():
            v.update(key_enc)

    # to get rid of the consolidated metadata warning
    import warnings
    from zarr.errors import ZarrUserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ZarrUserWarning, message=".*consolidated.*"
        )

        if overwrite:
            logger.debug("Overwritting %s", store)
            template_ds.to_zarr(
                store,
                mode="w",
                compute=False,
                encoding=encoding,
                zarr_format=zarr_format,
            )
        else:
            try:
                logger.debug("Initializing %s", store)
                template_ds.to_zarr(
                    store,
                    mode="w-",
                    compute=False,
                    encoding=encoding,
                    zarr_format=zarr_format,
                )
            except FileExistsError as e:
                raise FileExistsError(
                    f"Can't overwrite zarr store {store} by default"
                ) from e

    # Warn for unchunked variables that could create
    # memory problems once loaded/computed
    for var in template_ds.data_vars:
        template_ds[var].chunks
