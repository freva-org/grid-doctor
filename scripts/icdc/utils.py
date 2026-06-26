import asyncio
import itertools
import logging
import math
import xarray

import s3fs
import zarr
from zarr.storage._fsspec import FsspecStore

from typing import Literal, Set, Any


logger = logging.getLogger(__name__)


def _get_store(root: zarr.Group) -> FsspecStore:
    store = root.store
    while hasattr(store, "store"):
        store = store.store
    assert isinstance(store, FsspecStore), f"Unexpected store type: {type(store)}"
    return store


def zarr_get_chunk_path_coords(
    root: zarr.Group, chunk_key_path: str
) -> (zarr.Array, tuple[int, ...]):
    """
    Reverse lookup of
        f"{arr.path}/{chunk_key}"

    Returns:
        (arr, coord)

    where:
        arr   -> zarr Array object
        coord -> tuple[int, ...]
    """

    # Build lookup of all arrays by path
    arrays = {}

    for path, member in root.members(max_depth=None):
        if isinstance(member, zarr.Array):
            arrays[member.path] = member

    # Find the longest array-path prefix
    matched_path = None

    for path in arrays:
        prefix = f"{path}/"

        if chunk_key_path.startswith(prefix):
            if matched_path is None or len(path) > len(matched_path):
                matched_path = path

    if matched_path is None:
        raise KeyError(f"No array matches chunk path: {chunk_key_path}")

    arr = arrays[matched_path]

    chunk_key = chunk_key_path[len(matched_path) + 1 :]

    zarr_format = getattr(arr.metadata, "zarr_format", 2)

    if zarr_format == 3:
        coord = arr.metadata.chunk_key_encoding.decode_chunk_key(chunk_key)

    else:
        sep = getattr(arr.metadata, "dimension_separator", ".") or "."
        coord = tuple(map(int, chunk_key.split(sep)))

    return arr, coord


def generate_chunk_paths(root: zarr.Group):
    for _, member in root.members(max_depth=None):
        if not isinstance(member, zarr.core.array.Array):
            continue
        arr = member
        shape = arr.shape
        chunks = arr.chunks
        grid_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))
        zarr_format = getattr(arr.metadata, "zarr_format", 2)

        for coord in itertools.product(*[range(n) for n in grid_shape]):
            if zarr_format == 3:
                chunk_key = arr.metadata.chunk_key_encoding.encode_chunk_key(coord)
            else:
                sep = getattr(arr.metadata, "dimension_separator", ".") or "."
                chunk_key = sep.join(map(str, coord))

            yield f"{arr.path}/{chunk_key}"


async def _check_missing(
    root: zarr.Group,
    fs: s3fs.S3FileSystem,
    base_path: str,
    max_concurrent: int = 10,
    missing_chunks: set | None = None,
) -> list[str]:
    chunk_paths = list(generate_chunk_paths(root))
    total_n_chunks = len(chunk_paths)
    print(f"\tTotal expected chunks: {total_n_chunks}")
    if missing_chunks:
        chunk_paths = [p for p in chunk_paths if p in missing_chunks]
        print(f"\tOnly checking {len(chunk_paths)} previously missing")

    semaphore = asyncio.Semaphore(max_concurrent)
    missing = []
    done = 0

    async def check(path: str):
        nonlocal done
        async with semaphore:
            full_path = f"{base_path}/{path}"
            try:
                if not await fs._exists(full_path):
                    missing.append(path)
            except Exception as e:
                print(f"WARN: could not check {path}: {e}")
            finally:
                done += 1
                if done % 1000 == 0:
                    print(
                        f"  Progress: {done}/{len(chunk_paths)}, {len(missing)} missing so far",
                        end="\r",
                    )

    await asyncio.gather(*[check(p) for p in chunk_paths])

    print(f"\nDone. {len(missing)} missing chunk(s) out of {total_n_chunks}.")

    return missing


def zarr_check_missing_chunks(
    root: zarr.Group,
    missing_chunks: set | None = None,
    max_concurrent: int = 10,
) -> list[str]:
    store = _get_store(root)

    # Reconstruct fs with correct storage options since they aren't forwarded
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={
            "endpoint_url": ENDPOINT_URL,
        },
    )

    return asyncio.run(
        _check_missing(root, fs, store.path, max_concurrent, missing_chunks)
    )


def size_to_chunk_batchsize(
    ds: xarray.Dataset,
    chunking: dict[str, int],
    max_batches=1000,
    target_bytes: int | None = None,
) -> int:
    """
    Determines number of chunks to write (according to `chunking`) such that in total they are close to `target_size` but up to `max_batches`

    Parameters:
        ds       (xarray.Dataset): XArray dataset chunked in `time`
        target_size         (int): Desired batch_size in bytes
        max_batches         (int): Maximum total number of batches

    Returns:
        Batch size -  number of chunks

    """
    if len(chunking) != 1:
        raise ValueError("Invalid chunking, should be have a single dimension")
    dim, chunk_size = next(iter(chunking.items()))
    dim_size = ds[dim].size
    bytes_per_chunk = ds.isel({dim: slice(0, chunk_size)}).nbytes

    if target_bytes is None:
        bs_min = dim_size // (chunk_size * max_batches) + 1
        logger.info(
            "Target size not specifed, defaulting to %s batches with %s chunks (%.03f GiB) per batch",
            max_batches,
            bs_min,
            bytes_per_chunk * bs_min / (2**30),
        )
        return bs_min

    bs_target = target_bytes // bytes_per_chunk

    if bs_min > bs_target:
        logger.info(
            "Batch target size is too small, would required more them %s batches. Limiting to %s chunks (%.03f GiB) per batch",
            max_batches,
            bs_min,
            bytes_per_chunk * bs_min / (2**30),
        )
        return bs_min

    logger.info(
        "Computed batch size of %s chunks (%s GiB)",
        max_batches,
        bs_target,
        bytes_per_chunk * bs_target / (2**30),
    )
    return bs_target


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
