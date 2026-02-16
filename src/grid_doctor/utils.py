import hashlib
import logging
import numpy as np
import pickle
import tempfile
import xarray as xr
from os import environ
from pathlib import Path




def cache_dir() -> Path:
    """
    Returns the Path used for caching. 
    Tries setting a cache directory in `/scratch` setting `TMPDIR` accordingly.
    Otherwise fallsback to `tempfile.gettempdir()`.
    """
    scratch = Path('/scratch/{0:.1}/{0}'.format(environ['USER']))
    if scratch.is_dir():
        scratch_temp = scratch / tempfile.gettempprefix()
        try:
            scratch_temp.mkdir(exist_ok=True,parents=True)
        except PermissionError:
            logging.error("Unable to create cache in /scratch, continuing with default")
            pass
        else:
            environ['TMPDIR'] = str(scratch_temp)
            return scratch_temp
    return Path(tempfile.gettempdir)


def cached_open_dataset(files, **kwargs):
    """
    Opens specified files as single dataset, caching the resulting xarray.Dataset using pickle.
    """
    h = hashlib.sha256()
    h.update(np.sort(np.unique(files)).astype(bytes))
    pickle_file = cache_dir() / f'{h.hexdigest()}.pickle'

    if pickle_file.exists():
        logging.debug("Loading dataset from %s", pickle_file)
        with open(pickle_file,'rb') as f:
            _ds = pickle.load(f)
            return _ds

    print(f"Opening dataset with {len(files)} files ..." +(" "*40), flush=True, end='\r')
    from dask.diagnostics  import ProgressBar
    with ProgressBar():
        _kwargs  = {'parallel':True,'chunks':'auto'} | kwargs 
        _ds = xr.open_mfdataset(files, **_kwargs)

    with open(pickle_file,'wb') as f:
        logging.debug("Saving dataset in %s", pickle_file)
        pickle.dump(_ds, file=f)
    return _ds

