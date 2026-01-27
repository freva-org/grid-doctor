# Grid doctor HEALs your Grids

> [!NOTE]
> This is a scripting solution for a proof of concept. An operational ready
> approach will follow. For adding code for specific datasets please add
> your script solution into the `scripts/<yourname>` folder.


After you've cloned the repository

```console
git clone git@github.com:freva-org/grid-doctor.git && \
cd grid-doctor
```

You should follow these steps to get things working:


### Installation

Install the dependencies via pip

```console
python -m pip install -e .
```


### Add the solution to your dataset

```console
mkdir -p scripts/<yourname>

```
Then add your script solution into `scripts/<yourname>/convert.py`.
> [!IMPORTANT]
> Please add a descriptive README about what this script is trying to achieve.
> Also, document any problems you ran into.

### Usage

Please make use of the installed
[data-portal](https://github.com/freva-org/freva-nextgen/tree/main/freva-data-portal-worker)
libary and the `helpers` function within this repository.

An minimal example looks like this:

```python
import xarray as xr

from data_portal_worker.aggregator import DatasetAggregator
from data_portal_worker.rechunker import ChunkOptimizer
from grid_doctor import latlon_to_healpix_pyramid, save_pyramid_to_s3

agg = DatasetAggregator()
opt = ChunkOptimizer()

dset1 = xr.open_mfdataset("<path-to-file>", parallel=True, chunks="auto")
dset2 = xr.open_mfdataset("<path-to-file>", parallel=True, chunks="auto")

# The DatasetAggregator is able to combine non aggregatable data to into
# groups. This should be avoided though - try to aggregate into a single
# dataset.
dset_aggregated = agg.aggregate([dset1, dset2])["root"]
healpix_pyramid = latlon_to_healpix_pyramid(dset_aggregated)
chunked_heal_pix = {k: opt.apply(d) for k, d in healpix_pyramid.items()}
save_pyramid_to_s3(chunked_heal_pi, "s3://<bucket>/<path>.zarr",
                  s3_option={"https://s3.eu-dkrz-1.dkrz.cloud",
                             "key": os.getenv("S3_KEY"),
                             "serect": os.getenv("S3_SECRET")})

```
> [!CAUTION]
> DO NOT commit s3 keys or secrets to this repository. Use env variables.

More fine grained settings options for ``DatasetAggregator`` and ``ChunkOptimizer``
classes can be found in the
[DatasetAggregator](https://github.com/freva-org/freva-nextgen/blob/zarr-aggregation/freva-data-portal-worker/src/data_portal_worker/aggregator.py)
and the [ChunkOptimizer](https://github.com/freva-org/freva-nextgen/blob/zarr-aggregation/freva-data-portal-worker/src/data_portal_worker/rechunker.py)
source code.

> [!IMPORTANT]
> The example usage notebook is for demo purpose only. Pleas do not submit a
> jupyter notebook as your solution.
>
>
## Issues

As this is still very much work in progress it is very likely that you will
run into Problems. Please note any of those problems in the `README.md` file
for your dataset folder. Feel free to submit PR's if there are any issues
with the ``DatasetAggregator`` or ``ChunkOptimizer`` classes. If you don't feel
comfortable with submitting PR's you can file an issue report
[here](https://github.com/freva-org/freva-nextgen/issues)
