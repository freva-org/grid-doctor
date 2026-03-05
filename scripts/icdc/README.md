# ICDC sources

## Approach

There's a base class that opens, and iterates through the opened datasets for each source. It also provides a common implementation for writting to S3
The child class has to define the mapping zarr store -> fileglob, which is then used by the base case implementation.
It must also devine the `convert` method in which it iterates throught the opened datasets, applies the conversion and finaly evokes the `write` (either to inilialize the empty dataset, or add a region)

This approach aims at reducing duplicated code and offer a common way to process icdc datasets

There are cases one must use different parameters when opening the source datasets. E.g. some netcdf datasets cause segfault when opened in parallel.

### NaNs

So far NaNs are present in most datasets and when converting to healpix, they are lost (see [#3](https://github.com/freva-org/grid-doctor/issues/3)).

The current way of dealing with it is to apply the NaN mask after the regredding.
A side effect of this approach is that when coarsening the regridded dataset, NaN cells aren't processed and this their coarsened "parent cell" is left with the initialization value defined in `healpy.pixfunc.UNSEEN` which is (-1.6375e+30) (see [convention](https://healpy.readthedocs.io/en/latest/tutorial.html#Visualization)). To fix this NaNs are overwritten again, which may fail for variables with non float types. 
