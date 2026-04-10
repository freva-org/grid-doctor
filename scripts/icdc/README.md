# ICDC sources

## Desgin Approach

Since there are many sources in the case of ICDC, there is a pipeline (`Pipeline`) class that accepts 3 other, `Source`, `Transform`, `Sink`, that determines how to treat each input dataset.
One only needs to instanciante yet another class, `Config`, that parameterizes the process. The aim is that minimal amount of code need to be added for a new dataset.
To make structure things further, we use the class `Collection` which is an enum derived class that enumerates each of its' `Config` and implements a method `run_pipelines` that triggers the pipeline execution for each.

The collections code is structured into icdc/<realm>/<collection>.py

### NaNs

So far NaNs are present in most datasets and when converting to healpix, they are ignored (see [#3](https://github.com/freva-org/grid-doctor/issues/3)).

The current way of dealing with it is to apply the NaN mask after the regredding.
A side effect of this approach is that when coarsening the regridded dataset, NaN cells aren't processed and this their coarsened "parent cell" is left with the initialization value defined in `healpy.pixfunc.UNSEEN` which is (-1.6375e+30) (see [convention](https://healpy.readthedocs.io/en/latest/tutorial.html#Visualization)). To fix this NaNs are overwritten again, which may fail for variables with non float types. 
