# Technical decisions

This document records the design rationale behind the significant
technical choices in grid-doctor.  It is intended as a reference for
contributors, reviewers, and downstream consumers of the produced
HEALPix pyramids.

---

## Why HEALPix?

HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) tiles the
sphere into pixels of exactly equal area at every resolution level.
This property is critical for climate data:

- **Area-weighted statistics are trivial.**  Every pixel covers the same
  solid angle, so a global mean is just the arithmetic mean of all
  pixels, no latitude-dependent cosine weighting is needed.
- **Hierarchical nesting** gives a natural multi-resolution pyramid.
  Four child pixels compose exactly one parent pixel, making
  coarsening a simple reshape operation with no geometric computation.
- **No singularities at the poles.**  Regular lat-lon grids have
  converging meridians and a vanishing cell area at the poles.
  HEALPix avoids this entirely.

### Nested ordering, not ring

HEALPix defines two pixel numbering schemes: **ring** ordering (pixels
numbered along iso-latitude rings) and **nested** ordering (pixels
numbered by a hierarchical quad-tree).  Grid-doctor defaults to nested
ordering because it makes coarsening free.

In nested ordering, a parent pixel at level $L$ with index $i$ has
children $4i$, $4i+1$, $4i+2$, $4i+3$ at level $L+1$.  Coarsening
the full grid therefore reduces to reshaping the data array from
$(n_\text{cells},)$ to $(n_\text{cells} / 4,\; 4)$ and reducing along
the last axis, requiring no index lookup, no scatter, and no sorting.

Ring ordering does not have contiguous parent-child layout.  Grid-doctor
supports it, but in that mode every pyramid level is independently
remapped from the source grid, which is substantially more expensive.


## Spherical geometry

All HEALPix cell boundaries and centre coordinates are computed on a
**perfect sphere**, not on the WGS84 ellipsoid.  The remapping software
we use,
[ESMF](https://earthsystemmodeling.org/regrid/) (Earth System Modeling
Framework, a widely used library for regridding Earth-system model
output), also assumes a perfect sphere for its internal overlap area
calculations.  Using the same geometry for the HEALPix target mesh
avoids systematic area errors in the weight matrix.

This choice is also consistent with climate models (ICON, IFS, MPAS, …),
which use geocentric spherical coordinates internally.  The maximum
latitude discrepancy between spherical and WGS84 geodetic coordinates
is about 0.19° at 45° latitude, well within the spatial resolution of
any current global climate dataset.


## Target level selection

The target HEALPix level for each source dataset is chosen as the
highest level whose characteristic pixel spacing is coarser than or
comparable to the source resolution.  The heuristic is:

$$
\ell = \left\lfloor \log_2 \frac{58.6°}{\Delta} \right\rfloor
$$

where $\Delta$ is the estimated source grid spacing in degrees.  The
constant 58.6° is the approximate pixel spacing at level 0,
derived from $360° \;/\; \sqrt{12 \times 4^0}$.

This ensures that the HEALPix grid does not oversample the source.
Lower pyramid levels are derived by coarsening, not by remapping, so
no additional information is fabricated.

| Level | nside | Pixels       | Approx. spacing |
|------:|------:|-------------:|:---------------:|
|     0 |     1 |           12 | 58.6°           |
|     3 |     8 |          768 | 7.3°            |
|     5 |    32 |       12 288 | 1.8°            |
|     7 |   128 |      196 608 | 27.5′           |
|     9 |   512 |    3 145 728 | 6.9′            |
|    10 |  1024 |   12 582 912 | 3.4′            |


## Remapping methods

Remapping is performed by ESMF, which computes the geometric overlap
between source and target cells and produces a sparse weight matrix
that is applied to the data.  Grid-doctor supports two remapping
methods, chosen by variable type.

### Conservative remapping (continuous fields)

Used for temperature, precipitation, radiation, wind, and all other
fields that represent continuous physical quantities.

Conservative remapping computes area-weighted averages: each weight is
proportional to the fractional overlap area between a source cell and a
target cell.  The key property is **integral preservation**, the
area-integral of the field over any target cell equals the sum of the
area-integrals of the contributing source cells.

This is the only method that correctly handles sub-pixel variability.
If a HEALPix cell straddles a strong SST gradient, the conservative
average reflects the true area-weighted mean rather than the value at a
single sampled point.

### Nearest-neighbour remapping (categorical fields)

Used for land-sea masks, land cover classes, soil type, and all other
fields where values are discrete labels.

Nearest-neighbour assigns each target cell the value of the closest
source cell centre.  This guarantees that the output contains only
values that actually exist in the source, no interpolated intermediate
classes like "land-use 3.7" can appear.

Linear or bilinear interpolation is explicitly **not** supported.  While
it can produce smoother-looking fields, it fabricates values that were
not present in the original data.


## Weight generation

Remapping weights are precomputed and stored as reusable NetCDF files.
Weight generation is the most expensive step in the pipeline, but it
only needs to happen once per combination of source grid and target
HEALPix level.  Subsequent runs reuse the cached weight file.

For moderate grids, weights are generated in memory using ESMF's Python
bindings (ESMPy).  For very large grids where this would exceed
available RAM, grid-doctor writes both meshes to temporary files and
invokes the standalone `ESMF_RegridWeightGen` command under MPI, with a
configurable number of ranks.

Grid-doctor automatically classifies the source grid and constructs the
appropriate mesh representation:

| Source type     | Examples                    | Detection |
|:----------------|:----------------------------|:----------|
| Regular         | ERA5, CMIP on lat-lon grids | 1-D `lat` and `lon` coordinates |
| Curvilinear     | NEMO, WRF                   | 2-D `lat(y, x)` and `lon(y, x)` |
| Unstructured    | ICON, MPAS                  | Dimension named `cell` / `ncells`, or `CDI_grid_type` attribute |

Every weight file is annotated with provenance metadata
(`grid_doctor_method`, `grid_doctor_level`, `grid_doctor_order`, source
grid details) so that downstream tools can verify how the weights were
produced.


## Weight application

Applying the weight matrix to data is a sparse matrix multiplication.
The full batch of time steps is reshaped into a single dense matrix and
multiplied in one operation, which is typically 10–50× faster than
applying the weights one time step at a time.  GPU acceleration (via
CuPy) and JIT compilation (via Numba) are used automatically when those
packages are installed.  See
[Performance: application backends](#performance-application-backends)
for details.


## Missing-value handling during remapping

Source datasets commonly contain NaN values, for instance land pixels in
an ocean dataset, gaps in satellite swaths, or masked regions.  The
`missing_policy` parameter controls how these are treated during weight
application.

### Renormalize (default)

Source cells that are NaN are excluded from the weighted sum.  The
weights of the remaining valid source cells are rescaled so they sum
to 1.  A target cell receives a valid value as long as at least one
contributing source cell is valid.

This is the appropriate choice for virtually all climate fields.  A
HEALPix cell on a coastline that partly overlaps land (NaN) and partly
ocean still receives a valid SST from the ocean fraction.

### Propagate

If any contributing source cell is NaN, the target cell is set to NaN.
No partial averages are computed.

This mode exists for strict budget-closure applications.  If you are
integrating total precipitation over a HEALPix cell and one source cell
is missing, the renormalized average looks valid but represents a biased
estimate over a subset of the area.  For budget-closure studies it is
preferable to have a NaN than a plausible-looking number derived from
incomplete coverage.

Note that this mode is very aggressive: a single NaN source cell at the
edge of a target cell's footprint is enough to discard it.  Along
coastlines, swath edges, and orbit gaps, large portions of the output
will be NaN.


## Pyramid construction and coarsening

The multi-resolution pyramid is built by first remapping the source
dataset to the finest HEALPix level, then deriving all coarser levels
by iterated coarsening, one level at a time, always a factor of 4.

### Why coarsen rather than remap at each level?

Remapping from the original source grid at every level would be correct
but wasteful: each level would require a separate weight file, and
weight generation cost scales with the product of source and target
cell counts.  Coarsening is essentially free, a reshape plus a
vectorised reduction, and produces identical results for nested
HEALPix grids because the nested pixel hierarchy exactly matches the
coarsening hierarchy.

### Mean coarsening (continuous fields)

For continuous fields (those remapped with conservative weights), each
parent cell's value is the NaN-aware mean of its 4 children.

### Mode coarsening (categorical fields)

For categorical fields (those remapped with nearest-neighbour weights),
averaging class labels is meaningless.  Instead, each parent cell
receives the **mode** (most frequent value) of its valid children.
Ties are broken by the first occurrence.

The coarsening mode is inferred automatically from the remapping method
stored in the dataset attributes: `grid_doctor_method = "nearest"`
triggers mode coarsening, `"conservative"` triggers mean coarsening.
An explicit `coarsen_mode` parameter is available to override this.

### Minimum valid fraction (default: 50%)

A parent cell is set to NaN when fewer than half of its children are
valid (at least 2 of 4).

**Representativeness.**  A parent cell's value should represent the
majority of its area.  With at least 2 of 4 valid children, the value
is guaranteed to cover at least half the parent cell's area.  Below
that, a single child pixel's value would "speak for" 3 other pixels
that have no data, which is indistinguishable from interpolation into
unknown territory.

**Cascade prevention.**  Each coarsening step is a factor-of-4
reduction.  If only 1 of 4 valid children were sufficient, a single
valid pixel at the finest level could survive every coarsening step:
1/4 at level $N-1$, that surviving parent is again 1/4 at level $N-2$,
and so on.  After $k$ steps it represents a fraction of $4^{-k}$ of the
area it claims to cover.  This is the "Siberian lake problem" observed
with MODIS SST: a single lake pixel at level 10 bled through 7
coarsening steps to level 3, covering a 16 384× larger area.  With a
50% threshold, the cascade is impossible, the pixel is discarded at
the very first step because 1/4 < 1/2.  A feature can only survive
coarsening if it covers at least half the area at every scale, which is
exactly when it is a real, resolvable feature at that resolution.


## Output metadata and CRS convention

Every output dataset carries a standardised set of metadata.

### Global attributes

| Attribute                          | Description |
|:-----------------------------------|:------------|
| `healpix_level`                    | HEALPix refinement level |
| `healpix_nside`                    | $2^\ell$ |
| `healpix_order`                    | `nested` or `ring` |
| `grid_doctor_version`              | Package version that produced the data |
| `grid_doctor_method`               | `conservative` or `nearest` |
| `grid_doctor_coarsened_from_level` | Immediate parent level (coarsened levels only) |

### CRS variable

A scalar coordinate variable named `crs` follows the CF `grid_mapping`
convention.  It carries attributes that describe the coordinate
reference system:

```
crs: float64
    grid_mapping_name: healpix
    healpix_nside: 1024
    healpix_level: 10
    healpix_order: nested
```

Every spatially-dimensioned data variable carries
`grid_mapping = "crs"` so that CF-aware tools can discover the
projection automatically.  This convention is validated against the
[gridlook](https://gridlook.pages.dev/) viewer, which uses
`grid_mapping_name == "healpix"` to detect and render HEALPix datasets.


## Storage format

Pyramids are written as Zarr v2 stores with consolidated metadata, one
store per level (`level_0.zarr`, `level_1.zarr`, …).  Zarr v2 is used
because ecosystem support for v3 is still maturing: client libraries
such as `zarr-python`, `zarr-js` (used by gridlook and similar web
viewers), `xarray`, and `zarrita` have stable, well-tested v2
implementations, while v3 support varies in completeness.  The
transition to v3 is planned once the client ecosystem has converged.

Zarr's chunked, cloud-native layout enables byte-range reads from S3
without downloading entire files, which is essential for interactive
visualisation where only a single chunk of a single variable at a single
level needs to be fetched.


---

## Performance: application backends

The sparse weight-matrix multiplication supports three backends,
selected automatically based on what is installed.

**Batched SciPy** (always available) reshapes all time steps into a
single dense matrix and performs one sparse matmul, delegating to BLAS
for the dense side.  This is typically 10–50× faster than per-slice
application.

**Numba** (optional) JIT-compiles a fused kernel that computes the
weighted sum, NaN mask, and renormalization in a single pass over the
sparse matrix structure.  This halves memory traffic for single-slice
workloads.

**CuPy** (optional) transfers the weight matrix to the GPU once and
applies it via cuSPARSE.  This is worthwhile for large target grids
(HEALPix level 10+) with many time steps, where the transfer cost is
amortised over the batch.

The backend can be overridden explicitly via `backend="scipy"`,
`"numba"`, or `"cupy"` in any remapping call.
