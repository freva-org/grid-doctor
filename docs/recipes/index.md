# Recipes

Worked examples from source data to a published HEALPix pyramid. The
conversion recipes follow the same steps: open the data, get the weights
(or bin the samples), build the pyramid and write it.

| Recipe | Source data | Examples |
|--------|-------------|----------|
| [Structured grids](structured.md) | Regular and curvilinear lat/lon grids | ERA5, CMIP6 |
| [Unstructured grids](icon.md) | Triangular meshes | ICON |
| [Regional grids](regional.md) | Limited-area and rotated-pole domains | CORDEX |
| [Point data](point-data.md) | Swaths, stations, trajectories | EarthCARE MSI, station networks |

Two recipes cover what comes after or around the conversion:

- [Coarsening and averaging masked fields](averaging.md) computes
  means and area integrals of fields with missing values consistently
  across pyramid levels.
- [Parallel writes](parallel-writes.md) splits the conversion of a large
  source into independent array-job tasks driven by a manifest, with a
  complete [reflow](https://reflow-hpc.org) workflow.

To read the published data, including stores without coordinate arrays,
see [Working with the data](../high-level-access.md).
