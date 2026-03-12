# Recipes

These recipes show end-to-end conversion workflows for different grid
types.  Each one follows the same three-step pattern:

1. **Open** the dataset (with caching).
2. **Convert** to a HEALPix pyramid.
3. **Upload** to S3.

| Recipe | Grid type | Source example |
|--------|-----------|----------------|
| [Structured Grids](structured.md) | Regular / curvilinear lat/lon | ERA5, CMIP6 |
| [Unstructured Grids (ICON)](icon.md) | Triangular mesh | ICON-DREAM |
