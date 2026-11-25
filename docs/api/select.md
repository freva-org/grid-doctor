# Region selection

Select regions from HEALPix stores (`grid_doctor.select`).  They read
only the chunks a region touches and compute the coordinates of the
selected cells, so they also work on stores written without coordinate
arrays (`save_pyramid(write_coords=False)`).  See
[Working with the data](../high-level-access.md).

::: grid_doctor.select_bbox

::: grid_doctor.select_cone

::: grid_doctor.select_cells

::: grid_doctor.attach_cell_coords
