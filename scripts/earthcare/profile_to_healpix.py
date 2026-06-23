import earthcarekit as eck
import healpix_geo as hpxg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xdggs
import zarr

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from earthcare_dggs.convert import lonlat_to_healpix_cells
from earthcare_dggs.settings import ATLID_DEPTH, ELLIPSOID

PRODUCT = "ACM_CAP_2B"
ORBIT = "01164E"
OUTPUT_PATH = ""

result = eck.search_product(file_type=PRODUCT, orbit_and_frame=ORBIT)
with eck.read_product(result.filepath[0]) as ds:
    profile = ds.load()

print(f"Dimensions: {dict(profile.sizes)}")
print(f"Lat: [{float(profile['latitude'].min()):.1f}, {float(profile['latitude'].max()):.1f}]")
print(f"Lon: [{float(profile['longitude'].min()):.1f}, {float(profile['longitude'].max()):.1f}]")
print(f"\n2D variables (along_track, vertical):")
for v in profile.data_vars:
    if profile[v].dims == ("along_track", "vertical"):
        print(f"  {v}: shape={profile[v].shape}")

prof_cell_ids = lonlat_to_healpix_cells(
    profile["longitude"].values, profile["latitude"].values,
    depth=ATLID_DEPTH, ellipsoid=ELLIPSOID,
)

unique_cells = np.unique(prof_cell_ids)
print(f"HEALPix depth {ATLID_DEPTH}: {len(profile.along_track)} profiles → {len(unique_cells)} unique cells")
print(f"Average profiles per cell: {len(profile.along_track) / len(unique_cells):.1f}")

# Add cell_ids as coordinate
profile_healpix = profile.assign_coords(cell_ids=("along_track", prof_cell_ids))
profile_healpix.attrs.update({
    "healpix_level": ATLID_DEPTH,
    "healpix_indexing": "nested",
    "healpix_ellipsoid": ELLIPSOID,
})

print(f"\nDataset with cell_ids coordinate:")
print(profile_healpix)

profile_healpix

# Add metadata
profile_healpix.attrs.update({
    "source": "EarthCARE MSI L2A",
    "healpix_level": ATLID_DEPTH,
    "healpix_indexing": "nested",
    "healpix_ellipsoid": ELLIPSOID,
    "orbit_and_frame": ORBIT,
})

# Add xdggs index
profile_healpix = xdggs.decode(
    profile_healpix,
    grid_info=xdggs.HealpixInfo(level=ATLID_DEPTH, indexing_scheme="nested"),
)

# Encode with xdggs convention
profile_encoded = xdggs.encode(profile_healpix, "xdggs")

# Write to Zarr
output_path = f"{OUTPUT_PATH}earthcare_ACM_CAP_2B_{ORBIT}.zarr"
profile_encoded.to_zarr(output_path, mode="w", consolidated=True)

# Add DGGS Zarr convention metadata (compatible with legacy-converters format)
dggs_convention = {
    "uuid": "7b255807-140c-42ca-97f6-7a1cfecdbc38",
    "name": "dggs",
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/dggs/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/dggs/blob/v1/README.md",
    "description": "Discrete Global Grid Systems convention for zarr",
}

dggs_meta = {
    "name": "healpix",
    "refinement_level": ATLID_DEPTH,
    "indexing_scheme": "nested",
    "ellipsoid": {
        "name": "wgs84",
        "semimajor_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    },
    "spatial_dimension": "cell_ids",
    "coordinate": "cell_ids",
    "compression": "none",
}

root = zarr.open_group(output_path, mode="r+")
root.attrs["zarr_conventions"] = [dggs_convention]
root.attrs["dggs"] = dggs_meta

print(f"Saved to {output_path}")
print(f"\nDGGS metadata: {dggs_meta}")

