# EarthCARE to HEALPix DGGS

## About EarthCARE

[EarthCARE](https://www.esa.int/Applications/Observing_the_Earth/FutureEO/EarthCARE) (Earth Cloud, Aerosol and Radiation Explorer) is a satellite mission operated jointly by ESA and JAXA, launched in May 2024. Its goal is to improve our understanding of how clouds and aerosols influence Earth's radiation budget and climate.

The satellite carries four complementary instruments:

- **ATLID** (ATmospheric LIDar) — an ESA lidar that measures vertical profiles of aerosols and thin clouds along the satellite's nadir track.
- **CPR** (Cloud Profiling Radar) — a JAXA 94 GHz Doppler radar that profiles thicker clouds and precipitation, including their internal vertical dynamics.
- **MSI** (Multi-Spectral Imager) — a wide-swath passive imager providing spatial context across multiple wavelengths (~500 m resolution).
- **BBR** (BroadBand Radiometer) — measures reflected solar radiation and outgoing infrared radiation to characterise the top-of-atmosphere radiation budget (~10 km footprint).

Together, these instruments allow scientists to simultaneously observe the vertical structure, horizontal distribution, and radiative impact of clouds and aerosols — measurements critical for evaluating and improving weather and climate models.

---

## Notebooks

This repository provides notebooks for converting EarthCARE Level 2 data products into HEALPix DGGS (Discrete Global Grid System) format using [healpix-geo](https://github.com/eopf-dggs/healpix-geo), [xdggs](https://github.com/xarray-contrib/xdggs) and [earthcare-dggs](https://annefou.github.io/earthcare-dggs/). Three conversion patterns are covered, each matching a different EarthCARE data geometry:

### `footprint_to_healpix.ipynb` TBD

Converts **footprints of vertical profile data and BBR fluxes** to HEALPix. BBR observations are single-point footprints with a large spatial extent (~10 km). Each footprint is mapped to the nearest HEALPix cell at a coarse resolution appropriate for the instrument's sampling size.

### `profile_to_healpix.ipynb`

Converts **ATLID and CPR vertical profile data** to HEALPix. Both instruments measure along a 1D nadir track, producing curtains of atmospheric profiles. The conversion maps each along-track profile location to a HEALPix cell while preserving the vertical dimension.

### `swath_to_healpix.ipynb` TBD

Converts **MSI 2D swath data** to HEALPix. MSI provides a wide swath of 2D imagery. The conversion aggregates the high-resolution swath pixels into HEALPix cells, accounting for the varying overlap between swath pixels and grid cells.

---

## HEALPix Resolution per Instrument

The target HEALPix depth for each instrument is chosen to match its native spatial resolution, avoiding both oversampling and unnecessary data loss:

| Instrument | HEALPix Depth | Approximate Cell Size | Native Resolution |
|---|---|---|---|
| MSI | 17 | ~458 m | ~500 m |
| ATLID | 14 | ~3.7 km | ~1 km along-track |
| CPR | 14 | ~3.7 km | ~800 m horizontal |
| BBR | 10 | ~58 km | ~10 km footprint |


---

## Citation

The conversion algorithms implemented in these notebooks are based on:

> Fouilloux, A. (2026). *EarthCARE to DGGS* (v0.1.0). Zenodo.
> https://doi.org/10.5281/zenodo.19709327
