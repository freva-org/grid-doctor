"""Constants and invariants."""

# ===================================================================
# Well-known coordinate and dimension names
# ===================================================================

_UNSTRUCTURED_DIMS: frozenset[str] = frozenset({"cell", "ncells", "ncell", "nCells"})
"""Dimension names that signal an unstructured source grid."""

_LAT_NAMES: tuple[str, ...] = (
    "clat",
    "lat",
    "latitude",
    "LAT",
    "LATITUDE",
    "Latitude",
    "XLAT",
    "XLAT_M",
    "XLAT_U",
    "XLAT_V",
    "nav_lat",
    "nav_lat_rho",
    "lat_rho",
    "lat_u",
    "lat_v",
    "lat_psi",
    "gridlat_0",
    "g0_lat_0",
    "yt_ocean",
    "yu_ocean",
    "geolat",
    "geolat_t",
    "geolat_c",
)
"""Priority-ordered latitude variable names recognised by the backend."""

_LON_NAMES: tuple[str, ...] = (
    "clon",
    "lon",
    "longitude",
    "LON",
    "LONGITUDE",
    "Longitude",
    "XLONG",
    "XLONG_M",
    "XLONG_U",
    "XLONG_V",
    "nav_lon",
    "nav_lon_rho",
    "lon_rho",
    "lon_u",
    "lon_v",
    "lon_psi",
    "gridlon_0",
    "g0_lon_0",
    "xt_ocean",
    "xu_ocean",
    "geolon",
    "geolon_t",
    "geolon_c",
)
"""Priority-ordered longitude variable names recognised by the backend."""

_Y_CANDIDATES: tuple[str, ...] = (
    "rlat",
    "lat",
    "latitude",
    "y",
    "j",
    "nj",
    "south_north",
    "south_north_stag",
    "eta_rho",
    "eta_u",
    "eta_v",
    "eta_psi",
    "yh",
    "yq",
    "njp1",
)
"""Lower-cased dimension names considered as latitude / y axes."""

_X_CANDIDATES: tuple[str, ...] = (
    "rlon",
    "lon",
    "longitude",
    "x",
    "i",
    "ni",
    "west_east",
    "west_east_stag",
    "xi_rho",
    "xi_u",
    "xi_v",
    "xi_psi",
    "xh",
    "xq",
    "nip1",
)
"""Lower-cased dimension names considered as longitude / x axes."""


__all__ = [
    '_UNSTRUCTURED_DIMS',
    '_LAT_NAMES',
    '_LON_NAMES',
    '_Y_CANDIDATES',
    '_X_CANDIDATES',
]
