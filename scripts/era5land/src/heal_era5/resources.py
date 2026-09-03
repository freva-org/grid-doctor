"""Locations of packaged resources and externally managed CMOR tables."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"

# Tables are downloaded separately so they are not included in package wheels.
# An editable checkout defaults to its local tables directory; installed packages
# should set HEAL_ERA5_TABLES_DIR to the era5-cmor-tables checkout.
LOCAL_TABLES_ROOT = PACKAGE_DIR.parents[1] / "tables" / "era5-cmor-tables"
CMOR_TABLES_ROOT = Path(os.environ.get("HEAL_ERA5_TABLES_DIR", str(LOCAL_TABLES_ROOT))).expanduser()
CMOR_TABLES_DIR = CMOR_TABLES_ROOT / "Tables"
