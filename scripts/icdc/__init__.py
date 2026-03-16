from .atmosphere.imerg import IMERGSpec
from .atmosphere.hoaps import HOAPSSpecP1M, HOAPSSpecPT6H
from .atmosphere.bnsc import BNSCSpecP1M
from .atmosphere.ecad import ECADSpecMeanP1D, ECADSpecSpreadP1D
from .atmosphere.crutem import CRUTEMSpecP1M
from .atmosphere.modis import MODISSpecAquaP1D, MODISSpecTerraP1D
from .atmosphere.mswep import MSWEPSpecPT3H

specs = [
    IMERGSpec,
    HOAPSSpecP1M,
    HOAPSSpecPT6H,
    BNSCSpecP1M,
    ECADSpecMeanP1D,
    ECADSpecSpreadP1D,
    CRUTEMSpecP1M,
    MODISSpecAquaP1D,
    MODISSpecTerraP1D,
    MSWEPSpecPT3H,
]

__all__: list[str] = ["specs"]
