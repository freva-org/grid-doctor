from typing import Any
from types import FunctionType

def load_collections():
    from .atmosphere.imerg import IMERG
    from .atmosphere.hoaps import HOAPS
    from .atmosphere.bnsc import BNSC
    from .atmosphere.ecad import ECAD
    from .atmosphere.crutem import CRUTEM
    from .atmosphere.modis import MODIS
    from .atmosphere.mswep import MSWEP

    collections =   \
        BNSC,   \
        CRUTEM, \
        ECAD,   \
        HOAPS,  \
        IMERG,  \
        MODIS,  \
        MSWEP,  \

    return collections

_LAZY_IMPORTS: dict[str, str] = {
        'collections' : load_collections,
        "IMERG": ".atmosphere.imerg",
        "HOAPS": ".atmosphere.hoaps",
        "BNSC" : ".atmosphere.bnsc",
        "ECAD ": ".atmosphere.ecad",
        "CRUTE": ".atmosphere.crutem",
        "MODIS": ".atmosphere.modis",
        "MSWEP": ".atmosphere.mswep",
}


def __getattr__(name: str) -> Any:
    item = _LAZY_IMPORTS.get(name)
    if isinstance(item,str):
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    elif isinstance(item, FunctionType):
        return item()

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

__all__: list[str] = [_LAZY_IMPORTS.keys()]
