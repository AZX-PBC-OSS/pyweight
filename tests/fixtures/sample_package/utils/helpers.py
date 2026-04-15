# Relative import (exercises level>0 resolution)
from .. import core


def format_name(name: str) -> str:
    obj = core.CoreClass(name)
    return obj.name.upper()
