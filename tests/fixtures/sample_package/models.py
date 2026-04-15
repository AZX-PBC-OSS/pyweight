# Internal import from sibling (exercises relative import resolution)
from .core import CoreClass

__all__ = ["SomeModel"]


class SomeModel:
    def __init__(self, core: CoreClass) -> None:
        self.core = core
