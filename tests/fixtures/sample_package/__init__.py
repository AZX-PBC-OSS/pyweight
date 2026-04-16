# Barrel export — re-exports from submodules (exercises barrel detection)
from .core import *  # noqa: F403
from .models import SomeModel

__all__ = ["CoreClass", "SomeModel", "helper_fn"]  # noqa: F405
