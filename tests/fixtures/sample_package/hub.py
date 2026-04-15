# Hub module — high fan-in and fan-out (exercises hub-like dependency detection)
# Imports from core, models, utils.helpers, conditional (high fan-out)
# Imported by heavy, side_effects, deferred_candidate (high fan-in)
from .conditional import load_config
from .core import (  # noqa: F401 — helper_fn imported for graph edge; CoreClass used below
    CoreClass,
    helper_fn,
)
from .models import SomeModel
from .utils.helpers import format_name


class HubClass:
    def __init__(self) -> None:
        self.core = CoreClass("hub")
        self.model = SomeModel(self.core)

    def run(self) -> str:
        name = format_name(self.core.name)
        config = load_config("config.toml")
        return f"{name}: {config}"
