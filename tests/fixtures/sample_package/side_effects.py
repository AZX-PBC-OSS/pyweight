# Module with side effects (exercises purity=side_effects classification)
import sys

from .hub import HubClass  # noqa: F401 — graph edge: side_effects -> hub (fan-in fixture)

_registry: list[str] = []


def register(name: str) -> None:
    _registry.append(name)


# Module-level side effect — bare function call
register("side_effects")
print("side_effects module loaded", file=sys.stderr)
