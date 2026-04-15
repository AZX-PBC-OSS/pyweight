# Top-level import used in only one function (exercises deferral candidate detection)
import json

from .hub import HubClass  # noqa: F401 — graph edge: deferred_candidate -> hub (fan-in fixture)

CONSTANT = 42


def load_data(path: str) -> dict[str, object]:
    with open(path) as f:
        return json.load(f)


def get_constant() -> int:
    return CONSTANT
