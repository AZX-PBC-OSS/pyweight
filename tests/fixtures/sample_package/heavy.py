# Heavy module-level import (exercises cost estimation + hotspot detection)
# _mock_heavy_lib stands in for a real expensive third-party library (e.g. sklearn).
import json
import re

from ._mock_heavy_lib import HeavyPipeline, HeavyTransformer
from .hub import HubClass  # noqa: F401 — graph edge: heavy -> hub (fan-in fixture)


class HeavyProcessor:
    def __init__(self) -> None:
        self._pipeline = HeavyPipeline([("reduce", HeavyTransformer(n_components=2))])

    def process(self, data: str) -> dict[str, object]:
        pattern = re.compile(r"\s+")
        cleaned = pattern.sub(" ", data).strip()
        return json.loads(cleaned)
