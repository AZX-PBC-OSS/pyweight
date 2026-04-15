# Simulates a heavy third-party library (e.g. numpy/sklearn) for fixture purposes.
# Used by heavy.py to exercise the "expensive module-level import" analysis pattern.
import time as _time


class HeavyTransformer:
    """Pretend transformer that would be slow to import in a real library."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self._fitted = False

    def fit(self, data: list[list[float]]) -> "HeavyTransformer":
        _time.sleep(0)  # no-op stand-in for real computation
        self._fitted = True
        return self

    def transform(self, data: list[list[float]]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return [row[: self.n_components] for row in data]


class HeavyPipeline:
    """Pretend pipeline that chains transformers."""

    def __init__(self, steps: list[tuple[str, HeavyTransformer]]) -> None:
        self.steps = steps

    def run(self, data: list[list[float]]) -> list[list[float]]:
        result = data
        for _name, transformer in self.steps:
            transformer.fit(result)
            result = transformer.transform(result)
        return result
