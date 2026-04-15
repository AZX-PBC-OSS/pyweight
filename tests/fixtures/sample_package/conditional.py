# Conditional import pattern (exercises is_conditional detection)
try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore[assignment]


def load_config(path: str) -> dict[str, object] | None:
    if tomllib is None:
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)
