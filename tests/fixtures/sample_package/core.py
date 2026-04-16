# Pure module — stdlib imports only (exercises purity=pure classification)
import os
from pathlib import Path


class CoreClass:
    def __init__(self, name: str) -> None:
        self.name = name


def helper_fn(p: Path) -> str:
    return os.fspath(p)


__all__ = ["CoreClass", "helper_fn"]
