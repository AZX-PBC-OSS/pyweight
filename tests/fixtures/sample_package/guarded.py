# TYPE_CHECKING guard (exercises is_type_checking detection)
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .heavy import HeavyProcessor


def create_processor() -> HeavyProcessor:
    from .heavy import HeavyProcessor

    return HeavyProcessor()
