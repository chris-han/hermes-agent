from __future__ import annotations

from typing import Any


def register(ctx: Any) -> None:
    """Knowledge-only plugin; no runtime tools are registered."""
    return None
