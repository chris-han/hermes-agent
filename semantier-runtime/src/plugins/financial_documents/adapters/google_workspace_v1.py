from __future__ import annotations

from typing import Any

from .base import extract_generic


def extract(canonical: dict[str, Any], *, preferred_currency: str | None = None) -> dict[str, Any]:
    return extract_generic(canonical, preferred_currency=preferred_currency, adapter_name="google_workspace_v1")
