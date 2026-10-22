from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for persisted artifacts."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    """Return SHA-256 over canonical JSON."""
    return sha256_hex(canonical_json(value))
