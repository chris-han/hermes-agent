from __future__ import annotations

from typing import Any

from eos.artifact_hashing import canonical_json, sha256_hex


CANONICAL_HASH_PROFILE = "semantier-sensitive-governance-canonical-json-sha256-v1"


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_hex(canonical_json(value))
