"""Validated repo-owned copy for GraphIntent compiler evidence."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


_ASSET = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "graph_intent_compiler_v3.json"
)
_FIELDS = {
    "candidate_statement",
    "reasoning_summary",
    "extraction_plan_rationale",
}


@lru_cache(maxsize=1)
def graph_intent_compiler_copy() -> dict[str, str]:
    payload = json.loads(_ASSET.read_text(encoding="utf-8"))
    if set(payload) != _FIELDS or not all(
        isinstance(value, str) and value.strip() for value in payload.values()
    ):
        raise ValueError("GraphIntent compiler prompt asset is invalid")
    return dict(payload)


__all__ = ["graph_intent_compiler_copy"]
