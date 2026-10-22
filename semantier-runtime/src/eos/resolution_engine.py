"""Resolution engine boundary module (v6 Slice 5)."""

from __future__ import annotations

from typing import Any, Dict, List

from eos.resolution_service import resolve as _resolve


def run_resolution(
    *,
    claims: List[Dict[str, Any]],
    constraint_results: List[Dict[str, Any]],
    precedence_graph: Dict[str, Any],
    resolution_rules: Dict[str, Any],
    scope: str,
    pins: Dict[str, Any],
) -> Dict[str, Any]:
    return _resolve(
        claims=claims,
        constraint_results=constraint_results,
        precedence_graph=precedence_graph,
        resolution_rules=resolution_rules,
        scope=scope,
        pins=pins,
    )
