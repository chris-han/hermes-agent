"""Holographic Memory Provider boundary contract (v6).

Provider returns governed projection context hints only.
It must not return mutable financial truth fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


_FORBIDDEN_TOP_LEVEL_KEYS = {
    "account_balances",
    "balances",
    "revenue_totals",
    "cash_position",
    "tax_payable_totals",
    "profit_numbers",
    "ledger_totals",
}


@dataclass
class ProjectionContextSnapshot:
    source_refs: List[Dict[str, Any]]
    selected_context: Dict[str, Any]
    excluded_context_refs: List[Dict[str, Any]]
    selection_reason: str
    provider_version: str


class HolographicMemoryProvider:
    """Retrieval interface over KGL and approved memory."""

    def retrieve_projection_context(
        self,
        *,
        org_id: str,
        rea_event: dict,
        authority_domain: str,
        projection_bundle_id: str,
        knowledge_version: str,
        tag_ontology_version: str,
        constraint_bundle_version: str,
        query_intent: str,
    ) -> ProjectionContextSnapshot:
        raise NotImplementedError


class InMemoryHolographicProvider(HolographicMemoryProvider):
    """Minimal deterministic provider implementation for runtime/tests."""

    def __init__(self, provider_version: str = "holo_v1") -> None:
        self.provider_version = provider_version

    def retrieve_projection_context(
        self,
        *,
        org_id: str,
        rea_event: dict,
        authority_domain: str,
        projection_bundle_id: str,
        knowledge_version: str,
        tag_ontology_version: str,
        constraint_bundle_version: str,
        query_intent: str,
    ) -> ProjectionContextSnapshot:
        context = {
            "org_id": org_id,
            "authority_domain": authority_domain,
            "projection_bundle_id": projection_bundle_id,
            "knowledge_version": knowledge_version,
            "query_intent": query_intent,
            "resource_type": ((rea_event.get("resources") or [{}])[0] or {}).get("type"),
        }
        sanitized = sanitize_projection_context(context)
        return ProjectionContextSnapshot(
            source_refs=[],
            selected_context=sanitized,
            excluded_context_refs=[],
            selection_reason="deterministic default context",
            provider_version=self.provider_version,
        )


def sanitize_projection_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Reject forbidden financial truth keys from provider output."""
    forbidden = _FORBIDDEN_TOP_LEVEL_KEYS.intersection(context.keys())
    if forbidden:
        raise ValueError(f"forbidden_projection_context_fields:{','.join(sorted(forbidden))}")
    return context
