"""Binding contract for active investment mandate/thesis/ontology resolution."""
from __future__ import annotations

import json
from typing import Any, Protocol

from contracts.investment_opportunity import RuntimeContextDTO


class InvestmentGovernedContextResolver(Protocol):
    def resolve_active_context(self, context: RuntimeContextDTO) -> dict[str, Any]: ...


class MissingInvestmentGovernedContextResolver:
    def resolve_active_context(self, context: RuntimeContextDTO) -> dict[str, Any]:
        raise RuntimeError("GOVERNED_INVESTMENT_CONTEXT_BINDING_UNAVAILABLE")


class GovernanceStoreInvestmentContextResolver:
    """Resolve active investment bindings from the authoritative policy store."""

    def __init__(self, db_path=None):
        from eos.db import init_db
        from storage.governance_store import GovernanceStore
        init_db(db_path)
        self._store = GovernanceStore(db_path=db_path)

    def resolve_active_context(self, context: RuntimeContextDTO) -> dict[str, Any]:
        bindings: dict[str, dict[str, Any]] = {}
        for category, key in (
            ("policy_bundle", "mandate_ref"),
            ("policy_bundle", "thesis_ref"),
            ("ontology_entry", "ontology_version"),
        ):
            for row in self._store.list_candidates(category=category, status="ACTIVE"):
                try:
                    content = json.loads(row.get("content_yaml") or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if content.get("organization_id") != context.organization_id:
                    continue
                bindings.setdefault(key, row.get("candidate_id") or row.get("name"))
                break
        if not {"mandate_ref", "thesis_ref", "ontology_version"}.issubset(bindings):
            raise RuntimeError("GOVERNED_INVESTMENT_CONTEXT_INCOMPLETE")
        return bindings
