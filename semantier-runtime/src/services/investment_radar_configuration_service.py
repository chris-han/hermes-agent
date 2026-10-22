"""Typed governance bridge for reusable radar universes and profiles."""
from __future__ import annotations

import json
from typing import Any, Protocol

from contracts.investment_opportunity import RadarScoringProfileCandidateDTO, RadarUniverseCandidateDTO, RadarConfigurationStateDTO, RuntimeContextDTO
from eos.artifact_hashing import content_hash


class InvestmentRadarConfigurationService(Protocol):
    def propose_universe(self, payload: RadarUniverseCandidateDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def propose_scoring_profile(self, payload: RadarScoringProfileCandidateDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def request_review(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def approve(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def activate(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def supersede(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def resolve_active_universe(self, name: str, context: RuntimeContextDTO) -> dict[str, Any] | None: ...
    def resolve_active_scoring_profile(self, name: str, context: RuntimeContextDTO) -> dict[str, Any] | None: ...


class StoreBackedInvestmentRadarConfigurationService:
    def __init__(self, db_path=None):
        from eos.db import init_db
        from storage.governance_store import GovernanceStore
        init_db(db_path)
        self._store = GovernanceStore(db_path=db_path)

    def _propose(self, payload, category, context):
        data = dict(payload.__dict__)
        data["organization_id"] = context.organization_id
        serialized = json.dumps(data, sort_keys=True, default=list)
        digest = payload.content_hash or "sha256:" + content_hash(data)
        return self._store.propose(name=payload.name, category=category, content_yaml=serialized, content_hash=digest, proposed_by=context.actor_id, semantic_tier=payload.semantic_tier, scoped_owner=payload.owner_ref, rollback_window_hours=24, rollback_strategy="supersede_with_new_version")

    def propose_universe(self, payload, context): return self._propose(payload, "investment_radar_universe", context)
    def propose_scoring_profile(self, payload, context): return self._propose(payload, "investment_radar_scoring_profile", context)
    def request_review(self, candidate_id, context):
        row = self._store.get_by_id(candidate_id)
        if row is None: raise KeyError("RADAR_CONFIGURATION_NOT_FOUND")
        if row.get("status") == "PENDING" and not row.get("replay_result"):
            return self._store.record_replay_result(candidate_id, regressions=[], tested_claims=0)
        return row
    def approve(self, candidate_id, context):
        return self._store.record_approval(candidate_id, approved_by=context.actor_id, actor_roles=(context.actor_role,) if context.actor_role else None)
    def activate(self, candidate_id, context):
        return self._store.activate(candidate_id, actor_roles=(context.actor_role,) if context.actor_role else None)
    def supersede(self, candidate_id, context): return self._store.record_candidate_event(candidate_id=candidate_id, event_type="SUPERSEDED", actor=context.actor_id, event_payload={"reason": "superseded"}) or {"candidate_id": candidate_id}
    def _resolve(self, name, category):
        row = self._store.get_by_name(name)
        return row if row and row.get("category") == category and row.get("status") == "ACTIVE" else None
    def resolve_active_universe(self, name, context): return self._resolve(name, "investment_radar_universe")
    def resolve_active_scoring_profile(self, name, context): return self._resolve(name, "investment_radar_scoring_profile")
