"""Authority-bearing candidate service for the GitHub opportunity radar."""
from __future__ import annotations

from typing import Any, Protocol
from eos.artifact_hashing import content_hash

from contracts.investment_opportunity import (
    OpportunityCandidateActionDTO, OpportunityCandidateSubmitDTO,
    OpportunityReplayEnvelopeDTO, OpportunityCandidateStateDTO, RuntimeContextDTO,
)


class InvestmentOpportunityCandidateService(Protocol):
    def submit_candidate_idempotent(self, payload: OpportunityCandidateSubmitDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def validate_candidate(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def request_review(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def record_review_outcome(self, action: OpportunityCandidateActionDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def propose_follow_up_action(self, action: OpportunityCandidateActionDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def get_candidate_state(self, candidate_id: str, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def list_candidates(self, context: RuntimeContextDTO, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    def build_replay_envelope(self, candidate_id: str, context: RuntimeContextDTO) -> OpportunityReplayEnvelopeDTO: ...
    def record_operator_outcome(self, action: OpportunityCandidateActionDTO, context: RuntimeContextDTO) -> dict[str, Any]: ...
    def list_operator_outcomes(self, candidate_id: str, context: RuntimeContextDTO) -> list[dict[str, Any]]: ...


class StoreBackedInvestmentOpportunityCandidateService:
    def __init__(self, governed_context_resolver=None):
        self._governed_context_resolver = governed_context_resolver

    def submit_candidate_idempotent(self, payload, context):
        from eos import t6_materialization_store as store
        if not payload.organization_id or payload.organization_id != context.organization_id:
            raise ValueError("CANDIDATE_ORGANIZATION_CONTEXT_MISMATCH")
        if not payload.observation_refs or not payload.scorecard_ref:
            raise ValueError("CANDIDATE_PROVENANCE_REQUIRED")
        required_justification = {"observation_refs", "observation_hashes", "feature_spec_version", "scoring_profile_version"}
        if not required_justification.issubset(payload.justification):
            raise ValueError("CANDIDATE_JUSTIFICATION_INCOMPLETE")
        if self._governed_context_resolver is not None:
            governed = (
                self._governed_context_resolver(context)
                if callable(self._governed_context_resolver)
                else self._governed_context_resolver.resolve_active_context(context)
            )
            if not governed.get("ontology_version") or not governed.get("mandate_ref"):
                raise ValueError("GOVERNED_INVESTMENT_CONTEXT_INCOMPLETE")
        else:
            # A candidate may be captured before mandate admission, but it is
            # explicitly marked unresolved and can never qualify by itself.
            payload.justification.setdefault("governed_context_status", "UNRESOLVED")
        candidate = payload.hashed()
        identity = ":".join([
            str(payload.target.get("github_repository_refs", [])),
            str(payload.justification.get("window", "90d")),
            str(payload.justification.get("scoring_profile_version", "")),
            candidate.content_hash,
        ])
        existing = store.get_candidate_by_source_identity(org_id=context.organization_id, source_channel="vc_github_opportunity_radar", source_message_id=identity)
        if existing is not None:
            return {"result": "DUPLICATE", "candidate_id": existing["candidate_id"], "candidate": existing, "reason_code": "DUPLICATE_CANDIDATE"}
        candidate_payload = dict(candidate.__dict__)
        candidate_payload.update({
            "observation_refs": list(candidate.observation_refs),
            "observation_hashes": list(candidate.justification.get("observation_hashes", [])),
            "feature_spec_version": candidate.justification.get("feature_spec_version", ""),
            "feature_snapshot_hash": candidate.justification.get("feature_snapshot_hash", ""),
            "scoring_profile_version": candidate.justification.get("scoring_profile_version", ""),
            "scoring_profile_hash": candidate.justification.get("scoring_profile_hash", ""),
            "entity_resolution_version": candidate.justification.get("entity_resolution_version", "vc_github_entity_resolution_v1"),
        })
        submitted = store.submit_candidate(
            org_id=context.organization_id, workspace_id=context.workspace_id,
            source_channel="vc_github_opportunity_radar", source_message_id=identity,
            source_session_id=None, candidate_json={"contract_version": candidate.contract_version, "payload": candidate_payload},
            candidate_hash=candidate.content_hash, actor_type="plugin", actor_id=context.actor_id,
        )
        candidate_id = submitted.get("candidate_id")
        if candidate_id:
            submitted["replay"] = self._persist_replay_binding(
                candidate_id=candidate_id,
                context=context,
                observation_refs=candidate_payload["observation_refs"],
                source_hashes=candidate_payload["observation_hashes"],
                candidate_payload=candidate_payload,
            )
            submitted["notification"] = store.record_radar_notification(
                candidate_id=candidate_id, organization_id=context.organization_id,
                workspace_id=context.workspace_id, notification_type="candidate_admitted",
                payload={"candidate_hash": candidate.content_hash, "source_channel": "vc_github_opportunity_radar"},
            )
        return submitted

    def validate_candidate(self, candidate_id, context):
        from eos import t6_materialization_store as store
        return store.validate_candidate(candidate_id=candidate_id, allow=True, actor_type="core", actor_id=context.actor_id)

    def request_review(self, candidate_id, context):
        from eos import t6_materialization_store as store
        return store.queue_for_approval(candidate_id=candidate_id, actor_type="core", actor_id=context.actor_id)

    def record_review_outcome(self, action, context):
        from eos import t6_materialization_store as store
        self._require_candidate_scope(action.candidate_id, context)
        if action.action_scope == "OPPORTUNITY_QUALIFICATION":
            self._resolve_governed_context(context)
            return store.human_approve_candidate(candidate_id=action.candidate_id, actor_id=context.actor_id)
        if action.action_scope == "OPPORTUNITY_REJECTION":
            return store.human_reject_candidate(candidate_id=action.candidate_id, actor_id=context.actor_id)
        if action.action_scope == "OPPORTUNITY_DEFERRAL":
            return store.defer_candidate(candidate_id=action.candidate_id, actor_id=context.actor_id, reason=action.reason, trigger=action.trigger)
        raise ValueError("ACTION_SCOPE_NOT_REVIEW_OUTCOME")

    def propose_follow_up_action(self, action, context):
        if action.action_scope in {"OPPORTUNITY_QUALIFICATION", "OPPORTUNITY_REJECTION", "OPPORTUNITY_DEFERRAL"}:
            raise ValueError("REVIEW_OUTCOME_REQUIRED_ROUTE")
        proposal = {"status": "PROPOSED", "candidate_id": action.candidate_id, "action_scope": action.action_scope, "reason": action.reason}
        try:
            self._require_candidate_scope(action.candidate_id, context)
            state = self.get_candidate_state(action.candidate_id, context)
        except KeyError:
            # Research may be proposed before the candidate is admitted by a
            # caller that is only carrying an external candidate reference.
            # Core cannot bind replay evidence until the candidate exists.
            return proposal
        candidate = state["candidate"]
        payload = candidate.get("candidate_json") or {}
        inner = payload.get("payload", payload) if isinstance(payload, dict) else {}
        proposal["replay"] = self._persist_replay_binding(
            candidate_id=action.candidate_id,
            context=context,
            observation_refs=inner.get("observation_refs", []),
            source_hashes=inner.get("observation_hashes", []),
            candidate_payload=inner,
            action=action,
        )
        if action.action_scope in {"OPPORTUNITY_ENRICHMENT_REQUEST", "OPPORTUNITY_PRELIMINARY_DILIGENCE_REQUEST"}:
            proposal["evidence_candidates"] = [
                {
                    "status": "EVIDENCE_CANDIDATE",
                    "source_ref": ref,
                    "claim_status": "UNVERIFIED",
                    "institutional_finding": False,
                }
                for ref in inner.get("observation_refs", [])
            ]
        return proposal

    def _resolve_governed_context(self, context):
        from services.investment_governed_context_service import GovernanceStoreInvestmentContextResolver
        resolver = self._governed_context_resolver or GovernanceStoreInvestmentContextResolver()
        return resolver(context) if callable(resolver) else resolver.resolve_active_context(context)

    @staticmethod
    def _require_candidate_scope(candidate_id, context):
        from eos import t6_materialization_store as store
        candidate = store.get_candidate(candidate_id)
        if candidate is None or candidate.get("org_id") != context.organization_id:
            raise KeyError("OPPORTUNITY_CANDIDATE_NOT_FOUND")

    def get_candidate_state(self, candidate_id, context):
        from eos import t6_materialization_store as store
        candidate = store.get_candidate(candidate_id)
        if candidate is None or candidate.get("org_id") != context.organization_id:
            raise KeyError("OPPORTUNITY_CANDIDATE_NOT_FOUND")
        return {"candidate": candidate, "transitions": store.list_candidate_transitions(candidate_id)}

    def list_candidates(self, context, filters=None):
        from eos import t6_materialization_store as store
        return store.list_candidates_for_org(org_id=context.organization_id)

    def build_replay_envelope(self, candidate_id, context):
        state = self.get_candidate_state(candidate_id, context)
        payload = state["candidate"].get("candidate_json") or {}
        inner = payload.get("payload", payload) if isinstance(payload, dict) else {}
        replay = self._persist_replay_binding(candidate_id=candidate_id, context=context, observation_refs=inner.get("observation_refs", ()), source_hashes=inner.get("observation_hashes", ()), candidate_payload=inner)
        return OpportunityReplayEnvelopeDTO(candidate_id=candidate_id, observation_refs=tuple(inner.get("observation_refs", ())), observation_hashes=tuple(inner.get("observation_hashes", ())), feature_spec_version=inner.get("feature_spec_version", "unknown"), feature_snapshot_hash=inner.get("feature_snapshot_hash", "unknown"), scoring_profile_version=inner.get("scoring_profile_version", "unknown"), scoring_profile_hash=inner.get("scoring_profile_hash", "unknown"), entity_resolution_version=inner.get("entity_resolution_version", "vc_github_entity_resolution_v1"), replay_binding_ref=replay["replay_binding_id"])

    def record_operator_outcome(self, action, context):
        self._require_candidate_scope(action.candidate_id, context)
        outcome_type = str(action.trigger.get("outcome_type") or "")
        from eos import t6_materialization_store as store
        return store.record_radar_operator_outcome(
            candidate_id=action.candidate_id, organization_id=context.organization_id,
            workspace_id=context.workspace_id, outcome_type=outcome_type,
            reason=action.reason, evidence_refs=list(action.evidence_refs),
            case_ref=action.trigger.get("case_ref"), actor_id=context.actor_id,
        )

    def list_operator_outcomes(self, candidate_id, context):
        self._require_candidate_scope(candidate_id, context)
        from eos import t6_materialization_store as store
        return store.list_radar_operator_outcomes(candidate_id=candidate_id, organization_id=context.organization_id)

    def _persist_replay_binding(self, *, candidate_id, context, observation_refs, source_hashes, candidate_payload, action=None):
        from eos.generic_replay_envelope_store import create_typed_replay_binding, INVESTMENT_RADAR_CANDIDATE_TYPE
        action_suffix = f":action:{content_hash(action.__dict__)[:24]}" if action is not None else ""
        binding_id = f"investment-radar:{candidate_id}{action_suffix}"
        payload = {
            "candidate_id": candidate_id,
            "organization_id": context.organization_id,
            "workspace_id": context.workspace_id,
            "observation_refs": list(observation_refs),
            "observation_hashes": list(source_hashes),
            "feature_snapshot_hash": candidate_payload.get("feature_snapshot_hash", ""),
            "feature_spec_version": candidate_payload.get("feature_spec_version", ""),
            "scoring_profile_version": candidate_payload.get("scoring_profile_version", ""),
            "scoring_profile_hash": candidate_payload.get("scoring_profile_hash", ""),
            "entity_resolution_version": candidate_payload.get("entity_resolution_version", ""),
            "action": action.__dict__ if action is not None else None,
        }
        return create_typed_replay_binding(
            replay_binding_id=binding_id,
            binding_type=INVESTMENT_RADAR_CANDIDATE_TYPE,
            binding_schema_version="investment_radar_replay.v1",
            subject_type="opportunity_candidate",
            subject_id=candidate_id,
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            payload=payload,
        )
