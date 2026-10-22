from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

from .models import EvidenceRecord, LearningCandidate, LearningProposal, LearningProposalProvider, DecisionStep, ItemType, canonical_hash
from .service import CognitiveContextService


PROPOSAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LearningTelemetryResult:
    status: str
    evidence_id: str | None = None
    candidate_id: str | None = None
    item_id: str | None = None
    error_code: str | None = None


class LearningTelemetryAdapter:
    """Synchronous, keyed learning process in the existing cognitive store."""

    def __init__(self, cognitive_service: CognitiveContextService) -> None:
        self.service = cognitive_service

    @staticmethod
    def learning_evidence_key(*, step_id: str, event_kind: str, outcome_ref: str) -> str:
        return canonical_hash({"step_id": step_id, "event_kind": event_kind, "outcome_ref": outcome_ref})

    @staticmethod
    def _proposal_from_json(payload: str, version: int | None) -> LearningProposal:
        if version != PROPOSAL_SCHEMA_VERSION:
            raise ValueError("unsupported_proposal_schema_version")
        data = json.loads(payload)
        return LearningProposal(
            proposed_type=ItemType(data["proposed_type"]), hypothesis=data["hypothesis"],
            confidence=data["confidence"], proposed_change=data["proposed_change"],
        )

    @staticmethod
    def _proposal_json(proposal: LearningProposal) -> str:
        return json.dumps({"proposed_type": proposal.proposed_type.value, "hypothesis": proposal.hypothesis,
                           "confidence": proposal.confidence, "proposed_change": proposal.proposed_change},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def process(self, *, step: DecisionStep, outcome: dict, learner: LearningProposalProvider,
                failure_injector: Callable[[str], None] | None = None) -> LearningTelemetryResult:
        key = self.learning_evidence_key(step_id=step.step_id, event_kind=step.event_kind, outcome_ref=step.outcome_ref)
        store = self.service.store
        process = store.reserve_learning_process(key=key, organization_id=step.organization_id,
                                                 workspace_id=step.workspace_id, step_id=step.step_id,
                                                 event_kind=step.event_kind, outcome_ref=step.outcome_ref)
        if process.stage == "closed":
            terminal = "closed" if process.terminal_status == "active" else (process.terminal_status or "failed")
            return LearningTelemetryResult(terminal, process.evidence_id,
                                           process.candidate_id, process.item_id, process.last_error_code)
        if failure_injector and process.stage == "started":
            failure_injector("started")

        evidence: EvidenceRecord
        if process.evidence_id:
            evidence = store.get_evidence(process.evidence_id, workspace_id=step.workspace_id, organization_id=step.organization_id)
        else:
            evidence = EvidenceRecord(
                f"evidence-{key}", step.workspace_id, step.organization_id, step.session_id,
                f"episode-{step.session_id}", step.turn_id or step.outcome_ref, step.event_kind,
                {"outcome": outcome, "decision_context_ref": step.decision_context_ref,
                 "trajectory_ref": step.trajectory_ref, "logical_step_id": step.logical_step_id,
                 "execution_attempt_ref": step.execution_attempt_ref,
                 "execution_attempt_hash": step.execution_attempt_hash,
                 "attempt_index": step.attempt_index, "lane": step.lane,
                 "action_refs": list(step.action_refs), "tool_refs": list(step.tool_refs),
                 "feedback": outcome.get("feedback")},
            )
            evidence = store.save_or_get_evidence(evidence)
            process = store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                                    stage="evidence_captured", evidence_id=evidence.id)
            if failure_injector:
                failure_injector("evidence_captured")

        if process.proposal_json:
            try:
                proposal = self._proposal_from_json(process.proposal_json, process.proposal_schema_version)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                               stage=process.stage, last_error_code="unsupported_proposal_schema_version")
                return LearningTelemetryResult("failed", evidence.id, process.candidate_id, None,
                                               "unsupported_proposal_schema_version")
        else:
            try:
                proposal = learner.propose(evidence=evidence, event_kind=step.event_kind)
                if not isinstance(proposal, LearningProposal):
                    raise TypeError("invalid_learning_proposal")
                proposal_json = self._proposal_json(proposal)
            except Exception as exc:
                store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                               stage=process.stage, last_error_code=type(exc).__name__)
                return LearningTelemetryResult("failed", evidence.id, None, None, type(exc).__name__)
            process = store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                                    stage="proposal_captured", evidence_id=evidence.id,
                                                    proposal_schema_version=PROPOSAL_SCHEMA_VERSION, proposal_json=proposal_json)
            if failure_injector:
                failure_injector("proposal_captured")

        candidate: LearningCandidate
        if process.candidate_id:
            candidate = store.get_candidate(process.candidate_id, workspace_id=step.workspace_id, organization_id=step.organization_id)
        else:
            candidate = LearningCandidate(f"candidate-{key}", proposal.proposed_type, [evidence.id], proposal.hypothesis,
                                          proposal.confidence, proposal.proposed_change, step.workspace_id,
                                          step.organization_id)
            candidate = store.save_or_get_candidate(candidate)
            process = store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                                    stage="candidate_created", evidence_id=evidence.id, candidate_id=candidate.id)
            if failure_injector:
                failure_injector("candidate_created")

        if process.stage == "closed":
            return LearningTelemetryResult(process.terminal_status or "failed", evidence.id, candidate.id, process.item_id,
                                           process.last_error_code)
        item = self.service.close(candidate, item_id=f"item-{key}")
        refreshed = store.get_candidate(candidate.id, workspace_id=step.workspace_id, organization_id=step.organization_id)
        status = {"active": "closed", "pending": "pending", "rejected": "rejected"}.get(refreshed.status.value, "rejected")
        process = store.update_learning_process(key, organization_id=step.organization_id, workspace_id=step.workspace_id,
                                                stage="closed", evidence_id=evidence.id, candidate_id=candidate.id,
                                                terminal_status=("active" if status == "closed" else status),
                                                item_id=item.id if item else None)
        if failure_injector:
            failure_injector("closed")
        return LearningTelemetryResult(status, evidence.id, candidate.id, item.id if item else None)
