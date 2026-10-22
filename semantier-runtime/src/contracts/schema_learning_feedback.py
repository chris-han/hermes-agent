"""Typed evidence for routing Grounding findings into schema learning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from eos.artifact_hashing import content_hash


@dataclass(frozen=True)
class SchemaLearningFeedback:
    feedback_id: str
    source_binding: dict[str, Any]
    schema_decision_ref: str
    candidate_ref: str
    extraction_run_ref: str | None
    grounding_ref: str | None
    feedback_scope: str
    produced_by: str
    operating_mode: str
    policy_ref: str | None
    rationale_ref: str | None
    decision_evidence_refs: tuple[str, ...]
    fallback_reason: str | None
    operation_ref: str | None
    before_state_ref: str | None
    after_state_ref: str | None
    attribution: dict[str, bool]
    findings: tuple[dict[str, Any], ...]
    proposed_candidate_delta_ref: str | None
    created_at: str
    created_by: str
    schema_version: str = "schema_learning_feedback.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "schema_learning_feedback.v1":
            raise ValueError("schema_version must be schema_learning_feedback.v1")
        if not self.feedback_id.startswith("schema-feedback-"):
            raise ValueError("feedback_id is invalid")
        if self.feedback_scope not in {
            "instance_local",
            "schema_understanding",
            "ontology_or_higher_layer",
        }:
            raise ValueError("feedback_scope is invalid")
        if self.produced_by not in {
            "user_direct_edit",
            "user_chat_clarification",
            "agent_synthesis",
            "auto_pilot_feedback",
        }:
            raise ValueError("produced_by is invalid")
        if self.operating_mode not in {"assisted", "auto_pilot"}:
            raise ValueError("operating_mode must be assisted or auto_pilot")
        if self.operating_mode == "auto_pilot":
            if self.produced_by != "auto_pilot_feedback":
                raise ValueError("auto_pilot feedback must be produced by auto_pilot_feedback")
            if not self.policy_ref or not self.rationale_ref or not self.decision_evidence_refs:
                raise ValueError("auto_pilot feedback requires policy, rationale, and evidence pins")
        if self.produced_by == "user_direct_edit":
            if not all(
                (self.grounding_ref, self.operation_ref, self.before_state_ref, self.after_state_ref)
            ):
                raise ValueError("user_direct_edit requires grounding operation and before/after refs")
            operation = str(self.operation_ref).split(":", 1)[0]
            if operation not in {"split", "node_split", "merge", "node_merge", "edit_role", "accept", "reject"}:
                raise ValueError("user_direct_edit operation is not an existing Grounding primitive")
        if not self.source_binding:
            raise ValueError("source_binding is required")
        if not self.schema_decision_ref or not self.candidate_ref:
            raise ValueError("schema_decision_ref and candidate_ref are required")
        if not self.findings:
            raise ValueError("findings must not be empty")
        expected_attribution = {
            "source_structure",
            "ontology_induction",
            "synthesis",
            "extraction_plan",
            "execution",
        }
        if set(self.attribution) != expected_attribution or not all(
            isinstance(value, bool) for value in self.attribution.values()
        ):
            raise ValueError("attribution must contain the five typed boolean causes")
        if (
            self.feedback_scope == "instance_local"
            and self.proposed_candidate_delta_ref is not None
        ):
            raise ValueError("instance_local feedback cannot propose a candidate delta")
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("created_at must be timezone-aware UTC ISO-8601") from exc
        if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
            raise ValueError("created_at must be timezone-aware UTC ISO-8601")
        expected = f"schema-feedback-{content_hash(self._hash_payload())[:24]}"
        if self.feedback_id != expected:
            raise ValueError("feedback_id does not match feedback content")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "source_binding": dict(self.source_binding),
            "schema_decision_ref": self.schema_decision_ref,
            "candidate_ref": self.candidate_ref,
            "extraction_run_ref": self.extraction_run_ref,
            "grounding_ref": self.grounding_ref,
            "feedback_scope": self.feedback_scope,
            "produced_by": self.produced_by,
            "operating_mode": self.operating_mode,
            "policy_ref": self.policy_ref,
            "rationale_ref": self.rationale_ref,
            "decision_evidence_refs": list(self.decision_evidence_refs),
            "fallback_reason": self.fallback_reason,
            "operation_ref": self.operation_ref,
            "before_state_ref": self.before_state_ref,
            "after_state_ref": self.after_state_ref,
            "attribution": dict(self.attribution),
            "findings": [dict(item) for item in self.findings],
            "proposed_candidate_delta_ref": self.proposed_candidate_delta_ref,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("feedback_id")
        return payload

    @classmethod
    def create(cls, content: dict[str, Any]) -> "SchemaLearningFeedback":
        payload = dict(content)
        payload["schema_version"] = "schema_learning_feedback.v1"
        payload.pop("feedback_id", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {**payload, "feedback_id": f"schema-feedback-{digest[:24]}"}
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchemaLearningFeedback":
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            feedback_id=str(payload.get("feedback_id") or ""),
            source_binding=dict(payload.get("source_binding") or {}),
            schema_decision_ref=str(payload.get("schema_decision_ref") or ""),
            candidate_ref=str(payload.get("candidate_ref") or ""),
            extraction_run_ref=(
                str(payload["extraction_run_ref"])
                if payload.get("extraction_run_ref") is not None
                else None
            ),
            grounding_ref=(
                str(payload["grounding_ref"])
                if payload.get("grounding_ref") is not None
                else None
            ),
            feedback_scope=str(payload.get("feedback_scope") or ""),
            produced_by=str(payload.get("produced_by") or ""),
            operating_mode=str(payload.get("operating_mode") or ""),
            policy_ref=(str(payload["policy_ref"]) if payload.get("policy_ref") is not None else None),
            rationale_ref=(str(payload["rationale_ref"]) if payload.get("rationale_ref") is not None else None),
            decision_evidence_refs=tuple(str(ref) for ref in (payload.get("decision_evidence_refs") or [])),
            fallback_reason=(str(payload["fallback_reason"]) if payload.get("fallback_reason") is not None else None),
            operation_ref=(
                str(payload["operation_ref"])
                if payload.get("operation_ref") is not None
                else None
            ),
            before_state_ref=(
                str(payload["before_state_ref"])
                if payload.get("before_state_ref") is not None
                else None
            ),
            after_state_ref=(
                str(payload["after_state_ref"])
                if payload.get("after_state_ref") is not None
                else None
            ),
            attribution=dict(payload.get("attribution") or {}),
            findings=tuple(dict(item) for item in (payload.get("findings") or [])),
            proposed_candidate_delta_ref=(
                str(payload["proposed_candidate_delta_ref"])
                if payload.get("proposed_candidate_delta_ref") is not None
                else None
            ),
            created_at=str(payload.get("created_at") or ""),
            created_by=str(payload.get("created_by") or ""),
        )


__all__ = ["SchemaLearningFeedback"]
