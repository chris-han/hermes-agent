"""Governed Gate-1/Gate-2 auto-pilot guard policy and audit result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SchemaGateEvaluation:
    gate: Literal["gate_1", "gate_2"]
    gate_outcome: Literal["accepted", "escalated"]
    operating_mode: Literal["assisted", "auto_pilot"]
    policy_ref: str
    rationale_ref: str
    evidence_refs: tuple[str, ...]
    fallback_reason: str | None


@dataclass(frozen=True)
class SchemaGatePolicy:
    """Pinned product policy; UI clients may select but never redefine it."""

    policy_ref: str
    minimum_gate_two_confidence: float
    allowed_gate_two_authority_states: tuple[str, ...]

    @classmethod
    def default(cls) -> "SchemaGatePolicy":
        return cls(
            policy_ref="schema-gate-policy:bounded-test-flight.v1",
            minimum_gate_two_confidence=0.95,
            allowed_gate_two_authority_states=("candidate",),
        )

    def _fallback(
        self,
        *,
        gate: Literal["gate_1", "gate_2"],
        reason: str,
        guard: str,
        evidence_refs: tuple[str, ...],
    ) -> SchemaGateEvaluation:
        return SchemaGateEvaluation(
            gate=gate,
            gate_outcome="escalated",
            operating_mode="assisted",
            policy_ref=self.policy_ref,
            rationale_ref=f"guard:{guard}",
            evidence_refs=evidence_refs,
            fallback_reason=reason,
        )

    def evaluate_gate_one(
        self,
        *,
        source_pins_valid: bool,
        semantic_slots_complete: bool,
        graph_layers_consistent: bool,
        unresolved_material_ambiguities: int,
        representation_lossless: bool,
        assurance_level: str,
        evidence_refs: tuple[str, ...],
    ) -> SchemaGateEvaluation:
        guards = (
            (source_pins_valid, "invalid_source_pins", "source_pins_valid"),
            (semantic_slots_complete, "incomplete_semantic_slots", "semantic_slots_complete"),
            (graph_layers_consistent, "inconsistent_graph_layers", "graph_layers_consistent"),
            (unresolved_material_ambiguities == 0, "unresolved_material_ambiguity", "no_material_ambiguity"),
            (representation_lossless, "lossy_representation", "representation_lossless"),
            (assurance_level == "bounded_test_flight", "unsupported_assurance_level", "assurance_level"),
            (bool(evidence_refs), "missing_decision_evidence", "decision_evidence"),
        )
        for passed, reason, guard in guards:
            if not passed:
                return self._fallback(
                    gate="gate_1",
                    reason=reason,
                    guard=guard,
                    evidence_refs=evidence_refs,
                )
        return SchemaGateEvaluation(
            gate="gate_1",
            gate_outcome="accepted",
            operating_mode="auto_pilot",
            policy_ref=self.policy_ref,
            rationale_ref="guard:all_gate_1_guards_passed",
            evidence_refs=evidence_refs,
            fallback_reason=None,
        )

    def evaluate_gate_two(
        self,
        *,
        candidate_structurally_valid: bool,
        evidence_complete: bool,
        schema_compatible: bool,
        authority_state: str,
        confidence: float,
        evidence_refs: tuple[str, ...],
    ) -> SchemaGateEvaluation:
        guards = (
            (candidate_structurally_valid, "invalid_candidate_structure", "candidate_structurally_valid"),
            (evidence_complete, "incomplete_evidence", "evidence_complete"),
            (schema_compatible, "schema_incompatible", "schema_compatible"),
            (authority_state in self.allowed_gate_two_authority_states, "authority_state_forbids_auto_grounding", "authority_state"),
            (confidence >= self.minimum_gate_two_confidence, "confidence_below_policy", "minimum_confidence"),
            (bool(evidence_refs), "missing_decision_evidence", "decision_evidence"),
        )
        for passed, reason, guard in guards:
            if not passed:
                return self._fallback(
                    gate="gate_2",
                    reason=reason,
                    guard=guard,
                    evidence_refs=evidence_refs,
                )
        return SchemaGateEvaluation(
            gate="gate_2",
            gate_outcome="accepted",
            operating_mode="auto_pilot",
            policy_ref=self.policy_ref,
            rationale_ref="guard:all_gate_2_guards_passed",
            evidence_refs=evidence_refs,
            fallback_reason=None,
        )


__all__ = ["SchemaGateEvaluation", "SchemaGatePolicy"]
