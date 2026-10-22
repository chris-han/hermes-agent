"""Derived, non-authoritative readiness and bounded-autonomy projections."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


EpistemicQualificationDisposition = Literal["AUTO_QUALIFIED", "REPAIRABLE", "UNRESOLVED"]
TransitionDisposition = Literal[
    "AUTO_TRANSITION", "AUTO_REPAIR", "AUTO_INFER", "RECOMMEND", "ASK", "BLOCK"
]


class EpistemicReadinessV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["epistemic_readiness.v1"] = "epistemic_readiness.v1"
    disposition: EpistemicQualificationDisposition
    structural_qualification: Literal["QUALIFIED", "UNRESOLVED", "BLOCKED"]
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    repair_candidates: tuple[str, ...] = ()


class KnowledgeUseOrchestrationResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["knowledge_use_orchestration_result.v1"] = "knowledge_use_orchestration_result.v1"
    readiness: EpistemicReadinessV1
    transition: "BoundedAutonomyTransitionV1"
    admission_receipt_ref: str | None = None
    projection_receipt_ref: str | None = None
    activation_receipt_ref: str | None = None
    activated: bool = False


class BoundedAutonomyTransitionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bounded_autonomy_transition.v1"] = "bounded_autonomy_transition.v1"
    disposition: TransitionDisposition
    best_next_action: str
    reason_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    user_interruption_required: bool = False


__all__ = [
    "EpistemicQualificationDisposition",
    "EpistemicReadinessV1",
    "KnowledgeUseOrchestrationResultV1",
    "TransitionDisposition",
    "BoundedAutonomyTransitionV1",
]
