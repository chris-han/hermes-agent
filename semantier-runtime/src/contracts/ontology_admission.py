"""Ontology certification, admission, release, and execution-epoch contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
AssuranceProfile = Literal[
    "A0_EXPLORATORY",
    "A1_INTERNAL_POLICY_MVL",
    "A2_GOVERNED_ENTERPRISE",
    "A3_REGULATORY_HIGH_ASSURANCE",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class AdmitOntologyCommandV1(_Contract):
    schema_version: Literal["admit_ontology_command.v1"] = "admit_ontology_command.v1"
    build_ref: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)


class CertifiedSemanticExecutionPlanV1(_Contract):
    schema_version: Literal["certified_semantic_execution_plan.v1"] = (
        "certified_semantic_execution_plan.v1"
    )
    execution_plan_ref: str = Field(min_length=1)
    execution_plan_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    assurance_profile: AssuranceProfile
    assurance_policy_ref: str = Field(min_length=1)
    assurance_policy_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_normative_ir_ref: str = Field(min_length=1)
    canonical_normative_ir_hash: str = Field(pattern=SHA256_PATTERN)
    validation_contract_ref: str = Field(min_length=1)
    validation_contract_hash: str = Field(pattern=SHA256_PATTERN)
    quality_profile: ArtifactPinV1
    quality_evaluation: ArtifactPinV1
    compiled_rule_plan_ref: str = Field(min_length=1)
    compiled_rule_plan_hash: str = Field(pattern=SHA256_PATTERN)
    runtime_program_ref: str = Field(min_length=1)
    runtime_program_hash: str = Field(pattern=SHA256_PATTERN)
    projection_compiler_ref: str = Field(min_length=1)
    projection_compiler_hash: str = Field(pattern=SHA256_PATTERN)
    environment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    justification_projection: Literal["runtime_justification.v1"]
    certified_at: datetime


class OntologyAdmissionDecisionV1(_Contract):
    schema_version: Literal["ontology_admission_decision.v1"] = (
        "ontology_admission_decision.v1"
    )
    decision_ref: str = Field(min_length=1)
    decision_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=SHA256_PATTERN)
    assurance_profile: AssuranceProfile
    assurance_policy_ref: str = Field(min_length=1)
    assurance_policy_hash: str = Field(pattern=SHA256_PATTERN)
    decision: Literal["ADMIT", "ADMIT_WITH_LIMITATIONS", "RETURN_FOR_REPAIR", "REJECT"]
    actor_ref: str = Field(min_length=1)
    quality_profile: ArtifactPinV1
    quality_evaluation: ArtifactPinV1
    execution_plan_ref: str = Field(min_length=1)
    execution_plan_hash: str = Field(pattern=SHA256_PATTERN)
    unsupported_scope: tuple[str, ...]
    unresolved_items: tuple[str, ...]
    prohibited_runtime_uses: tuple[str, ...]
    decided_at: datetime

    @model_validator(mode="after")
    def limitations_are_machine_readable(self) -> "OntologyAdmissionDecisionV1":
        if self.decision == "ADMIT_WITH_LIMITATIONS" and not (
            self.unsupported_scope or self.unresolved_items or self.prohibited_runtime_uses
        ):
            raise ValueError("limited admission requires machine-readable limitations")
        if self.decision == "ADMIT" and (
            self.unsupported_scope or self.unresolved_items or self.prohibited_runtime_uses
        ):
            raise ValueError("unrestricted admission cannot retain limitations")
        return self


class OntologyReleaseBundleV1(_Contract):
    schema_version: Literal["ontology_release_bundle.v1"] = "ontology_release_bundle.v1"
    release_bundle_ref: str = Field(min_length=1)
    release_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=SHA256_PATTERN)
    admission_decision_ref: str = Field(min_length=1)
    admission_decision_hash: str = Field(pattern=SHA256_PATTERN)
    execution_plan_ref: str = Field(min_length=1)
    execution_plan_hash: str = Field(pattern=SHA256_PATTERN)
    accepted_graph_release_ref: str | None = None
    governed_rule_artifact_refs: tuple[str, ...]
    activation_state: Literal["NOT_ACTIVE", "ACTIVE"] = "NOT_ACTIVE"
    created_at: datetime


class RuntimeExecutionEpochV1(_Contract):
    schema_version: Literal["runtime_execution_epoch.v1"] = "runtime_execution_epoch.v1"
    epoch_ref: str = Field(min_length=1)
    epoch_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    business_event_ref: str = Field(min_length=1)
    plan_ref: str = Field(min_length=1)
    plan_version_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_normative_ir_hash: str = Field(pattern=SHA256_PATTERN)
    effective_policy_ref: str = Field(min_length=1)
    effective_policy_hash: str = Field(pattern=SHA256_PATTERN)
    runtime_program_hash: str = Field(pattern=SHA256_PATTERN)
    intake_at: datetime
    business_effective_at: datetime


class HumanDispositionV1(_Contract):
    schema_version: Literal["human_disposition.v1"] = "human_disposition.v1"
    disposition_ref: str = Field(min_length=1)
    disposition_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    evaluation_bundle_ref: str = Field(min_length=1)
    evaluation_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    actor_ref: str = Field(min_length=1)
    disposition: Literal[
        "CONFIRMED", "FALSE_POSITIVE", "FALSE_NEGATIVE", "DISPUTED", "NEEDS_MORE_EVIDENCE"
    ]
    rationale: str = Field(min_length=1)
    disposed_at: datetime


class LearningEvidenceV1(_Contract):
    schema_version: Literal["learning_evidence.v1"] = "learning_evidence.v1"
    learning_evidence_ref: str = Field(min_length=1)
    learning_evidence_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    evaluation_bundle_ref: str = Field(min_length=1)
    evaluation_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    disposition_ref: str = Field(min_length=1)
    disposition_hash: str = Field(pattern=SHA256_PATTERN)
    fixture_candidate_kind: Literal["positive", "hard_negative", "ambiguity", "boundary", "disputed"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    admissible_reuse_scope: tuple[str, ...] = Field(min_length=1)
    mutation_forbidden: Literal[True] = True
    created_at: datetime


class FailureAttributionV1(_Contract):
    schema_version: Literal["failure_attribution.v1"] = "failure_attribution.v1"
    attribution_ref: str = Field(min_length=1)
    attribution_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    learning_evidence_ref: str = Field(min_length=1)
    learning_evidence_hash: str = Field(pattern=SHA256_PATTERN)
    owning_layer: Literal[
        "observation_extraction",
        "competency_schema",
        "canonical_normative_ir",
        "validation_contract",
        "source_boundary",
        "runtime_plan_compiler",
    ]
    owner_artifact_ref: str = Field(min_length=1)
    proposed_successor_kind: str = Field(min_length=1)
    created_at: datetime


__all__ = [
    "AdmitOntologyCommandV1",
    "CertifiedSemanticExecutionPlanV1",
    "FailureAttributionV1",
    "HumanDispositionV1",
    "LearningEvidenceV1",
    "OntologyAdmissionDecisionV1",
    "OntologyReleaseBundleV1",
    "RuntimeExecutionEpochV1",
]
