"""Derived decision-local projection contracts over an exact WorkingContext.

These values are content-addressed projections/configuration, not sixth semantic-kernel
objects and not independent authorities.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or _ASCII_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable identifier/reference")
    return value


class SemanticCandidateV1(_Contract):
    candidate_id: Literal["A", "B", "C"]
    semantic_value: Literal["TRUE", "FALSE", "UNKNOWN"]


class MaterialDependencyRequirementV1(_Contract):
    dependency_id: str = Field(min_length=1)
    required_evidence_role: str = Field(min_length=1)
    min_count: int = Field(default=1, ge=1)

    @field_validator("dependency_id", "required_evidence_role")
    @classmethod
    def stable_identity(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)


class MaterialDependencyResolutionV1(_Contract):
    predicate_id: str = Field(min_length=1)
    dependency_id: str = Field(min_length=1)
    required_evidence_role: str = Field(min_length=1)
    min_count: int = Field(ge=1)
    status: Literal["SATISFIED", "MISSING"]
    matched_evidence: tuple[ArtifactPinV1, ...] = ()

    @field_validator("predicate_id", "dependency_id", "required_evidence_role")
    @classmethod
    def stable_identity(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)

    @model_validator(mode="after")
    def status_matches_evidence(self) -> "MaterialDependencyResolutionV1":
        if (self.status == "SATISFIED") != (len(self.matched_evidence) >= self.min_count):
            raise ValueError("material dependency status does not match exact matched evidence")
        if self.status == "MISSING" and self.matched_evidence:
            raise ValueError("missing material dependency cannot contain matched evidence")
        if any(pin.artifact_kind != "evidence" for pin in self.matched_evidence):
            raise ValueError("material dependency matches must pin evidence")
        if len({pin.artifact_ref for pin in self.matched_evidence}) != len(self.matched_evidence):
            raise ValueError("material dependency matched evidence must be unique")
        return self


_DEFAULT_CANDIDATES = (
    SemanticCandidateV1(candidate_id="A", semantic_value="TRUE"),
    SemanticCandidateV1(candidate_id="B", semantic_value="FALSE"),
    SemanticCandidateV1(candidate_id="C", semantic_value="UNKNOWN"),
)


class SemanticPredicateSpecV1(_Contract):
    predicate_id: str = Field(min_length=1)
    semantic_type_ref: str = Field(min_length=1)
    family_ref: str = Field(min_length=1)
    candidates: tuple[SemanticCandidateV1, ...] = _DEFAULT_CANDIDATES
    material_requirements: tuple[MaterialDependencyRequirementV1, ...] = ()

    @field_validator("predicate_id", "semantic_type_ref", "family_ref")
    @classmethod
    def stable_refs(cls, value: str, info) -> str:
        return _ascii(value, info.field_name)

    @model_validator(mode="after")
    def exact_candidate_contract(self) -> "SemanticPredicateSpecV1":
        if self.candidates != _DEFAULT_CANDIDATES:
            raise ValueError("semantic predicate candidates must be exact A=TRUE/B=FALSE/C=UNKNOWN")
        requirements = self.material_requirements
        identities = tuple(item.dependency_id for item in requirements)
        if len(identities) != len(set(identities)):
            raise ValueError("material dependency IDs must be unique within a predicate")
        if tuple(sorted(requirements, key=lambda item: item.dependency_id.encode("ascii"))) != requirements:
            raise ValueError("material requirements must be in canonical dependency ID order")
        return self


class DecisionContractV1(_Contract):
    schema_version: Literal["decision_contract.v1"] = "decision_contract.v1"
    contract_ref: str = Field(min_length=1)
    contract_hash: str = Field(pattern=SHA256_PATTERN)
    purpose: str = Field(min_length=1)
    semantic_predicates: tuple[SemanticPredicateSpecV1, ...] = Field(min_length=1)
    deterministic_constraints: tuple[ArtifactPinV1, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        predicates = tuple(value.get("semantic_predicates", ()))
        value["semantic_predicates"] = tuple(
            sorted(
                predicates,
                key=lambda item: (
                    item.get("predicate_id") if isinstance(item, dict) else item.predicate_id
                ).encode("utf-8"),
            )
        )
        constraints = tuple(value.get("deterministic_constraints", ()))
        value["deterministic_constraints"] = tuple(
            sorted(
                constraints,
                key=lambda item: (
                    item.get("artifact_kind") if isinstance(item, dict) else item.artifact_kind,
                    item.get("artifact_ref") if isinstance(item, dict) else item.artifact_ref,
                    item.get("artifact_hash") if isinstance(item, dict) else item.artifact_hash,
                ),
            )
        )
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "DecisionContractV1":
        _ascii(self.contract_ref, "contract_ref")
        predicate_ids = tuple(item.predicate_id for item in self.semantic_predicates)
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("decision contract predicate identities must be unique")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.contract_hash != expected or self.contract_ref != "decision-contract:" + expected[7:]:
            raise ValueError("decision contract ref/hash does not bind exact content")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("contract_ref", None)
        payload.pop("contract_hash", None)
        for predicate in payload["semantic_predicates"]:
            if not predicate.get("material_requirements"):
                predicate.pop("material_requirements", None)
        return payload

    @classmethod
    def create(
        cls,
        *,
        purpose: str,
        semantic_predicates: tuple[SemanticPredicateSpecV1, ...],
        deterministic_constraints: tuple[ArtifactPinV1, ...] = (),
    ) -> "DecisionContractV1":
        canonical = cls.canonicalize(
            {
                "schema_version": "decision_contract.v1",
                "purpose": purpose,
                "semantic_predicates": semantic_predicates,
                "deterministic_constraints": deterministic_constraints,
            }
        )
        provisional = {
            **canonical,
            "contract_ref": "decision-contract:pending",
            "contract_hash": "sha256:" + "0" * 64,
        }
        draft = cls.model_construct(**provisional)
        digest = content_hash(draft.hash_payload())
        return cls.model_validate(
            {
                **canonical,
                "contract_ref": f"decision-contract:{digest}",
                "contract_hash": f"sha256:{digest}",
            }
        )


class DecisionEvidenceProjectionV1(_Contract):
    assertion_id: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    subject_label: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)
    object_label: str = Field(min_length=1)
    evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)


class DecisionLocalProjectionV1(_Contract):
    schema_version: Literal["decision_local_projection.v1"] = "decision_local_projection.v1"
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    context_graph: ArtifactPinV1
    working_context: ArtifactPinV1
    ontology: ArtifactPinV1
    decision_subject_ref: str = Field(min_length=1)
    decision_contract_ref: str = Field(min_length=1)
    decision_contract_hash: str = Field(pattern=SHA256_PATTERN)
    selected_assertion_ids: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    evidence_projection: tuple[DecisionEvidenceProjectionV1, ...] = Field(min_length=1)
    semantic_predicates: tuple[SemanticPredicateSpecV1, ...] = Field(min_length=1)
    deterministic_constraints: tuple[ArtifactPinV1, ...] = ()
    material_dependency_resolutions: tuple[MaterialDependencyResolutionV1, ...] = ()
    model_input_hash: str = Field(pattern=SHA256_PATTERN)
    world_mode: Literal["current", "point", "interval", "prior_state", "causal_window"]
    world_time_basis: dict[str, str | None]
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["selected_assertion_ids"] = tuple(sorted(value.get("selected_assertion_ids", ())))
        value["evidence_projection"] = tuple(
            sorted(
                value.get("evidence_projection", ()),
                key=lambda item: (
                    item.get("assertion_id") if isinstance(item, dict) else item.assertion_id
                ).encode("utf-8"),
            )
        )
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "DecisionLocalProjectionV1":
        for field in (
            "organization_id",
            "workspace_id",
            "decision_subject_ref",
            "decision_contract_ref",
            "compiler_ref",
        ):
            _ascii(getattr(self, field), field)
        if self.context_graph.artifact_kind != "context_graph":
            raise ValueError("decision-local projection requires a context_graph pin")
        if self.working_context.artifact_kind != "working_context":
            raise ValueError("decision-local projection requires a working_context pin")
        if self.ontology.artifact_kind != "ontology":
            raise ValueError("decision-local projection requires an ontology pin")
        expected_input = "sha256:" + content_hash(self.model_input_payload())
        if self.model_input_hash != expected_input:
            raise ValueError("model_input_hash does not bind exact model-visible projection")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.projection_hash != expected or self.projection_ref != "decision-local-projection:" + expected[7:]:
            raise ValueError("decision-local projection ref/hash does not bind exact content")
        return self

    def model_input_payload(self) -> dict[str, Any]:
        payload = {
            "decision_subject_ref": self.decision_subject_ref,
            "decision_contract_ref": self.decision_contract_ref,
            "decision_contract_hash": self.decision_contract_hash,
            "evidence_projection": [item.model_dump(mode="json") for item in self.evidence_projection],
            "semantic_predicates": [item.model_dump(mode="json") for item in self.semantic_predicates],
        }
        for predicate in payload["semantic_predicates"]:
            if not predicate.get("material_requirements"):
                predicate.pop("material_requirements", None)
        if self.material_dependency_resolutions:
            payload["material_dependency_resolutions"] = [
                item.model_dump(mode="json") for item in self.material_dependency_resolutions
            ]
        return payload

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("projection_ref", None)
        payload.pop("projection_hash", None)
        if not payload.get("material_dependency_resolutions"):
            payload.pop("material_dependency_resolutions", None)
        for predicate in payload["semantic_predicates"]:
            if not predicate.get("material_requirements"):
                predicate.pop("material_requirements", None)
        return payload


__all__ = [
    "DecisionContractV1",
    "DecisionEvidenceProjectionV1",
    "DecisionLocalProjectionV1",
    "MaterialDependencyRequirementV1",
    "MaterialDependencyResolutionV1",
    "SemanticCandidateV1",
    "SemanticPredicateSpecV1",
]
