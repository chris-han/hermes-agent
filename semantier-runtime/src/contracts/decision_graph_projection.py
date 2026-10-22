"""Derived graph-native decision projection contracts over DecisionLocalProjectionV2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash

_SHA = r"^sha256:[0-9a-f]{64}$"


def _digest(value) -> str:
    return "sha256:" + content_hash(value)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectionSufficiencyProofV1(_Frozen):
    schema_version: Literal["projection_sufficiency_proof.v1"] = "projection_sufficiency_proof.v1"
    proof_ref: str = Field(min_length=1)
    proof_hash: str = Field(pattern=_SHA)
    status: Literal["SUFFICIENT", "INSUFFICIENT"]
    missing_dependency_ids: tuple[str, ...] = ()
    frontier_ref: str = Field(min_length=1)
    frontier_hash: str = Field(pattern=_SHA)
    dlp_ref: str = Field(min_length=1)
    dlp_hash: str = Field(pattern=_SHA)
    required_node_ids: tuple[str, ...] = ()
    required_edge_ids: tuple[str, ...] = ()
    required_assertion_ids: tuple[str, ...] = ()
    evidence: tuple[ArtifactPinV1, ...] = ()
    authority_bindings: tuple[ArtifactPinV1, ...] = ()
    unresolved_conflict_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def verify_identity_and_status(self):
        if self.status == "SUFFICIENT" and self.missing_dependency_ids:
            raise ValueError("sufficient proof cannot contain missing dependencies")
        digest = _digest(self.hash_payload())
        if (self.proof_ref, self.proof_hash) != (
            "projection-sufficiency-proof:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("projection sufficiency proof identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"proof_ref", "proof_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            proof_ref="projection-sufficiency-proof:v1:pending",
            proof_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "proof_ref": "projection-sufficiency-proof:v1:" + digest[7:],
            "proof_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="projection_sufficiency_proof",
            artifact_ref=self.proof_ref,
            artifact_hash=self.proof_hash,
        )




class PredicateEpistemicRequirementV1(_Frozen):
    schema_version: Literal["predicate_epistemic_requirement.v1"] = "predicate_epistemic_requirement.v1"
    predicate_id: str = Field(min_length=1)
    required_semantic_dimension_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def verify_dimensions(self):
        ordered = tuple(sorted(self.required_semantic_dimension_ids, key=lambda item: item.encode("utf-8")))
        if ordered != self.required_semantic_dimension_ids or len(ordered) != len(set(ordered)):
            raise ValueError("semantic dimension identifiers must be unique and canonically ordered")
        return self


class PredicateEpistemicAdmissibilityV1(_Frozen):
    schema_version: Literal["predicate_epistemic_admissibility.v1"] = "predicate_epistemic_admissibility.v1"
    predicate_id: str = Field(min_length=1)
    status: Literal["ADMISSIBLE", "INADMISSIBLE"]
    required_semantic_dimension_ids: tuple[str, ...] = ()
    available_semantic_dimension_ids: tuple[str, ...] = ()
    missing_semantic_dimension_ids: tuple[str, ...] = ()
    projection_sufficiency_proof: ArtifactPinV1
    disposition: Literal["SCORE_RESIDUAL_FRONTIER", "UNKNOWN"]

    @model_validator(mode="after")
    def verify_admissibility(self):
        for values in (
            self.required_semantic_dimension_ids,
            self.available_semantic_dimension_ids,
            self.missing_semantic_dimension_ids,
        ):
            ordered = tuple(sorted(values, key=lambda item: item.encode("utf-8")))
            if ordered != values or len(ordered) != len(set(ordered)):
                raise ValueError("semantic dimension identifiers must be unique and canonically ordered")
        expected_missing = tuple(
            item for item in self.required_semantic_dimension_ids
            if item not in set(self.available_semantic_dimension_ids)
        )
        if self.missing_semantic_dimension_ids != expected_missing:
            raise ValueError("semantic dimension missing-set mismatch")
        expected_status = "INADMISSIBLE" if expected_missing else "ADMISSIBLE"
        expected_disposition = "UNKNOWN" if expected_missing else "SCORE_RESIDUAL_FRONTIER"
        if self.status != expected_status or self.disposition != expected_disposition:
            raise ValueError("epistemic admissibility status/disposition mismatch")
        return self


class EpistemicAdmissibilityRoutingV1(_Frozen):
    schema_version: Literal["epistemic_admissibility_routing.v1"] = "epistemic_admissibility_routing.v1"
    routing_ref: str = Field(min_length=1)
    routing_hash: str = Field(pattern=_SHA)
    decision_graph_projection: ArtifactPinV1
    projection_sufficiency_proof: ArtifactPinV1
    predicate_admissibility: tuple[PredicateEpistemicAdmissibilityV1, ...] = ()
    scorer_eligible_predicate_ids: tuple[str, ...] = ()
    structurally_unknown_predicate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def verify_routing(self):
        predicate_ids = tuple(item.predicate_id for item in self.predicate_admissibility)
        if predicate_ids != tuple(sorted(predicate_ids, key=lambda item: item.encode("utf-8"))):
            raise ValueError("predicate admissibility must be canonically ordered")
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("predicate admissibility identifiers must be unique")
        eligible = tuple(item.predicate_id for item in self.predicate_admissibility if item.status == "ADMISSIBLE")
        unknown = tuple(item.predicate_id for item in self.predicate_admissibility if item.status == "INADMISSIBLE")
        if self.scorer_eligible_predicate_ids != eligible or self.structurally_unknown_predicate_ids != unknown:
            raise ValueError("epistemic routing partition mismatch")
        if set(eligible) & set(unknown):
            raise ValueError("epistemic routing partitions must be disjoint")
        digest = _digest(self.hash_payload())
        if (self.routing_ref, self.routing_hash) != (
            "epistemic-admissibility-routing:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("epistemic admissibility routing identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"routing_ref", "routing_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            routing_ref="epistemic-admissibility-routing:v1:pending",
            routing_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "routing_ref": "epistemic-admissibility-routing:v1:" + digest[7:],
            "routing_hash": digest,
        })


class DecisionGraphProjectionV1(_Frozen):
    schema_version: Literal["decision_graph_projection.v1"] = "decision_graph_projection.v1"
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=_SHA)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_dlp_ref: str = Field(min_length=1)
    source_dlp_hash: str = Field(pattern=_SHA)
    source_model_input_hash: str = Field(pattern=_SHA)
    frontier_ref: str = Field(min_length=1)
    frontier_hash: str = Field(pattern=_SHA)
    canonical_context: ArtifactPinV1
    working_context: ArtifactPinV1
    ontology: ArtifactPinV1
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    assertion_ids: tuple[str, ...] = ()
    evidence: tuple[ArtifactPinV1, ...] = ()
    authority_bindings: tuple[ArtifactPinV1, ...] = ()
    unresolved_conflict_ids: tuple[str, ...] = ()
    world_time_basis: dict
    sufficiency_proof: ArtifactPinV1

    @model_validator(mode="after")
    def verify_identity(self):
        for values, label in (
            (self.node_ids, "node"),
            (self.edge_ids, "edge"),
            (self.assertion_ids, "assertion"),
            (self.unresolved_conflict_ids, "conflict"),
        ):
            if tuple(sorted(values, key=lambda item: item.encode("utf-8"))) != values:
                raise ValueError(f"{label} identifiers must be canonically ordered")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} identifiers must be unique")
        digest = _digest(self.hash_payload())
        if (self.projection_ref, self.projection_hash) != (
            "decision-graph-projection:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("decision graph projection identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"projection_ref", "projection_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            projection_ref="decision-graph-projection:v1:pending",
            projection_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "projection_ref": "decision-graph-projection:v1:" + digest[7:],
            "projection_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_graph_projection",
            artifact_ref=self.projection_ref,
            artifact_hash=self.projection_hash,
        )


class DecisionGraphProjectionReceiptV1(_Frozen):
    schema_version: Literal["decision_graph_projection_receipt.v1"] = "decision_graph_projection_receipt.v1"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=_SHA)
    projection: ArtifactPinV1
    sufficiency_proof: ArtifactPinV1
    evidence_view: ArtifactPinV1
    source_dlp_ref: str = Field(min_length=1)
    source_dlp_hash: str = Field(pattern=_SHA)
    frontier_ref: str = Field(min_length=1)
    frontier_hash: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def verify_identity(self):
        digest = _digest(self.hash_payload())
        if (self.receipt_ref, self.receipt_hash) != (
            "decision-graph-projection-receipt:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("decision graph projection receipt identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            receipt_ref="decision-graph-projection-receipt:v1:pending",
            receipt_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "receipt_ref": "decision-graph-projection-receipt:v1:" + digest[7:],
            "receipt_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_graph_projection_receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
        )


class DecisionEvidenceViewV1(_Frozen):
    schema_version: Literal["decision_evidence_view.v1"] = "decision_evidence_view.v1"
    view_ref: str = Field(min_length=1)
    view_hash: str = Field(pattern=_SHA)
    decision_graph_projection: ArtifactPinV1
    source_dlp_ref: str = Field(min_length=1)
    source_dlp_hash: str = Field(pattern=_SHA)
    model_input_hash: str = Field(pattern=_SHA)
    evidence_handles: tuple[tuple[str, ArtifactPinV1], ...] = ()

    @model_validator(mode="after")
    def verify_identity(self):
        refs = tuple(handle for handle, _pin in self.evidence_handles)
        if len(refs) != len(set(refs)):
            raise ValueError("evidence handles must be unique")
        digest = _digest(self.hash_payload())
        if (self.view_ref, self.view_hash) != (
            "decision-evidence-view:v1:" + digest[7:],
            digest,
        ):
            raise ValueError("decision evidence view identity mismatch")
        return self

    def hash_payload(self) -> dict:
        return self.model_dump(mode="json", exclude={"view_ref", "view_hash"})

    @classmethod
    def create(cls, **body):
        provisional = cls.model_construct(
            view_ref="decision-evidence-view:v1:pending",
            view_hash="sha256:" + "0" * 64,
            **body,
        )
        digest = _digest(provisional.hash_payload())
        return cls.model_validate({
            **body,
            "view_ref": "decision-evidence-view:v1:" + digest[7:],
            "view_hash": digest,
        })

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_evidence_view",
            artifact_ref=self.view_ref,
            artifact_hash=self.view_hash,
        )
