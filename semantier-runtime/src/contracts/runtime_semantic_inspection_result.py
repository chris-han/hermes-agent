"""Content-addressed completion of one governed semantic inspection."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.governed_frame_finding_v1 import GovernedFrameFindingV1
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from eos.artifact_hashing import content_hash


class InspectionArtifactRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: str = Field(min_length=1, pattern=r"^[\x21-\x7e]+$")
    hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class RuntimeSemanticInspectionResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["runtime_semantic_inspection_result.v1"] = "runtime_semantic_inspection_result.v1"
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    result_ref: str
    result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str = Field(min_length=1, pattern=r"^[\x21-\x7e]+$")
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canonical_source_ir_ref: str = Field(min_length=1)
    canonical_source_ir_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    retrieval_snapshot_refs: tuple[InspectionArtifactRefV1, ...] = Field(min_length=1)
    observed_hypothesis_refs: tuple[InspectionArtifactRefV1, ...]
    frame_match_result_refs: tuple[InspectionArtifactRefV1, ...]
    governed_finding_refs: tuple[InspectionArtifactRefV1, ...]
    governed_findings: tuple[GovernedFrameFindingV1, ...]

    @model_validator(mode="after")
    def verify_identity(self):
        basis = self.model_dump(mode="json", by_alias=True, exclude={"result_ref", "result_hash"})
        digest = content_hash(basis)
        if self.result_hash != "sha256:" + digest or self.result_ref != "semantic-inspection-result:" + digest:
            raise ValueError("inspection result hash mismatch")
        expected = [(item.finding_id, item.finding_hash) for item in self.governed_findings]
        if [(item.ref, item.hash) for item in self.governed_finding_refs] != expected:
            raise ValueError("inspection finding pins mismatch")
        matches = {(item.ref, item.hash) for item in self.frame_match_result_refs}
        hypotheses = {(item.ref, item.hash) for item in self.observed_hypothesis_refs}
        for finding in self.governed_findings:
            if (
                (finding.organization_id, finding.workspace_id, finding.source_identity_ref, finding.source_hash)
                != (self.organization_id, self.workspace_id, self.source_identity_ref, self.source_hash)
                or (finding.frame_match_result_ref, finding.frame_match_result_hash) not in matches
                or (finding.observed_hypothesis_ref, finding.observed_hypothesis_hash) not in hypotheses
            ):
                raise ValueError("inspection finding lineage mismatch")
        for refs in (self.retrieval_snapshot_refs, self.observed_hypothesis_refs, self.frame_match_result_refs, self.governed_finding_refs):
            if len({item.ref for item in refs}) != len(refs):
                raise ValueError("duplicate inspection artifact ref")
        return self

    @classmethod
    def create(cls, **payload):
        digest = content_hash(payload)
        return cls.model_validate({**payload, "result_ref": "semantic-inspection-result:" + digest, "result_hash": "sha256:" + digest})


class RuntimeSemanticInspectionContextV1(BaseModel):
    """Exact input regions, perception and formed contexts preceding judgment."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["runtime_semantic_inspection_context.v1"] = "runtime_semantic_inspection_context.v1"
    organization_id: str
    workspace_id: str
    context_ref: str
    context_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    run_id: str
    normalized_document_artifact_ref: str
    request_id: str | None
    runtime_context_pin: ArtifactPinV1 | None
    source_regions: tuple[RuntimeSemanticSourceRegionV1, ...] = Field(min_length=1)
    retrieval_snapshot_refs: tuple[InspectionArtifactRefV1, ...] = Field(min_length=1)
    working_context_pins: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    decision_projection_refs: tuple[InspectionArtifactRefV1, ...] = Field(min_length=1)

    perception_pins: tuple[ArtifactPinV1, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def verify_identity(self):
        basis = self.model_dump(mode="json", by_alias=True, exclude={"context_ref", "context_hash"})
        digest = content_hash(basis)
        if self.context_hash != "sha256:" + digest or self.context_ref != "semantic-inspection-context:" + digest:
            raise ValueError("inspection context hash mismatch")
        if self.perception_pins and len(self.perception_pins) != len(self.source_regions):
            raise ValueError("inspection perception membership mismatch")
        if len(self.source_regions) != len(self.retrieval_snapshot_refs) or len(self.working_context_pins) != len(self.decision_projection_refs):
            raise ValueError("inspection context membership mismatch")
        if any((r.evidence[0].evidence_ref.organization_id, r.evidence[0].evidence_ref.workspace_id) != (self.organization_id, self.workspace_id) for r in self.source_regions):
            raise ValueError("inspection context tenant mismatch")
        if any(r.source_binding != self.source_regions[0].source_binding for r in self.source_regions):
            raise ValueError("inspection context source mismatch")
        return self

    @classmethod
    def create(cls, **payload):
        digest = content_hash(payload)
        return cls.model_validate({**payload, "context_ref": "semantic-inspection-context:" + digest, "context_hash": "sha256:" + digest})
