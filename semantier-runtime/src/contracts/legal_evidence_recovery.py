"""DEVELOPMENT evidence availability observations, never legal judgments."""
from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import SourceIdentity
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1, Sha256
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block

StableId = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")]
ScopeId = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:\-]*$")]
Role = Literal["definition", "scope_condition", "size_condition", "exception", "anti_evasion", "approval_procedure"]
State = Literal["RECOVERED", "MISSING", "UNKNOWN", "CONFLICTING"]

def text_hash(text: str) -> str:
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()

def payload_hash(payload) -> str:
    return "sha256:" + content_hash(payload)

class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

class LegalEvidenceQueryV1(_Contract):
    query_id: int = Field(strict=True, ge=0)
    title: str = Field(min_length=1)
    clause_content: str = Field(min_length=1)

class LegalEvidenceRequestV1(_Contract):
    schema_version: Literal["legal_evidence_request.v1"] = "legal_evidence_request.v1"
    task_id: Literal["legal_proposition_evidence_recovery"] = "legal_proposition_evidence_recovery"
    classification: Literal["DEVELOPMENT"]
    organization_id: ScopeId
    workspace_id: ScopeId
    run_id: ScopeId
    question: str = Field(min_length=1)
    query: LegalEvidenceQueryV1
    corpus: ArtifactPinV1
    configuration: ArtifactPinV1
    normalized_document_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def input_binding(self):
        if self.corpus.artifact_kind != "source_only_corpus":
            raise ValueError("source-only corpus pin required")
        if self.configuration.artifact_kind != "evidence_recovery_configuration":
            raise ValueError("evidence recovery configuration pin required")
        refs = self.normalized_document_refs
        if len(set(refs)) != len(refs) or any(not ref or not ref.isascii() for ref in refs):
            raise ValueError("unique ASCII normalized document references required")
        return self

class RequirementObservationV1(_Contract):
    requirement_id: StableId
    question: str = Field(min_length=1)
    basis_refs: tuple[StableId, ...]
    evidence_refs: tuple[StableId, ...]
    candidate_role: Role | None = None
    state: State
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_state(self):
        for refs in (self.basis_refs, self.evidence_refs):
            if len(set(refs)) != len(refs):
                raise ValueError("duplicate observation refs")
        if self.state in {"RECOVERED", "CONFLICTING"} and not self.evidence_refs:
            raise ValueError("located evidence required")
        if self.state == "MISSING" and self.evidence_refs:
            raise ValueError("MISSING cannot contain recovered evidence")
        if self.state != "UNKNOWN" and not self.basis_refs:
            raise ValueError("unresolved basis must remain UNKNOWN")
        return self

class LegalEvidenceSourceV1(_Contract):
    normalized_document_ref: str
    identity: SourceIdentity
    canonical_document: CanonicalSourceDocument
    regions: tuple[RuntimeSemanticSourceRegionV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def source_membership(self):
        identity, document = self.identity, self.canonical_document
        if identity.created_at.utcoffset() != timedelta(0):
            raise ValueError("source timestamp must be UTC")
        if document.source_identity_ref != identity.source_identity_ref:
            raise ValueError("canonical source identity mismatch")
        canonical_hash = payload_hash(document.model_dump(mode="json", by_alias=True))
        seen = set()
        blocks = {block.block_id: block for block in document.blocks}
        if len(blocks) != len(document.blocks):
            raise ValueError("duplicate canonical source block")
        for region in self.regions:
            binding = region.source_binding
            if (binding.source_identity_ref, binding.source_hash) != (identity.source_identity_ref, identity.source_hash):
                raise ValueError("source identity/hash mismatch")
            if binding.canonical_source_ir.artifact_hash != canonical_hash:
                raise ValueError("canonical source hash mismatch")
            # The persisted content-addressed canonical reference is itself bound.
            expected_ref = "artifacts/canonical-source-ir/" + canonical_hash[7:] + "/canonical_source_ir.v1.json"
            if binding.canonical_source_ir.artifact_ref != expected_ref:
                raise ValueError("canonical reference/hash mismatch")
            for anchor in region.evidence:
                evidence = anchor.evidence_ref
                if evidence.evidence_ref in seen:
                    raise ValueError("duplicate native evidence")
                seen.add(evidence.evidence_ref)
                block = blocks.get(evidence.provenance.get("source_block_id"))
                if block is None:
                    raise ValueError("evidence block not in canonical source")
                expected = evidence_ref_for_block(document, block, source_identity=identity,
                    evidence_role="target", created_at=identity.created_at)
                provenance = dict(expected.provenance) | {
                    "canonical_source_ir_ref": expected_ref, "canonical_source_ir_hash": canonical_hash,
                }
                expected_payload = expected.model_dump(mode="json", by_alias=True) | {"provenance": provenance}
                if evidence.model_dump(mode="json", by_alias=True) != expected_payload:
                    raise ValueError("native source selector/quote/provenance mismatch")
        return self

class LegalEvidencePassageV1(_Contract):
    evidence_ref: StableId
    source_identity_ref: StableId
    source_version: str | None
    quote_text_hash: Sha256
    start_codepoint: int = Field(strict=True, ge=0)
    end_codepoint: int = Field(strict=True, gt=0)
    offset_unit: Literal["unicode_codepoint"] = "unicode_codepoint"
    exact_quote: str = Field(min_length=1)

class LegalEvidenceResultV1(_Contract):
    schema_version: Literal["legal_evidence_result.v1"] = "legal_evidence_result.v1"
    contract_hash: Sha256
    request: LegalEvidenceRequestV1
    request_hash: Sha256
    query_text_hash: Sha256
    sources: tuple[LegalEvidenceSourceV1, ...] = Field(min_length=1)
    input_hash: Sha256
    passages: tuple[LegalEvidencePassageV1, ...]
    observations: tuple[RequirementObservationV1, ...]
    review_state: Literal["UNREVIEWED"] = "UNREVIEWED"
    complete_dependency_closure: Literal["UNESTABLISHED"] = "UNESTABLISHED"
    applicability: Literal["UNKNOWN"] = "UNKNOWN"
    produced_at: datetime
    result_hash: Sha256

    @model_validator(mode="after")
    def verify(self):
        if self.contract_hash != contract_hash():
            raise ValueError("task contract hash mismatch")
        if self.produced_at.tzinfo is None or self.produced_at.utcoffset() != timedelta(0):
            raise ValueError("production timestamp must be UTC")
        req = self.request
        if self.request_hash != payload_hash(req.model_dump(mode="json")):
            raise ValueError("request/configuration hash mismatch")
        if self.query_text_hash != text_hash(req.query.clause_content):
            raise ValueError("query text hash mismatch")
        if tuple(s.normalized_document_ref for s in self.sources) != req.normalized_document_refs:
            raise ValueError("access scope differs from request")
        refs = {}
        for source in self.sources:
            identity = source.identity
            if (identity.organization_id, identity.workspace_id) != (req.organization_id, req.workspace_id):
                raise ValueError("source scope mismatch")
            for region in source.regions:
                for anchor in region.evidence:
                    ref = anchor.evidence_ref.evidence_ref
                    if ref in refs:
                        raise ValueError("ambiguous evidence across sources")
                    refs[ref] = (source, anchor)
        sources_payload = [source.model_dump(mode="json", by_alias=True) for source in self.sources]
        if self.input_hash != payload_hash({"request_hash": self.request_hash, "sources": sources_payload}):
            raise ValueError("source/version/input hash mismatch")
        passage_refs = set()
        for passage in self.passages:
            if passage.evidence_ref in passage_refs or passage.evidence_ref not in refs:
                raise ValueError("passage must uniquely reference retained native evidence")
            passage_refs.add(passage.evidence_ref)
            source, anchor = refs[passage.evidence_ref]
            quote = anchor.evidence_ref.selector.exact_quote
            if (passage.source_identity_ref, passage.source_version) != (
                source.identity.source_identity_ref, source.identity.source_version
            ):
                raise ValueError("passage source/version mismatch")
            if passage.quote_text_hash != text_hash(quote):
                raise ValueError("passage text hash mismatch")
            if not 0 <= passage.start_codepoint < passage.end_codepoint <= len(quote):
                raise ValueError("Unicode codepoint span out of bounds")
            if quote[passage.start_codepoint:passage.end_codepoint] != passage.exact_quote:
                raise ValueError("Unicode span quote mismatch")
        ids = [o.requirement_id for o in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate task requirement")
        for observation in self.observations:
            if not set(observation.basis_refs) <= refs.keys():
                raise ValueError("requirement basis must be retained evidence")
            if not set(observation.evidence_refs) <= passage_refs:
                raise ValueError("observation evidence must have retained passages")
            if observation.state == "CONFLICTING":
                claims = {(refs[ref][0].identity.source_identity_ref,
                           refs[ref][0].identity.source_version, refs[ref][1].text_hash)
                          for ref in observation.evidence_refs}
                if len(claims) < 2:
                    raise ValueError("CONFLICTING requires distinct retained claims")
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("result_hash")
        if self.result_hash != payload_hash(payload):
            raise ValueError("evidence result hash mismatch")
        return self

    @classmethod
    def create(cls, *, request, sources, passages, observations, produced_at):
        request = LegalEvidenceRequestV1.model_validate(request)
        request_hash = payload_hash(request.model_dump(mode="json"))
        payload = dict(schema_version="legal_evidence_result.v1", contract_hash=contract_hash(),
            request=request.model_dump(mode="json"), request_hash=request_hash,
            query_text_hash=text_hash(request.query.clause_content),
            sources=[source.model_dump(mode="json", by_alias=True) for source in sources],
            passages=[passage.model_dump(mode="json") for passage in passages],
            observations=[o.model_dump(mode="json") for o in observations],
            review_state="UNREVIEWED", complete_dependency_closure="UNESTABLISHED",
            applicability="UNKNOWN", produced_at=TypeAdapter(datetime).dump_python(produced_at, mode="json"))
        payload["input_hash"] = payload_hash({"request_hash": request_hash, "sources": payload["sources"]})
        payload["result_hash"] = payload_hash(payload)
        return cls.model_validate(payload)

def contract_hash() -> str:
    # Schema plus machine-readable semantic limits, not runtime prompt prose.
    return payload_hash({
        "version": "legal_proposition_evidence_recovery.v1",
        "request_schema": LegalEvidenceRequestV1.model_json_schema(),
        "result_schema": LegalEvidenceResultV1.model_json_schema(),
        "verification": "mechanical_membership_only.v1",
        "recovered": "passage_availability", "missing": "recorded_access_only",
        "role": "candidate_association", "conflicting": "no_precedence",
        "closure": "UNESTABLISHED", "applicability": "UNKNOWN", "review": "UNREVIEWED",
    })
