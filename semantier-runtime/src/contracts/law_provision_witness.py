"""Additive complete-provision contracts; historical witness v1 is unchanged."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from contracts.evidence_location import EvidenceRef
from contracts.source_semantic_witness import SemanticExpectation, WitnessSourceBinding, SHA256_PATTERN
from eos.artifact_hashing import content_hash


class _Pinned(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode='after')
    def verify_hash(self):
        payload = self.model_dump(mode='json')
        payload.pop('content_hash')
        if self.content_hash != 'sha256:' + content_hash(payload):
            raise ValueError('provision content hash mismatch')
        return self

    @classmethod
    def create(cls, payload):
        payload = dict(payload)
        payload.pop('content_hash', None)
        return cls.model_validate({**payload, 'content_hash': 'sha256:' + content_hash(payload)})


class ProvisionMemberV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    block_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    block_content_hash: str = Field(pattern=SHA256_PATTERN)
    use: Literal['provision_text', 'page_furniture', 'section_heading']
    evidence: EvidenceRef


class LawProvisionSpanV1(_Pinned):
    schema_version: Literal['law_provision_span.v1']
    source_role: Literal['law_provisions']
    boundary_policy: Literal['cn_article_blocks.v1']
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_binding: WitnessSourceBinding
    graph_intent_ref: str = Field(min_length=1)
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    profile_id: str = Field(min_length=1)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    article_label: str = Field(min_length=1)
    members: tuple[ProvisionMemberV1, ...] = Field(min_length=1)
    complete_text_hash: str = Field(pattern=SHA256_PATTERN)


class LawProvisionWitnessV1(_Pinned):
    schema_version: Literal['law_provision_witness.v1']
    source_role: Literal['law_provisions']
    span_hash: str = Field(pattern=SHA256_PATTERN)
    span_review_ref: str = Field(min_length=1)
    span_review_hash: str = Field(pattern=SHA256_PATTERN)
    semantic_expectation: SemanticExpectation


class LawProvisionContrastV1(_Pinned):
    schema_version: Literal['law_provision_contrast.v1']
    construction_kind: Literal['review_only_counterexample']
    authority_promotion_allowed: Literal[False]
    positive_witness_hash: str = Field(pattern=SHA256_PATTERN)
    scenario: str = Field(min_length=1)
    rejected_interpretation: str = Field(min_length=1)
    expected_distinction: str = Field(min_length=1)


class ProvisionManifestEntryV1(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    witness_hash: str = Field(pattern=SHA256_PATTERN)
    review_event_ref: str | None
    review_event_hash: str | None = Field(pattern=SHA256_PATTERN)


class LawProvisionWitnessManifestV1(_Pinned):
    schema_version: Literal['law_provision_witness_manifest.v1']
    witness_kind: Literal['positive_exemplar', 'hard_negative']
    state: Literal['proposed', 'accepted']
    predecessor_manifest_hash: str | None = Field(pattern=SHA256_PATTERN)
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    source_binding: WitnessSourceBinding
    taxonomy_ref: str
    taxonomy_hash: str = Field(pattern=SHA256_PATTERN)
    profile_id: str
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    entries: tuple[ProvisionManifestEntryV1, ...] = Field(min_length=1)

    @model_validator(mode='after')
    def verify_lineage(self):
        if len({e.witness_hash for e in self.entries}) != len(self.entries):
            raise ValueError('duplicate witness in provision manifest')
        if self.state == 'proposed':
            if self.predecessor_manifest_hash is not None or any(
                e.review_event_ref is not None or e.review_event_hash is not None for e in self.entries
            ):
                raise ValueError('proposed manifest cannot assert acceptance')
        elif self.predecessor_manifest_hash is None or any(
            not e.review_event_ref or not e.review_event_hash for e in self.entries
        ):
            raise ValueError('accepted manifest requires predecessor and exact reviews')
        return self
