"""Corpus observations that may justify governed schema/ontology proposals."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.knowledge_temporal_state import parse_utc
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
GOVERNANCE_PATH = (
    "ProfileRelationTypeProposalV1",
    "versioned_successor_profile_catalog",
    "CorpusGraphSchemaCandidate.v4",
    "SchemaDecision.v3",
)


class CorpusGrowthObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["corpus_growth_observation.v1"] = (
        "corpus_growth_observation.v1"
    )
    observation_ref: str
    observation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    semantic_key: str
    proposal_kind: Literal["concept", "relation_type"]
    source_identity_ref: str
    evidence_ref: str
    graph_intent_ref: str
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    observed_at: str

    @model_validator(mode="after")
    def integrity(self) -> "CorpusGrowthObservationV1":
        parse_utc(self.observed_at, "observed_at")
        for field in (
            "organization_id",
            "workspace_id",
            "semantic_key",
            "source_identity_ref",
            "evidence_ref",
            "graph_intent_ref",
        ):
            if _ASCII.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"{field} must be ASCII-stable")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.observation_hash != expected:
            raise ValueError("observation hash does not match canonical content")
        if self.observation_ref != "corpus-growth-observation:" + expected[7:31]:
            raise ValueError("observation ref does not match observation hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_ref")
        payload.pop("observation_hash")
        return payload


class CorpusGrowthCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["corpus_growth_candidate.v1"] = "corpus_growth_candidate.v1"
    candidate_ref: str
    candidate_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    semantic_key: str
    proposal_kind: Literal["concept", "relation_type"]
    recurrence_count: int = Field(ge=1)
    observation_refs: tuple[str, ...]
    source_identity_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    graph_intent_refs: tuple[str, ...]
    lifecycle_state: Literal["candidate"] = "candidate"
    required_governance_path: tuple[str, ...] = GOVERNANCE_PATH

    @model_validator(mode="after")
    def integrity(self) -> "CorpusGrowthCandidateV1":
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.candidate_hash != expected:
            raise ValueError("candidate hash does not match canonical projection")
        if self.candidate_ref != "corpus-growth-candidate:" + expected[7:31]:
            raise ValueError("candidate ref does not match candidate hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("candidate_ref")
        payload.pop("candidate_hash")
        return payload


def create_corpus_growth_observation(**values: Any) -> CorpusGrowthObservationV1:
    payload = {"schema_version": "corpus_growth_observation.v1", **values}
    digest = content_hash(payload)
    return CorpusGrowthObservationV1.model_validate(
        {
            **payload,
            "observation_ref": "corpus-growth-observation:" + digest[:24],
            "observation_hash": "sha256:" + digest,
        }
    )


def project_corpus_growth_candidate(
    observations: tuple[CorpusGrowthObservationV1, ...],
) -> CorpusGrowthCandidateV1:
    if not observations:
        raise ValueError("at least one corpus observation is required")
    first = observations[0]
    scope = {
        (item.organization_id, item.workspace_id, item.semantic_key, item.proposal_kind)
        for item in observations
    }
    if len(scope) != 1:
        raise ValueError("corpus observations must share one governed scope")
    payload = {
        "schema_version": "corpus_growth_candidate.v1",
        "organization_id": first.organization_id,
        "workspace_id": first.workspace_id,
        "semantic_key": first.semantic_key,
        "proposal_kind": first.proposal_kind,
        "recurrence_count": len({item.source_identity_ref for item in observations}),
        "observation_refs": tuple(
            sorted((item.observation_ref for item in observations), key=str.encode)
        ),
        "source_identity_refs": tuple(
            sorted({item.source_identity_ref for item in observations}, key=str.encode)
        ),
        "evidence_refs": tuple(
            sorted({item.evidence_ref for item in observations}, key=str.encode)
        ),
        "graph_intent_refs": tuple(
            sorted({item.graph_intent_ref for item in observations}, key=str.encode)
        ),
        "lifecycle_state": "candidate",
        "required_governance_path": GOVERNANCE_PATH,
    }
    digest = content_hash(payload)
    return CorpusGrowthCandidateV1.model_validate(
        {
            **payload,
            "candidate_ref": "corpus-growth-candidate:" + digest[:24],
            "candidate_hash": "sha256:" + digest,
        }
    )


__all__ = [
    "CorpusGrowthCandidateV1",
    "CorpusGrowthObservationV1",
    "create_corpus_growth_observation",
    "project_corpus_growth_candidate",
]
