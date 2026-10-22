from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from eos.artifact_hashing import content_hash
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OntologySeedFitV1(_Contract):
    domain_supported: bool
    represented_intended_tasks: tuple[str, ...]
    missing_intended_tasks: tuple[str, ...]
    represented_competencies: tuple[str, ...]
    missing_competencies: tuple[str, ...]
    represented_definition_families: tuple[str, ...]
    missing_definition_families: tuple[str, ...]
    represented_relation_families: tuple[str, ...]
    missing_relation_families: tuple[str, ...]
    supported_dimensions: tuple[str, ...]
    unsupported_dimensions: tuple[str, ...]
    unresolved_reason_codes: tuple[str, ...]
    coverage_ratio: float = Field(ge=0, le=1)
    retrieval_score: int = Field(ge=0)


class OntologySeedAuthorityV1(_Contract):
    seed_class: Literal["REFERENCE", "ORGANIZATION", "DOMAIN_TEMPLATE"]
    authority_role: str = Field(min_length=1)
    governance_state: str = Field(min_length=1)
    is_authoritative: Literal[False] = False


class SeedRecommendationV1(_Contract):
    schema_version: Literal["ontology_seed_recommendation.v1"] = "ontology_seed_recommendation.v1"
    recommendation_id: str
    recommendation_ref: str
    recommendation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    query_ref: str = Field(min_length=1)
    query_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_ref: str = Field(min_length=1)
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    competency_contract_ref: str | None = None
    competency_contract_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    competency_contract_content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    ontology_ref: str = Field(min_length=1)
    ontology_version: str = Field(min_length=1)
    ontology_source_content_hash: str = Field(pattern=SHA256_PATTERN)
    ontology_graph_content_hash: str = Field(pattern=SHA256_PATTERN)
    profile_ref: str = Field(min_length=1)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    role: Literal["PRIMARY_SEED", "SUPPORTING_SEED"]
    fit: OntologySeedFitV1
    authority: OntologySeedAuthorityV1
    policy_version: str = Field(min_length=1)
    policy_hash: str = Field(pattern=SHA256_PATTERN)
    status: Literal["PROPOSED", "SELECTED", "IGNORED", "SUPERSEDED"] = "PROPOSED"
    created_at: datetime

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"recommendation_id", "recommendation_ref", "recommendation_hash"}
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "SeedRecommendationV1":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        pins = (self.competency_contract_ref, self.competency_contract_hash,
                self.competency_contract_content_hash)
        if any(item is None for item in pins) != all(item is None for item in pins):
            raise ValueError("competency contract pins must be paired")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.recommendation_hash != expected:
            raise ValueError("recommendation_hash does not bind exact content")
        if self.recommendation_ref != f"ontology-seed-recommendation:{expected[7:]}":
            raise ValueError("recommendation_ref does not bind exact content")
        if self.recommendation_id != f"seed-rec-{expected[7:31]}":
            raise ValueError("recommendation_id does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: object) -> "SeedRecommendationV1":
        provisional = {
            "schema_version": "ontology_seed_recommendation.v1", "status": "PROPOSED", **values
        }
        digest = content_hash(to_jsonable_python(provisional))
        return cls(
            recommendation_id=f"seed-rec-{digest[:24]}",
            recommendation_ref=f"ontology-seed-recommendation:{digest}",
            recommendation_hash=f"sha256:{digest}", **values,
        )


class OntologySeedMatchResultV1(_Contract):
    schema_version: Literal["ontology_seed_match_result.v1"] = "ontology_seed_match_result.v1"
    status: Literal["matched", "no_match"]
    query_ref: str
    query_hash: str = Field(pattern=SHA256_PATTERN)
    policy_version: str
    policy_hash: str = Field(pattern=SHA256_PATTERN)
    recommendations: tuple[SeedRecommendationV1, ...]
    reason_codes: tuple[str, ...]

    @model_validator(mode="after")
    def result_shape(self) -> "OntologySeedMatchResultV1":
        if (self.status == "matched") != bool(self.recommendations):
            raise ValueError("match status and recommendations disagree")
        return self
