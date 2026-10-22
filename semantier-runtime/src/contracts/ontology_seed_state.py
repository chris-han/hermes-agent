from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from eos.artifact_hashing import content_hash
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from contracts.ontology_catalog import OntologyCatalogCandidate
from contracts.ontology_seed_query import OntologySeedQuery
from contracts.ontology_seed_recommendation import SeedRecommendationV1


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactPinV1(_Contract):
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=SHA256_PATTERN)


class SeedPolicySnapshotV1(_Contract):
    policy_version: str = Field(min_length=1)
    policy_hash: str = Field(pattern=SHA256_PATTERN)
    config: dict[str, object]

    @model_validator(mode="after")
    def validate_config_hash(self) -> "SeedPolicySnapshotV1":
        if self.policy_hash != "sha256:" + content_hash(self.config):
            raise ValueError("policy_hash does not bind complete effective config")
        if self.config.get("policy_version") != self.policy_version:
            raise ValueError("policy version does not match effective config")
        return self


class SeedProviderFailureV1(_Contract):
    provider_id: str = Field(min_length=1, max_length=128)
    reason_code: Literal["PROVIDER_FAILURE", "PROVIDER_TIMEOUT", "DISCOVERY_DEADLINE"]


class OntologySeedStateV1(_Contract):
    schema_version: Literal["ontology_seed_state.v1"] = "ontology_seed_state.v1"
    state_ref: str
    state_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=SHA256_PATTERN)
    revision: int = Field(ge=1)
    predecessor_state_ref: str | None = None
    predecessor_state_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    search_status: Literal["not_started", "searching", "searched", "matched", "no_match"]
    query: OntologySeedQuery | None = None
    policy: SeedPolicySnapshotV1 | None = None
    recommendations: tuple[SeedRecommendationV1, ...] = ()
    ignored_recommendation_refs: tuple[str, ...] = ()
    external_candidates: tuple[OntologyCatalogCandidate, ...] = ()
    provider_failures: tuple[SeedProviderFailureV1, ...] = ()
    selected_recommendation: ArtifactPinV1 | None = None
    imported_snapshot: ArtifactPinV1 | None = None
    alignment_actions: tuple[ArtifactPinV1, ...] = ()
    source_grounded_delta: ArtifactPinV1 | None = None
    selection_receipt: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    latest_apply_receipt: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    created_at: datetime

    @property
    def visible_recommendations(self) -> tuple[SeedRecommendationV1, ...]:
        ignored = set(self.ignored_recommendation_refs)
        return tuple(item for item in self.recommendations if item.recommendation_ref not in ignored)

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"state_ref", "state_hash"})

    @model_validator(mode="after")
    def validate_state(self) -> "OntologySeedStateV1":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        if (self.predecessor_state_ref is None) != (self.predecessor_state_hash is None):
            raise ValueError("predecessor state ref/hash must be paired")
        if (self.revision == 1) != (self.predecessor_state_ref is None):
            raise ValueError("state revision and predecessor disagree")
        if self.search_status == "not_started" and (
            self.query is not None or self.policy is not None or self.recommendations
            or self.external_candidates or self.provider_failures
            or self.ignored_recommendation_refs
        ):
            raise ValueError("not_started state cannot contain discovery inputs or results")
        if self.search_status != "not_started" and (self.query is None or self.policy is None):
            raise ValueError("active discovery state requires exact query and policy")
        if self.query is not None and (
            self.query.organization_id, self.query.workspace_id, self.query.build_ref,
        ) != (self.organization_id, self.workspace_id, self.build_ref):
            raise ValueError("query does not bind state tenant/build")
        if self.search_status == "searching" and (
            self.recommendations or self.external_candidates or self.provider_failures
        ):
            raise ValueError("searching state cannot publish completion results")
        if self.search_status == "matched" and not self.recommendations:
            raise ValueError("matched state requires recommendations")
        if self.search_status in {"searched", "no_match"} and self.recommendations:
            raise ValueError("non-matched completion cannot publish recommendations")
        if len(self.ignored_recommendation_refs) != len(set(self.ignored_recommendation_refs)):
            raise ValueError("ignored recommendation refs must be unique")
        known = {item.recommendation_ref for item in self.recommendations}
        if not set(self.ignored_recommendation_refs) <= known:
            raise ValueError("ignored recommendation must belong to this state")
        for recommendation in self.recommendations:
            if (
                recommendation.organization_id, recommendation.workspace_id,
                recommendation.build_ref, recommendation.query_ref,
                recommendation.query_hash, recommendation.policy_hash,
            ) != (
                self.organization_id, self.workspace_id, self.build_ref,
                self.query.query_ref, self.query.query_hash, self.policy.policy_hash,
            ):
                raise ValueError("recommendation does not bind exact state inputs")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.state_hash != expected or self.state_ref != f"ontology-seed-state:{expected[7:]}":
            raise ValueError("state ref/hash does not bind exact state content")
        return self

    @classmethod
    def create(cls, **values: object) -> "OntologySeedStateV1":
        values = {name: value for name, value in values.items()
                  if name not in ('selection_receipt', 'latest_apply_receipt') or value is not None}
        material = to_jsonable_python({
            "schema_version": "ontology_seed_state.v1",
            "predecessor_state_ref": None, "predecessor_state_hash": None,
            "query": None, "policy": None, "recommendations": (),
            "ignored_recommendation_refs": (), "external_candidates": (),
            "provider_failures": (), "selected_recommendation": None,
            "imported_snapshot": None, "alignment_actions": (),
            "source_grounded_delta": None, **values,
        })
        digest = content_hash(material)
        return cls(state_ref=f"ontology-seed-state:{digest}", state_hash=f"sha256:{digest}", **values)


class OntologySeedIgnoreReceiptV1(_Contract):
    schema_version: Literal["ontology_seed_ignore.v1"] = "ontology_seed_ignore.v1"
    ignore_ref: str
    ignore_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    build_ref: str
    query_ref: str
    query_hash: str = Field(pattern=SHA256_PATTERN)
    recommendation_ref: str
    recommendation_hash: str = Field(pattern=SHA256_PATTERN)
    predecessor_state_ref: str
    predecessor_state_hash: str = Field(pattern=SHA256_PATTERN)
    successor_state_ref: str
    successor_state_hash: str = Field(pattern=SHA256_PATTERN)
    reason_code: str = Field(min_length=1, max_length=128)
    ignored_at: datetime

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"ignore_ref", "ignore_hash"})

    @model_validator(mode="after")
    def validate_receipt(self) -> "OntologySeedIgnoreReceiptV1":
        if self.ignored_at.tzinfo is None or self.ignored_at.utcoffset() != timedelta(0):
            raise ValueError("ignored_at must be timezone-aware UTC")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.ignore_hash != expected or self.ignore_ref != f"ontology-seed-ignore:{expected[7:]}":
            raise ValueError("ignore receipt ref/hash does not bind exact action")
        return self

    @classmethod
    def create(cls, **values: object) -> "OntologySeedIgnoreReceiptV1":
        material = to_jsonable_python({"schema_version": "ontology_seed_ignore.v1", **values})
        digest = content_hash(material)
        return cls(ignore_ref=f"ontology-seed-ignore:{digest}", ignore_hash=f"sha256:{digest}", **values)
