from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from eos.artifact_hashing import content_hash
from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class CompetencyHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(min_length=1)
    intended_use: tuple[str, ...] = Field(min_length=1)
    required_competencies: tuple[str, ...] = Field(min_length=1)
    required_definition_families: tuple[str, ...] = Field(min_length=1)
    required_normative_dimensions: tuple[str, ...] = ()
    required_relation_families: tuple[str, ...] = ()


class OntologySeedQuery(BaseModel):
    """Content-bound discovery request derived from governed intention artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="ontology_seed_query.v1", pattern=r"^ontology_seed_query\.v1$")
    query_ref: str
    query_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    graph_intent_ref: str = Field(min_length=1)
    graph_intent_hash: str = Field(pattern=SHA256_PATTERN)
    competency_contract_ref: str | None = None
    competency_contract_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    competency_contract_content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    competency_hypothesis: CompetencyHypothesis
    created_at: datetime

    def hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "build_ref": self.build_ref,
            "graph_intent_ref": self.graph_intent_ref,
            "graph_intent_hash": self.graph_intent_hash,
            "competency_contract_ref": self.competency_contract_ref,
            "competency_contract_hash": self.competency_contract_hash,
            "competency_contract_content_hash": self.competency_contract_content_hash,
            "competency_hypothesis": self.competency_hypothesis.model_dump(mode="json"),
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
        }

    @model_validator(mode="after")
    def validate_content_binding(self) -> "OntologySeedQuery":
        if self.created_at.utcoffset() != timezone.utc.utcoffset(self.created_at):
            raise ValueError("created_at must be timezone-aware UTC")
        competency_pins = (
            self.competency_contract_ref, self.competency_contract_hash,
            self.competency_contract_content_hash,
        )
        if any(item is None for item in competency_pins) != all(item is None for item in competency_pins):
            raise ValueError("competency contract ref, claimed hash, and content hash must be paired")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.query_hash != expected_hash:
            raise ValueError("query_hash does not match canonical query content")
        if self.query_ref != f"ontology-seed-query:{expected_hash}":
            raise ValueError("query_ref does not bind canonical query content")
        return self

    @classmethod
    def create(cls, **values: Any) -> "OntologySeedQuery":
        created_at = values.get("created_at")
        if not isinstance(created_at, datetime) or created_at.tzinfo is None:
            # Construct through Pydantic so callers consistently receive ValidationError.
            values["query_hash"] = "sha256:" + "0" * 64
            values["query_ref"] = "ontology-seed-query:invalid"
            return cls(**values)
        material = {
            "schema_version": values.get("schema_version", "ontology_seed_query.v1"),
            "organization_id": values.get("organization_id"),
            "workspace_id": values.get("workspace_id"),
            "build_ref": values.get("build_ref"),
            "graph_intent_ref": values.get("graph_intent_ref"),
            "graph_intent_hash": values.get("graph_intent_hash"),
            "competency_contract_ref": values.get("competency_contract_ref"),
            "competency_contract_hash": values.get("competency_contract_hash"),
            "competency_contract_content_hash": values.get("competency_contract_content_hash"),
            "competency_hypothesis": (
                values["competency_hypothesis"].model_dump(mode="json")
                if isinstance(values.get("competency_hypothesis"), CompetencyHypothesis)
                else values.get("competency_hypothesis")
            ),
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
        }
        digest = "sha256:" + content_hash(material)
        return cls(query_ref=f"ontology-seed-query:{digest}", query_hash=digest, **values)
