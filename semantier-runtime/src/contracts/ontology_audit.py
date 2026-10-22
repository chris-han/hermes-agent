"""Deterministic offline audit package for one exact ontology build."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import SHA256_PATTERN


class OntologyAuditArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_kind: str = Field(min_length=1)
    artifact_ref: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=SHA256_PATTERN)
    schema_version: str = Field(min_length=1)
    payload: dict[str, Any]


class OntologyAuditExportV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["ontology_audit_export.v1"] = "ontology_audit_export.v1"
    audit_ref: str = Field(min_length=1)
    audit_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    build_hash: str = Field(pattern=SHA256_PATTERN)
    artifacts: tuple[OntologyAuditArtifactV1, ...]
    replay_mode: Literal["OFFLINE_PINNED_ONLY"] = "OFFLINE_PINNED_ONLY"
    exported_at: datetime

    @model_validator(mode="after")
    def audit_invariants(self) -> "OntologyAuditExportV1":
        if self.exported_at.tzinfo is None or self.exported_at.utcoffset() != timedelta(0):
            raise ValueError("exported_at must be timezone-aware UTC")
        identities = [(item.artifact_kind, item.artifact_ref) for item in self.artifacts]
        if len(identities) != len(set(identities)):
            raise ValueError("audit artifacts must be uniquely pinned")
        return self


__all__ = ["OntologyAuditArtifactV1", "OntologyAuditExportV1"]
