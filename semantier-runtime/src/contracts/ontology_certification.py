"""Immutable contracts for the A0-to-certify ontology successor seam."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class AuthoritySectionPinV1(_Contract):
    section_id: str = Field(min_length=1)
    stable_locator: str = Field(min_length=1)
    text_hash: str = Field(min_length=1)
    local_hash: str = Field(min_length=1)
    dependency_roles: tuple[
        Literal[
            "rule_authority",
            "scope_authority",
            "exception_authority",
            "qualification_authority",
            "effectivity_authority",
            "precedence_authority",
        ],
        ...,
    ] = Field(min_length=1)


class NormativeAuthorityDependencyV1(_Contract):
    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    authority_tier: str = Field(min_length=1)
    governance_state: str = Field(min_length=1)
    jurisdiction: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    artifact_hash: str = Field(pattern=SHA256_PATTERN)
    # Legal artifact identity hashes its envelope; source identity hashes raw bytes.
    source_content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN,
        exclude_if=lambda value: value is None)
    storage_ref: str = Field(min_length=1)
    effective_from: datetime
    effective_to: datetime | None = None
    sections: tuple[AuthoritySectionPinV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def effective_window_is_valid(self) -> "NormativeAuthorityDependencyV1":
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("authority dependency effective_to must follow effective_from")
        return self


class NormativeAuthorityDependencySetV1(_Contract):
    schema_version: Literal["normative_authority_dependency_set.v1"] = (
        "normative_authority_dependency_set.v1"
    )
    set_ref: str = Field(min_length=1)
    set_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    origin_build_ref: str = Field(min_length=1)
    origin_build_hash: str = Field(pattern=SHA256_PATTERN)
    origin_snapshot_ref: str = Field(min_length=1)
    origin_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    jurisdiction: str = Field(min_length=1)
    as_of_time: datetime
    dependencies: tuple[NormativeAuthorityDependencyV1, ...] = Field(min_length=1)
    search_coverage_ref: str = Field(min_length=1)
    search_coverage_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime

    @model_validator(mode="after")
    def dependencies_are_compatible(self) -> "NormativeAuthorityDependencySetV1":
        keys = [(item.source_id, item.version_id) for item in self.dependencies]
        if len(keys) != len(set(keys)):
            raise ValueError("authority dependencies must be unique")
        for dependency in self.dependencies:
            if dependency.jurisdiction != self.jurisdiction:
                raise ValueError("authority dependency jurisdiction mismatch")
            if self.as_of_time < dependency.effective_from:
                raise ValueError("authority dependency is not yet effective")
            if dependency.effective_to is not None and self.as_of_time > dependency.effective_to:
                raise ValueError("authority dependency is no longer effective")
        return self


class OntologyCertificationCommandV1(_Contract):
    schema_version: Literal["ontology_certification_command.v1"] = (
        "ontology_certification_command.v1"
    )
    request_id: str = Field(min_length=1)
    origin_build_ref: str = Field(min_length=1)
    origin_build_hash: str = Field(pattern=SHA256_PATTERN)
    origin_snapshot_ref: str = Field(min_length=1)
    origin_snapshot_hash: str = Field(pattern=SHA256_PATTERN)
    reviewed_competency_contract_ref: str = Field(min_length=1)
    reviewed_competency_contract_hash: str = Field(pattern=SHA256_PATTERN)
    authority_dependency_set_ref: str = Field(min_length=1)
    authority_dependency_set_hash: str = Field(pattern=SHA256_PATTERN)
    target_assurance_profile: Literal["A1_INTERNAL_POLICY_MVL"] = "A1_INTERNAL_POLICY_MVL"
    actor_ref: str = Field(min_length=1)


class OntologyCertificationReadinessV1(_Contract):
    schema_version: Literal["ontology_certification_readiness.v1"] = (
        "ontology_certification_readiness.v1"
    )
    readiness_ref: str = Field(min_length=1)
    readiness_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    origin_build_ref: str = Field(min_length=1)
    origin_build_hash: str = Field(pattern=SHA256_PATTERN)
    compile_certifiable_blockers: tuple[str, ...]
    certify_blockers: tuple[str, ...]
    admit_blockers: tuple[str, ...]
    activation_blockers: tuple[str, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    assessed_at: datetime


class OntologyCertificationReceiptV1(_Contract):
    schema_version: Literal["ontology_certification_receipt.v1"] = (
        "ontology_certification_receipt.v1"
    )
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    request_hash: str = Field(pattern=SHA256_PATTERN)
    origin_build_ref: str = Field(min_length=1)
    origin_build_hash: str = Field(pattern=SHA256_PATTERN)
    successor_build_ref: str | None = None
    successor_build_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    successor_snapshot_ref: str | None = None
    successor_snapshot_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    status: Literal["CERTIFIED_BOUNDED", "PAUSED", "FAILED"]
    blocker_refs: tuple[str, ...]
    closed_gap_ids: tuple[str, ...]
    introduced_gap_ids: tuple[str, ...]
    target_assurance_profile: Literal["A1_INTERNAL_POLICY_MVL"]
    certified_scope: str = Field(min_length=1)
    prohibited_claims: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def successor_is_paired(self) -> "OntologyCertificationReceiptV1":
        pairs = (
            (self.successor_build_ref, self.successor_build_hash),
            (self.successor_snapshot_ref, self.successor_snapshot_hash),
        )
        if any((ref is None) != (digest is None) for ref, digest in pairs):
            raise ValueError("certification successor refs/hashes must be paired")
        if self.status == "CERTIFIED_BOUNDED" and any(ref is None for ref, _ in pairs):
            raise ValueError("certified receipt requires successor build and snapshot")
        return self


__all__ = [
    "AuthoritySectionPinV1",
    "NormativeAuthorityDependencyV1",
    "NormativeAuthorityDependencySetV1",
    "OntologyCertificationCommandV1",
    "OntologyCertificationReadinessV1",
    "OntologyCertificationReceiptV1",
]
