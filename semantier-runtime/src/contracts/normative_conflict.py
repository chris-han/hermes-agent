"""Content-addressed contracts for governed normative-set compatibility analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import content_hash
from contracts.normative_schema import NormativeFactDefinitionV1


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc_timestamps(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class NormativeDomainBasisV2(_Contract):
    schema_version: Literal["normative_domain_basis.v2"] = "normative_domain_basis.v2"
    organization_id: str
    workspace_id: str
    schema_ref: str
    schema_hash: str = Field(pattern=SHA256_PATTERN)
    facts: tuple[NormativeFactDefinitionV1, ...]

    @model_validator(mode="after")
    def unique_facts(self):
        refs = tuple(f.fact_ref for f in self.facts)
        if refs != tuple(sorted(set(refs))):
            raise ValueError("domain facts must be unique and ordered")
        return self

    @classmethod
    def from_schema(cls, schema):
        return cls(organization_id=schema.organization_id, workspace_id=schema.workspace_id,
                   schema_ref=schema.schema_ref, schema_hash=schema.schema_hash,
                   facts=tuple(sorted(schema.fact_definitions or (), key=lambda f: f.fact_ref)))

    def verify_candidate(self, candidate):
        if (self.organization_id, self.workspace_id, self.schema_ref, self.schema_hash) != (
            candidate.organization_id, candidate.workspace_id,
            candidate.normative_schema_ref, candidate.normative_schema_hash,
        ):
            raise ValueError("NORMATIVE_DOMAIN_BASIS_MISMATCH")


class NormativeAnalysisScopeV2(_Contract):
    schema_version: Literal["normative_analysis_scope.v2"] = "normative_analysis_scope.v2"
    policy: Literal["typed_bounded_algebra.v2"] = "typed_bounded_algebra.v2"
    coverage: Literal["COMPLETE", "INCOMPLETE"]
    checked_effect_refs: tuple[str, ...]
    conflict_groups: tuple[tuple[str, ...], ...]
    enumeration: Literal["DISCOVERED_FACT_GROUPS"] = "DISCOVERED_FACT_GROUPS"
    minimality: Literal["NOT_CHECKED"] = "NOT_CHECKED"


class NormativeEffectV1(_Contract):
    schema_version: Literal["normative_effect.v1"] = "normative_effect.v1"
    effect_ref: str
    effect_hash: str = Field(pattern=SHA256_PATTERN)
    rule_ref: str
    modality: Literal["obligation", "prohibition"]
    fact_ref: str
    operator: Literal["EQ", "IN", "GE", "GT", "LE", "LT"]
    expected_value: str | int | float | bool | tuple[str, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    ir_node_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity(self):
        payload = self.model_dump(mode="json", exclude={"effect_ref", "effect_hash"})
        digest = content_hash(payload)
        if self.effect_hash != "sha256:" + digest or self.effect_ref != "normative-effect:" + digest:
            raise ValueError("normative effect identity mismatch")
        return self


class NormativeSetV1(_Contract):
    schema_version: Literal["normative_set.v1"] = "normative_set.v1"
    normative_set_ref: str
    normative_set_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    context_ref: str
    context_hash: str = Field(pattern=SHA256_PATTERN)
    compiled_rule_plan_ref: str
    compiled_rule_plan_hash: str = Field(pattern=SHA256_PATTERN)
    member_rule_refs: tuple[str, ...]
    effects: tuple[NormativeEffectV1, ...]
    inactive_rule_refs: tuple[str, ...]
    unknown_rule_refs: tuple[str, ...]
    closure_state: Literal["COMPLETE", "INCOMPLETE"]
    unresolved_refs: tuple[str, ...]
    domain_basis: NormativeDomainBasisV2 | None = Field(default=None, exclude_if=lambda v: v is None)
    formed_at: datetime

    @model_validator(mode="after")
    def integrity(self):
        for refs in (self.member_rule_refs, self.inactive_rule_refs, self.unknown_rule_refs, self.unresolved_refs):
            if refs != tuple(sorted(set(refs), key=lambda x: x.encode("utf-8"))):
                raise ValueError("normative set refs must be unique and canonically ordered")
        if self.domain_basis is not None and (
            self.domain_basis.organization_id, self.domain_basis.workspace_id
        ) != (self.organization_id, self.workspace_id):
            raise ValueError("normative domain scope mismatch")
        effect_rules = {item.rule_ref for item in self.effects}
        if not effect_rules.issubset(set(self.member_rule_refs)):
            raise ValueError("normative effects must belong to member rules")
        payload = self.model_dump(mode="json", exclude={"normative_set_ref", "normative_set_hash"})
        digest = content_hash(payload)
        if self.normative_set_hash != "sha256:" + digest or self.normative_set_ref != "normative-set:" + digest:
            raise ValueError("normative set identity mismatch")
        return self


class NormativeCompatibilityResultV1(_Contract):
    schema_version: Literal["normative_compatibility_result.v1"] = "normative_compatibility_result.v1"
    result_ref: str
    result_hash: str = Field(pattern=SHA256_PATTERN)
    normative_set_ref: str
    normative_set_hash: str = Field(pattern=SHA256_PATTERN)
    outcome: Literal["COMPATIBLE", "CONFLICT", "UNKNOWN"]
    proof_kind: Literal["bounded_effect_algebra", "incomplete_semantics"]
    conflicting_effect_refs: tuple[str, ...]
    unresolved_refs: tuple[str, ...]
    evaluator_ref: str
    evaluator_hash: str = Field(pattern=SHA256_PATTERN)
    analysis_scope: NormativeAnalysisScopeV2 | None = Field(default=None, exclude_if=lambda v: v is None)
    evaluated_at: datetime

    @model_validator(mode="after")
    def identity(self):
        minimum = 1 if self.analysis_scope is not None else 2
        if self.outcome == "CONFLICT" and len(self.conflicting_effect_refs) < minimum:
            raise ValueError("conflict requires witness effects")
        if self.outcome != "CONFLICT" and self.conflicting_effect_refs:
            raise ValueError("non-conflict result cannot carry conflicting effects")
        if self.analysis_scope is not None:
            groups = self.analysis_scope.conflict_groups
            discovered = tuple(sorted({ref for group in groups for ref in group}))
            if discovered != self.conflicting_effect_refs or not set(discovered) <= set(self.analysis_scope.checked_effect_refs):
                raise ValueError("conflict witnesses must conserve checked scope")
            if self.outcome == "COMPATIBLE" and self.analysis_scope.coverage != "COMPLETE":
                raise ValueError("incomplete scope cannot claim compatibility")
            if self.unresolved_refs and self.analysis_scope.coverage == "COMPLETE":
                raise ValueError("unresolved references require incomplete coverage")
        payload = self.model_dump(mode="json", exclude={"result_ref", "result_hash"})
        digest = content_hash(payload)
        if self.result_hash != "sha256:" + digest or self.result_ref != "normative-compatibility:" + digest:
            raise ValueError("compatibility result identity mismatch")
        return self


class NormativeConflictRevalidationTargetV1(_Contract):
    schema_version: Literal["normative_conflict_revalidation_target.v1"] = (
        "normative_conflict_revalidation_target.v1"
    )
    target_ref: str
    target_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    changed_rule_ref: str
    changed_release_ref: str
    changed_release_hash: str = Field(pattern=SHA256_PATTERN)
    reason: Literal["RULE_CHANGED", "RULE_SUPERSEDED"]
    conflict_core_ref: str
    conflict_core_hash: str = Field(pattern=SHA256_PATTERN)
    normative_set_ref: str
    context_ref: str
    prior_resolution_refs: tuple[str, ...]

    @model_validator(mode="after")
    def identity(self):
        if self.prior_resolution_refs != tuple(
            sorted(set(self.prior_resolution_refs), key=lambda x: x.encode("utf-8"))
        ):
            raise ValueError("prior resolution refs must be unique and canonical")
        payload = self.model_dump(mode="json", exclude={"target_ref", "target_hash"})
        digest = content_hash(payload)
        if (
            self.target_hash != "sha256:" + digest
            or self.target_ref != "normative-conflict-revalidation-target:" + digest
        ):
            raise ValueError("normative conflict revalidation target identity mismatch")
        return self


class NormativeConflictRevalidationSuccessorV1(_Contract):
    schema_version: Literal["normative_conflict_revalidation_successor.v1"] = (
        "normative_conflict_revalidation_successor.v1"
    )
    successor_ref: str
    successor_hash: str = Field(pattern=SHA256_PATTERN)
    target_ref: str
    prior_conflict_core_ref: str
    successor_compatibility_result_ref: str
    successor_compatibility_result_hash: str = Field(pattern=SHA256_PATTERN)
    successor_normative_set_ref: str
    outcome: Literal[
        "CONFLICT_PERSISTS",
        "CONFLICT_DISAPPEARS",
        "CONFLICT_MEMBERSHIP_CHANGED",
        "UNKNOWN",
    ]
    successor_conflict_core_ref: str | None = None

    @model_validator(mode="after")
    def identity(self):
        requires_core = self.outcome in {
            "CONFLICT_PERSISTS", "CONFLICT_MEMBERSHIP_CHANGED"
        }
        if requires_core != (self.successor_conflict_core_ref is not None):
            raise ValueError("revalidation successor core/outcome mismatch")
        payload = self.model_dump(mode="json", exclude={"successor_ref", "successor_hash"})
        digest = content_hash(payload)
        if (
            self.successor_hash != "sha256:" + digest
            or self.successor_ref != "normative-conflict-revalidation-successor:" + digest
        ):
            raise ValueError("normative conflict revalidation successor identity mismatch")
        return self


class ConflictResolutionV1(_Contract):
    schema_version: Literal["conflict_resolution.v1"] = "conflict_resolution.v1"
    resolution_ref: str
    resolution_hash: str = Field(pattern=SHA256_PATTERN)
    conflict_core_ref: str
    outcome: Literal["RESOLVED_BY_PRECEDENCE", "UNRESOLVED"]
    controlling_rule_refs: tuple[str, ...]
    resolution_evidence_refs: tuple[str, ...]
    resolved_at: datetime

    @model_validator(mode="after")
    def identity(self):
        if self.outcome == "RESOLVED_BY_PRECEDENCE" and not self.controlling_rule_refs:
            raise ValueError("precedence resolution requires a controlling rule")
        if self.outcome == "UNRESOLVED" and self.controlling_rule_refs:
            raise ValueError("unresolved conflict cannot claim a controlling rule")
        payload = self.model_dump(mode="json", exclude={"resolution_ref", "resolution_hash"})
        digest = content_hash(payload)
        if self.resolution_hash != "sha256:" + digest or self.resolution_ref != "conflict-resolution:" + digest:
            raise ValueError("conflict resolution identity mismatch")
        return self


class ConflictCoreV1(_Contract):
    schema_version: Literal["conflict_core.v1"] = "conflict_core.v1"
    conflict_core_ref: str
    conflict_core_hash: str = Field(pattern=SHA256_PATTERN)
    compatibility_result_ref: str
    normative_set_ref: str
    context_ref: str
    analysis_policy: Literal["typed_bounded_algebra.v2"] | None = Field(default=None, exclude_if=lambda v: v is None)
    member_rule_refs: tuple[str, ...] = Field(min_length=1)
    member_effect_refs: tuple[str, ...] = Field(min_length=1)
    minimality: Literal["VERIFIED_MINIMAL", "NOT_CHECKED"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity(self):
        if self.analysis_policy is None and (len(self.member_rule_refs) < 2 or len(self.member_effect_refs) < 2):
            raise ValueError("legacy conflict cores require multiple members")
        if self.member_rule_refs != tuple(sorted(set(self.member_rule_refs), key=lambda x: x.encode("utf-8"))):
            raise ValueError("conflict core rule refs must be canonical")
        payload = self.model_dump(mode="json", exclude={"conflict_core_ref", "conflict_core_hash"})
        digest = content_hash(payload)
        if self.conflict_core_hash != "sha256:" + digest or self.conflict_core_ref != "conflict-core:" + digest:
            raise ValueError("conflict core identity mismatch")
        return self
