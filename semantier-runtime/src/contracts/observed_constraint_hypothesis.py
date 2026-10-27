"""Runtime semantic-hypothesis and effective-authority contracts."""

from __future__ import annotations

from datetime import datetime
import re
import math
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
_HYPOTHESIS_ID = re.compile(r"^och_[0-9a-f]{24}$")
_FRAME_REQUIRED = {
    "target_type_ref",
    "attribute_ref",
    "operator_ref",
    "value_constraint",
    "modality_ref",
    "stage_ref",
    "consequence_ref",
}
_FRAME_OPTIONAL = {"slot_evidence_refs", "inferred_slots", "schema_version"}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


def _ascii_ref(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.isascii() or _ASCII_REF.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable identifier/reference")
    return value


def _ordered_unique_refs(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    refs = tuple(_ascii_ref(item, f"{field}[]") for item in value)
    if len(refs) != len(set(refs)):
        raise ValueError(f"{field} contains a duplicate identity")
    return refs


def _validate_json(value: Any, field: str) -> None:
    # Native PDF locators carry finite geometry, not semantic numeric values.
    # Preserve those exact bytes in retrieval snapshots; candidate frames and
    # all other query fields retain the existing no-floating-point contract.
    if field in (
        "query.evidence_refs.provenance.source_locator.native_metadata.bbox",
        "query.evidence_refs.provenance.source_locator.native_metadata.text_spans.bbox",
    ):
        if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in value
        ):
            raise ValueError(f"{field} must contain four finite geometry coordinates")
        return
    if isinstance(value, float):
        raise ValueError(f"{field} contains a forbidden floating-point value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or not key.isascii() or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                raise ValueError(f"{field} contains a non-ASCII-stable schema key: {key!r}")
            _validate_json(nested, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json(nested, field)


class HypothesisSourceBinding(_Contract):
    source_identity_schema_version: Literal["semantier.source_identity.v1"] = Field(
        alias="sourceIdentitySchemaVersion"
    )
    source_identity_ref: str = Field(alias="sourceIdentityRef")
    source_hash: str = Field(alias="sourceHash", pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str = Field(alias="canonicalSourceIrRef")
    canonical_source_ir_hash: str = Field(alias="canonicalSourceIrHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_refs(self) -> "HypothesisSourceBinding":
        _ascii_ref(self.source_identity_ref, "sourceIdentityRef")
        _ascii_ref(self.canonical_source_ir_ref, "canonicalSourceIrRef")
        return self


class EffectiveRestrictionRuntimeContextV1(_Contract):
    schema_version: Literal["effective_restriction_runtime_context.v1"] = Field(
        default="effective_restriction_runtime_context.v1", alias="schemaVersion"
    )
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=SHA256_PATTERN)
    governed_rule_projection_ref: str = Field(alias="governedRuleProjectionRef")
    governed_rule_projection_hash: str = Field(alias="governedRuleProjectionHash", pattern=SHA256_PATTERN)
    ordered_governed_rule_version_refs: tuple[str, ...] = Field(alias="orderedGovernedRuleVersionRefs")
    ordered_governed_rule_version_set_hash: str = Field(
        alias="orderedGovernedRuleVersionSetHash", pattern=SHA256_PATTERN
    )
    activation_set_snapshot_ref: str = Field(alias="activationSetSnapshotRef")
    activation_set_snapshot_hash: str = Field(alias="activationSetSnapshotHash", pattern=SHA256_PATTERN)
    activation_resolver_policy_version: str = Field(alias="activationResolverPolicyVersion")
    resolved_activation_set_hash: str = Field(alias="resolvedActivationSetHash", pattern=SHA256_PATTERN)
    effective_context_resolver_hash: str = Field(alias="effectiveContextResolverHash", pattern=SHA256_PATTERN)
    effective_at: str = Field(alias="effectiveAt")

    @field_validator("ordered_governed_rule_version_refs", mode="before")
    @classmethod
    def validate_ordered_rules(cls, value: Any) -> tuple[str, ...]:
        refs = _ordered_unique_refs(value, "orderedGovernedRuleVersionRefs")
        if not refs:
            raise ValueError("orderedGovernedRuleVersionRefs must not be empty")
        return refs

    @model_validator(mode="after")
    def validate_context(self) -> "EffectiveRestrictionRuntimeContextV1":
        for field in (
            "organization_id",
            "workspace_id",
            "accepted_graph_release_ref",
            "governed_rule_projection_ref",
            "activation_set_snapshot_ref",
            "activation_resolver_policy_version",
        ):
            _ascii_ref(getattr(self, field), field)
        try:
            parsed = datetime.fromisoformat(self.effective_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("effectiveAt must be RFC3339 UTC") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("effectiveAt must be RFC3339 UTC")
        return self


class RetrievalMaterialV1(_Contract):
    kind: Literal[
        "canonical_pattern", "lexical_cue", "positive_exemplar", "hard_negative"
    ]
    ref: str
    hash: str = Field(pattern=SHA256_PATTERN)
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_ref(self) -> "RetrievalMaterialV1":
        _ascii_ref(self.ref, "orderedMaterial[].ref")
        return self


class RuntimeSemanticRetrievalSnapshotV1(_Contract):
    schema_version: Literal["runtime_semantic_retrieval_snapshot.v1"] = Field(
        default="runtime_semantic_retrieval_snapshot.v1", alias="schemaVersion"
    )
    snapshot_id: str = Field(alias="snapshotId")
    snapshot_hash: str = Field(alias="snapshotHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    run_id: str = Field(alias="runId")
    query: dict[str, Any]
    ordered_material: tuple[RetrievalMaterialV1, ...] = Field(alias="orderedMaterial", min_length=1)
    retrieval_policy_ref: Literal["runtime-semantic-retrieval.v1", "runtime-semantic-retrieval.v2"] = Field(alias="retrievalPolicyRef")
    retrieval_policy_hash: str = Field(alias="retrievalPolicyHash", pattern=SHA256_PATTERN)
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=SHA256_PATTERN)
    governed_rule_projection_ref: str = Field(alias="governedRuleProjectionRef")
    governed_rule_projection_hash: str = Field(alias="governedRuleProjectionHash", pattern=SHA256_PATTERN)
    ordered_governed_rule_version_refs: tuple[str, ...] = Field(alias="orderedGovernedRuleVersionRefs")
    ordered_governed_rule_version_set_hash: str = Field(alias="orderedGovernedRuleVersionSetHash", pattern=SHA256_PATTERN)
    activation_set_snapshot_ref: str = Field(alias="activationSetSnapshotRef")
    activation_set_snapshot_hash: str = Field(alias="activationSetSnapshotHash", pattern=SHA256_PATTERN)
    activation_resolver_policy_version: str = Field(alias="activationResolverPolicyVersion")
    resolved_activation_set_hash: str = Field(alias="resolvedActivationSetHash", pattern=SHA256_PATTERN)
    effective_context_resolver_hash: str = Field(alias="effectiveContextResolverHash", pattern=SHA256_PATTERN)
    effective_at: str = Field(alias="effectiveAt")

    @field_validator("ordered_governed_rule_version_refs", mode="before")
    @classmethod
    def validate_ordered_rule_refs(cls, value: Any) -> tuple[str, ...]:
        refs = _ordered_unique_refs(value, "orderedGovernedRuleVersionRefs")
        if not refs:
            raise ValueError("orderedGovernedRuleVersionRefs must not be empty")
        return refs

    @model_validator(mode="after")
    def validate_snapshot(self) -> "RuntimeSemanticRetrievalSnapshotV1":
        for field in (
            "organization_id", "workspace_id", "run_id", "accepted_graph_release_ref",
            "governed_rule_projection_ref", "activation_set_snapshot_ref",
            "activation_resolver_policy_version",
        ):
            _ascii_ref(getattr(self, field), field)
        _validate_json(self.query, "query")
        try:
            parsed = datetime.fromisoformat(self.effective_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("effectiveAt must be RFC3339 UTC") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("effectiveAt must be RFC3339 UTC")
        if tuple(item.rank for item in self.ordered_material) != tuple(
            range(1, len(self.ordered_material) + 1)
        ):
            raise ValueError("orderedMaterial ranks must be contiguous from one")
        if len({(item.kind, item.ref) for item in self.ordered_material}) != len(self.ordered_material):
            raise ValueError("orderedMaterial contains a duplicate governed identity")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.snapshot_hash != expected:
            raise ValueError("retrieval snapshot hash does not match canonical content")
        if self.snapshot_id != "rsn_" + expected[7:31]:
            raise ValueError("retrieval snapshot id does not match snapshot hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("snapshotId", None)
        payload.pop("snapshotHash", None)
        return payload


def create_runtime_semantic_retrieval_snapshot(
    *, organization_id: str, workspace_id: str, run_id: str,
    query: Mapping[str, Any], ordered_material: Sequence[Mapping[str, Any]],
    retrieval_policy_ref: str, retrieval_policy_hash: str,
    runtime_context: EffectiveRestrictionRuntimeContextV1,
) -> RuntimeSemanticRetrievalSnapshotV1:
    if (runtime_context.organization_id, runtime_context.workspace_id) != (organization_id, workspace_id):
        raise ValueError("runtime context organization/workspace scope mismatch")
    context = runtime_context.model_dump(mode="json", by_alias=True)
    payload = {
        "schemaVersion": "runtime_semantic_retrieval_snapshot.v1",
        "organizationId": organization_id,
        "workspaceId": workspace_id,
        "runId": run_id,
        "query": dict(query),
        "orderedMaterial": [dict(item) for item in ordered_material],
        "retrievalPolicyRef": retrieval_policy_ref,
        "retrievalPolicyHash": retrieval_policy_hash,
        **{key: value for key, value in context.items() if key not in {"schemaVersion", "organizationId", "workspaceId"}},
    }
    digest = "sha256:" + content_hash(payload)
    return RuntimeSemanticRetrievalSnapshotV1.model_validate(
        {**payload, "snapshotId": "rsn_" + digest[7:31], "snapshotHash": digest}
    )


class ObservedConstraintHypothesisV1(_Contract):
    schema_version: Literal["observed_constraint_hypothesis.v1"] = Field(
        default="observed_constraint_hypothesis.v1", alias="schemaVersion"
    )
    hypothesis_id: str = Field(alias="hypothesisId")
    hypothesis_hash: str = Field(alias="hypothesisHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    run_id: str = Field(alias="runId")
    source_binding: HypothesisSourceBinding = Field(alias="sourceBinding")
    evidence_refs: tuple[EvidenceRef, ...] = Field(alias="evidenceRefs", min_length=1)
    source_anchor_refs: tuple[str, ...] = Field(default=(), alias="sourceAnchorRefs")
    candidate_frames: tuple[dict[str, Any], ...] = Field(alias="candidateFrames", min_length=1)
    retrieved_pattern_refs: tuple[str, ...] = Field(alias="retrievedPatternRefs")
    retrieved_pattern_hashes: tuple[str, ...] = Field(alias="retrievedPatternHashes")
    assumptions: tuple[str, ...]
    supporting_evidence_refs: tuple[str, ...] = Field(alias="supportingEvidenceRefs")
    counter_evidence_refs: tuple[str, ...] = Field(alias="counterEvidenceRefs")
    uncertainty: dict[str, Any]
    provider_ref: str = Field(alias="providerRef")
    model_ref: str = Field(alias="modelRef")
    prompt_asset_ref: Literal["runtime_semantic_hypothesis.v1"] = Field(alias="promptAssetRef")
    prompt_asset_hash: str = Field(alias="promptAssetHash", pattern=SHA256_PATTERN)
    schema_ref: Literal["observed_constraint_hypothesis.v1"] = Field(alias="schemaRef")
    retrieval_snapshot_ref: str = Field(alias="retrievalSnapshotRef")
    retrieval_snapshot_hash: str = Field(alias="retrievalSnapshotHash", pattern=SHA256_PATTERN)
    accepted_graph_release_ref: str = Field(alias="acceptedGraphReleaseRef")
    accepted_graph_release_hash: str = Field(alias="acceptedGraphReleaseHash", pattern=SHA256_PATTERN)
    governed_rule_projection_ref: str = Field(alias="governedRuleProjectionRef")
    governed_rule_projection_hash: str = Field(alias="governedRuleProjectionHash", pattern=SHA256_PATTERN)
    activation_set_snapshot_ref: str = Field(alias="activationSetSnapshotRef")
    activation_set_snapshot_hash: str = Field(alias="activationSetSnapshotHash", pattern=SHA256_PATTERN)
    effective_context_resolver_hash: str = Field(alias="effectiveContextResolverHash", pattern=SHA256_PATTERN)
    lifecycle_state: Literal["proposed", "validated", "accepted", "rejected", "superseded"] = Field(
        alias="lifecycleState"
    )
    predecessor_hypothesis_ref: str | None = Field(default=None, alias="predecessorHypothesisRef")

    @field_validator(
        "source_anchor_refs",
        "retrieved_pattern_refs",
        "supporting_evidence_refs",
        "counter_evidence_refs",
        mode="before",
    )
    @classmethod
    def validate_ref_arrays(cls, value: Any, info) -> tuple[str, ...]:
        return _ordered_unique_refs(value, info.field_name)

    @field_validator("retrieved_pattern_hashes", mode="before")
    @classmethod
    def validate_pattern_hashes(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("retrievedPatternHashes must be an array")
        hashes = tuple(value)
        if any(not isinstance(item, str) or re.fullmatch(SHA256_PATTERN, item) is None for item in hashes):
            raise ValueError("retrievedPatternHashes contains an invalid hash")
        return hashes

    @model_validator(mode="after")
    def validate_contract(self) -> "ObservedConstraintHypothesisV1":
        for field in (
            "organization_id", "workspace_id", "run_id", "provider_ref", "model_ref",
            "prompt_asset_ref", "schema_ref", "retrieval_snapshot_ref",
            "accepted_graph_release_ref", "governed_rule_projection_ref",
            "activation_set_snapshot_ref",
        ):
            _ascii_ref(getattr(self, field), field)
        if self.predecessor_hypothesis_ref is not None:
            _ascii_ref(self.predecessor_hypothesis_ref, "predecessorHypothesisRef")
        if len(self.retrieved_pattern_refs) != len(self.retrieved_pattern_hashes):
            raise ValueError("retrieved pattern refs/hashes must be one-for-one")
        evidence_ids: set[str] = set()
        for evidence in self.evidence_refs:
            if evidence.organization_id != self.organization_id or evidence.workspace_id != self.workspace_id:
                raise ValueError("EvidenceRef organization/workspace scope mismatch")
            if evidence.source_identity_ref != self.source_binding.source_identity_ref:
                raise ValueError("EvidenceRef SourceIdentity mismatch")
            if evidence.selector_hash != evidence.selector.selector_hash():
                raise ValueError("EvidenceRef selector hash mismatch")
            if evidence.selector_schema_version != evidence.selector.schema_version:
                raise ValueError("EvidenceRef selector schema mismatch")
            if (
                evidence.created_at.tzinfo is None
                or evidence.created_at.utcoffset() is None
                or evidence.created_at.utcoffset().total_seconds() != 0
            ):
                raise ValueError("EvidenceRef.createdAt must be timezone-aware UTC")
            expected = evidence_ref_identity(
                organization_id=evidence.organization_id,
                workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash,
                evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected:
                raise ValueError("EvidenceRef identity mismatch")
            provenance = evidence.provenance
            if provenance.get("source_hash") != self.source_binding.source_hash:
                raise ValueError("EvidenceRef source provenance mismatch")
            if provenance.get("canonical_source_ir_ref") != self.source_binding.canonical_source_ir_ref or provenance.get("canonical_source_ir_hash") != self.source_binding.canonical_source_ir_hash:
                raise ValueError("EvidenceRef CanonicalSourceIR provenance mismatch")
            evidence_ids.add(evidence.evidence_ref)
        if len(evidence_ids) != len(self.evidence_refs):
            raise ValueError("evidenceRefs contains a duplicate evidenceRef identity")
        if not set(self.supporting_evidence_refs).issubset(evidence_ids) or not set(self.counter_evidence_refs).issubset(evidence_ids):
            raise ValueError("supporting/counter evidence refs must resolve within evidenceRefs")
        for frame in self.candidate_frames:
            keys = set(frame)
            if not _FRAME_REQUIRED.issubset(keys) or not keys.issubset(_FRAME_REQUIRED | _FRAME_OPTIONAL):
                raise ValueError("candidateFrames must satisfy the bounded constraint-frame schema")
            _validate_json(frame, "candidateFrames")
            for ref_key in _FRAME_REQUIRED - {"value_constraint"}:
                _ascii_ref(frame[ref_key], f"candidateFrames.{ref_key}")
        _validate_json(self.uncertainty, "uncertainty")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.hypothesis_hash != expected_hash:
            raise ValueError("hypothesis hash does not match canonical content")
        if _HYPOTHESIS_ID.fullmatch(self.hypothesis_id) is None or self.hypothesis_id != "och_" + expected_hash[7:31]:
            raise ValueError("hypothesis id does not match hypothesis hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("hypothesisId", None)
        payload.pop("hypothesisHash", None)
        payload.pop("sourceAnchorRefs", None)
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))


class ObservedConstraintHypothesisV2(ObservedConstraintHypothesisV1):
    """Additive observation-shape contract; v1 remains alternative-only."""

    schema_version: Literal["observed_constraint_hypothesis.v2"] = Field(
        default="observed_constraint_hypothesis.v2", alias="schemaVersion"
    )
    schema_ref: Literal["observed_constraint_hypothesis.v2"] = Field(alias="schemaRef")
    observation_shape: Literal["alternatives", "composed_frame_set"] = Field(
        alias="observationShape"
    )

    @model_validator(mode="after")
    def validate_observation_shape(self) -> "ObservedConstraintHypothesisV2":
        if self.observation_shape == "composed_frame_set" and len(self.candidate_frames) < 2:
            raise ValueError("composed_frame_set requires at least two observed frames")
        return self


class ObservedConstraintHypothesisV3(ObservedConstraintHypothesisV2):
    """Contextual judgment bound to the exact persisted executable projection."""

    schema_version: Literal["observed_constraint_hypothesis.v3"] = Field(
        default="observed_constraint_hypothesis.v3", alias="schemaVersion")
    schema_ref: Literal["observed_constraint_hypothesis.v3"] = Field(alias="schemaRef")
    prompt_asset_ref: Literal["runtime_semantic_hypothesis.v2"] = Field(alias="promptAssetRef")
    decision_projection_ref: str = Field(alias="decisionProjectionRef", min_length=1)
    decision_projection_hash: str = Field(alias="decisionProjectionHash", pattern=SHA256_PATTERN)
    judgment_request_hash: str = Field(alias="judgmentRequestHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_projection_identity(self) -> "ObservedConstraintHypothesisV3":
        if self.decision_projection_ref != "decision-local-projection:v2:" + self.decision_projection_hash[7:]:
            raise ValueError("judgment projection identity mismatch")
        return self


def create_observed_constraint_hypothesis(
    *,
    organization_id: str,
    workspace_id: str,
    run_id: str,
    source_binding: HypothesisSourceBinding,
    evidence_refs: Sequence[EvidenceRef],
    source_anchor_refs: Sequence[str],
    candidate_frames: Sequence[Mapping[str, Any]],
    retrieved_pattern_refs: Sequence[str],
    retrieved_pattern_hashes: Sequence[str],
    assumptions: Sequence[str],
    supporting_evidence_refs: Sequence[str],
    counter_evidence_refs: Sequence[str],
    uncertainty: Mapping[str, Any],
    provider_ref: str,
    model_ref: str,
    prompt_asset_ref: str,
    prompt_asset_hash: str,
    schema_ref: str,
    retrieval_snapshot_ref: str,
    retrieval_snapshot_hash: str,
    runtime_context: EffectiveRestrictionRuntimeContextV1,
    lifecycle_state: Literal["proposed", "validated", "accepted", "rejected", "superseded"],
    predecessor_hypothesis_ref: str | None = None,
    judgment_request: Mapping[str, Any] | None = None,
) -> ObservedConstraintHypothesisV1:
    if runtime_context.organization_id != organization_id or runtime_context.workspace_id != workspace_id:
        raise ValueError("runtime context organization/workspace scope mismatch")
    payload: dict[str, Any] = {
        "schemaVersion": "observed_constraint_hypothesis.v1",
        "organizationId": organization_id,
        "workspaceId": workspace_id,
        "runId": run_id,
        "sourceBinding": source_binding.model_dump(mode="json", by_alias=True),
        "evidenceRefs": [item.model_dump(mode="json", by_alias=True) for item in evidence_refs],
        "sourceAnchorRefs": list(source_anchor_refs),
        "candidateFrames": [dict(item) for item in candidate_frames],
        "retrievedPatternRefs": list(retrieved_pattern_refs),
        "retrievedPatternHashes": list(retrieved_pattern_hashes),
        "assumptions": list(assumptions),
        "supportingEvidenceRefs": list(supporting_evidence_refs),
        "counterEvidenceRefs": list(counter_evidence_refs),
        "uncertainty": dict(uncertainty),
        "providerRef": provider_ref,
        "modelRef": model_ref,
        "promptAssetRef": prompt_asset_ref,
        "promptAssetHash": prompt_asset_hash,
        "schemaRef": schema_ref,
        "retrievalSnapshotRef": retrieval_snapshot_ref,
        "retrievalSnapshotHash": retrieval_snapshot_hash,
        "acceptedGraphReleaseRef": runtime_context.accepted_graph_release_ref,
        "acceptedGraphReleaseHash": runtime_context.accepted_graph_release_hash,
        "governedRuleProjectionRef": runtime_context.governed_rule_projection_ref,
        "governedRuleProjectionHash": runtime_context.governed_rule_projection_hash,
        "activationSetSnapshotRef": runtime_context.activation_set_snapshot_ref,
        "activationSetSnapshotHash": runtime_context.activation_set_snapshot_hash,
        "effectiveContextResolverHash": runtime_context.effective_context_resolver_hash,
        "lifecycleState": lifecycle_state,
        "predecessorHypothesisRef": predecessor_hypothesis_ref,
    }
    contract = ObservedConstraintHypothesisV1
    if judgment_request is not None:
        if schema_ref != "observed_constraint_hypothesis.v3":
            raise ValueError("projection-bound judgment requires hypothesis v3")
        payload.update(schemaVersion="observed_constraint_hypothesis.v3", observationShape="alternatives",
                       decisionProjectionRef=judgment_request["decision_projection_ref"],
                       decisionProjectionHash=judgment_request["decision_projection_hash"],
                       judgmentRequestHash="sha256:" + content_hash(dict(judgment_request)))
        contract = ObservedConstraintHypothesisV3
    hash_payload = dict(payload)
    hash_payload.pop("sourceAnchorRefs")
    hypothesis_hash = "sha256:" + content_hash(hash_payload)
    return contract.model_validate(
        {
            **payload,
            "hypothesisId": "och_" + hypothesis_hash[7:31],
            "hypothesisHash": hypothesis_hash,
        }
    )


def create_observed_constraint_hypothesis_v2(
    *,
    observation_shape: Literal["alternatives", "composed_frame_set"],
    **kwargs: Any,
) -> ObservedConstraintHypothesisV2:
    if kwargs.get("schema_ref") != "observed_constraint_hypothesis.v2":
        raise ValueError("v2 hypothesis requires schema_ref observed_constraint_hypothesis.v2")
    v1 = create_observed_constraint_hypothesis(
        **{**kwargs, "schema_ref": "observed_constraint_hypothesis.v1"}
    )
    payload = v1.model_dump(mode="json", by_alias=True)
    payload.pop("hypothesisId")
    payload.pop("hypothesisHash")
    payload["schemaVersion"] = "observed_constraint_hypothesis.v2"
    payload["schemaRef"] = "observed_constraint_hypothesis.v2"
    payload["observationShape"] = observation_shape
    hash_payload = dict(payload)
    hash_payload.pop("sourceAnchorRefs")
    hypothesis_hash = "sha256:" + content_hash(hash_payload)
    return ObservedConstraintHypothesisV2.model_validate(
        {
            **payload,
            "hypothesisId": "och_" + hypothesis_hash[7:31],
            "hypothesisHash": hypothesis_hash,
        }
    )


__all__ = [
    "EffectiveRestrictionRuntimeContextV1",
    "HypothesisSourceBinding",
    "ObservedConstraintHypothesisV1",
    "ObservedConstraintHypothesisV2",
    "ObservedConstraintHypothesisV3",
    "RetrievalMaterialV1",
    "RuntimeSemanticRetrievalSnapshotV1",
    "create_observed_constraint_hypothesis",
    "create_observed_constraint_hypothesis_v2",
    "create_runtime_semantic_retrieval_snapshot",
]
