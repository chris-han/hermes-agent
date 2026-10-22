"""Governed source-semantic witness, review, and set-manifest contracts."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.evidence_location import EvidenceRef
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_identity


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MACHINE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
SourceRoleKind = Literal[
    "canonical_objects",
    "lexical_evidence",
    "reference_examples",
    "classifications",
    "source_assessments",
    "remediation_fields",
    "contextual_fields",
]
WitnessKind = Literal[
    "positive_exemplar", "hard_negative", "ambiguous_exemplar", "negative_exemplar"
]
RepresentationKind = Literal["binary_relation", "event", "frame", "composed_frame_set"]
ReviewState = Literal["proposed", "accepted", "edited", "rejected"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)


def _require_machine_id(value: str, field: str) -> str:
    if not value or not value.isascii() or _MACHINE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be an ASCII-stable machine identifier")
    return value


def _require_ascii_object_keys(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_machine_id(str(key), f"{field} key")
            _require_ascii_object_keys(nested, field)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _require_ascii_object_keys(nested, field)


def _validate_evidence_ref(
    evidence: EvidenceRef,
    *,
    organization_id: str,
    workspace_id: str,
    source_binding: "WitnessSourceBinding",
) -> None:
    _require_machine_id(evidence.evidence_ref, "EvidenceRef.evidenceRef")
    _require_machine_id(evidence.evidence_role, "EvidenceRef.evidenceRole")
    if evidence.organization_id != organization_id:
        raise ValueError("EvidenceRef organization scope mismatch")
    if evidence.workspace_id != workspace_id:
        raise ValueError("EvidenceRef workspace scope mismatch")
    if evidence.source_identity_ref != source_binding.source_identity_ref:
        raise ValueError("EvidenceRef SourceIdentity mismatch")
    if evidence.selector_schema_version != evidence.selector.schema_version:
        raise ValueError("EvidenceRef selector schema mismatch")
    if evidence.selector_hash != evidence.selector.selector_hash():
        raise ValueError("EvidenceRef selector hash mismatch")
    expected_evidence_ref = evidence_ref_identity(
        organization_id=evidence.organization_id,
        workspace_id=evidence.workspace_id,
        source_identity_ref=evidence.source_identity_ref,
        selector_hash=evidence.selector_hash,
        evidence_role=evidence.evidence_role,
    )
    if evidence.evidence_ref != expected_evidence_ref:
        raise ValueError("EvidenceRef identity mismatch")
    provenance = evidence.provenance
    expected = {
        "source_hash": source_binding.source_hash,
        "canonical_source_ir_ref": source_binding.canonical_source_ir_ref,
        "canonical_source_ir_hash": source_binding.canonical_source_ir_hash,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            label = "CanonicalSourceIR" if key.startswith("canonical_source_ir") else "source"
            raise ValueError(f"EvidenceRef {label} provenance mismatch")


class WitnessSourceBinding(_Contract):
    source_identity_schema_version: Literal["semantier.source_identity.v1"] = Field(
        alias="sourceIdentitySchemaVersion"
    )
    source_identity_ref: str = Field(alias="sourceIdentityRef", min_length=1)
    source_hash: str = Field(alias="sourceHash", pattern=SHA256_PATTERN)
    canonical_source_ir_ref: str = Field(alias="canonicalSourceIrRef", min_length=1)
    canonical_source_ir_hash: str = Field(alias="canonicalSourceIrHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def ascii_ids(self) -> "WitnessSourceBinding":
        _require_machine_id(self.source_identity_ref, "sourceIdentityRef")
        _require_machine_id(self.canonical_source_ir_ref, "canonicalSourceIrRef")
        return self


class SemanticExpectation(_Contract):
    representation_kind: RepresentationKind = Field(alias="representationKind")
    required_dimensions: tuple[str, ...] = Field(alias="requiredDimensions")
    expected_structure: dict[str, Any] = Field(alias="expectedStructure")
    allowed_variants: tuple[dict[str, Any], ...] = Field(alias="allowedVariants")

    @model_validator(mode="after")
    def validate_shape(self) -> "SemanticExpectation":
        for value in self.required_dimensions:
            _require_machine_id(value, "requiredDimensions[]")
        if self.representation_kind == "composed_frame_set":
            frames = self.expected_structure.get("frames")
            if not isinstance(frames, list) or len(frames) < 2:
                raise ValueError("composed_frame_set requires more than one expected frame")
        _require_ascii_object_keys(self.expected_structure, "expectedStructure")
        _require_ascii_object_keys(self.allowed_variants, "allowedVariants")
        return self


class WitnessReviewRecord(_Contract):
    state: ReviewState
    actor_ref: str = Field(alias="actorRef", min_length=1)
    review_ref: str = Field(alias="reviewRef", min_length=1)
    review_hash: str = Field(alias="reviewHash", pattern=SHA256_PATTERN)
    decision_evidence_refs: tuple[EvidenceRef, ...] = Field(alias="decisionEvidenceRefs")
    predecessor_witness_ref: str | None = Field(default=None, alias="predecessorWitnessRef")
    successor_witness_ref: str | None = Field(default=None, alias="successorWitnessRef")

    @model_validator(mode="after")
    def validate_review(self) -> "WitnessReviewRecord":
        _require_machine_id(self.actor_ref, "actorRef")
        _require_machine_id(self.review_ref, "reviewRef")
        if self.predecessor_witness_ref is not None:
            _require_machine_id(self.predecessor_witness_ref, "predecessorWitnessRef")
        if self.successor_witness_ref is not None:
            _require_machine_id(self.successor_witness_ref, "successorWitnessRef")
        expected = "sha256:" + content_hash(self._hash_payload())
        if self.review_hash != expected:
            raise ValueError("review hash does not match review content")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("reviewHash", None)
        payload.pop("predecessorWitnessRef", None)
        payload.pop("successorWitnessRef", None)
        return payload

    @classmethod
    def create(
        cls,
        *,
        state: ReviewState,
        actor_ref: str,
        review_ref_seed: str,
        decision_evidence_refs: Sequence[EvidenceRef] = (),
        predecessor_witness_ref: str | None = None,
        successor_witness_ref: str | None = None,
    ) -> "WitnessReviewRecord":
        payload = {
            "state": state,
            "actorRef": actor_ref,
            "reviewRef": review_ref_seed,
            "decisionEvidenceRefs": list(decision_evidence_refs),
            "predecessorWitnessRef": predecessor_witness_ref,
            "successorWitnessRef": successor_witness_ref,
        }
        hash_payload = dict(payload)
        hash_payload.pop("predecessorWitnessRef")
        hash_payload.pop("successorWitnessRef")
        hash_payload["decisionEvidenceRefs"] = [
            item.model_dump(mode="json", by_alias=True) for item in decision_evidence_refs
        ]
        return cls.model_validate(
            {**payload, "reviewHash": "sha256:" + content_hash(hash_payload)}
        )


class SourceSemanticWitnessV1(_Contract):
    schema_version: Literal["source_semantic_witness.v1"] = Field(
        default="source_semantic_witness.v1", alias="schemaVersion"
    )
    witness_id: str = Field(alias="witnessId", min_length=1)
    witness_hash: str = Field(alias="witnessHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId", min_length=1)
    workspace_id: str = Field(alias="workspaceId", min_length=1)
    source_binding: WitnessSourceBinding = Field(alias="sourceBinding")
    witness_kind: WitnessKind = Field(alias="witnessKind")
    discovered_source_role: SourceRoleKind = Field(alias="discoveredSourceRole")
    evidence_refs: tuple[EvidenceRef, ...] = Field(alias="evidenceRefs", min_length=1)
    source_anchor_refs: tuple[str, ...] = Field(default=(), alias="sourceAnchorRefs")
    exact_text_hash: str = Field(alias="exactTextHash", pattern=SHA256_PATTERN)
    semantic_expectation: SemanticExpectation = Field(alias="semanticExpectation")
    review: WitnessReviewRecord

    @model_validator(mode="after")
    def validate_contract(self) -> "SourceSemanticWitnessV1":
        for field in ("witness_id", "organization_id", "workspace_id"):
            _require_machine_id(getattr(self, field), field)
        for ref in self.source_anchor_refs:
            _require_machine_id(ref, "sourceAnchorRefs[]")
        for evidence in (*self.evidence_refs, *self.review.decision_evidence_refs):
            _validate_evidence_ref(
                evidence,
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                source_binding=self.source_binding,
            )
        expected = "sha256:" + content_hash(self._hash_payload())
        if self.witness_hash != expected:
            raise ValueError("witness hash does not match witness content")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("witnessHash", None)
        payload.pop("sourceAnchorRefs", None)
        review = dict(payload["review"])
        review.pop("predecessorWitnessRef", None)
        review.pop("successorWitnessRef", None)
        payload["review"] = review
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))

    @classmethod
    def create(
        cls,
        *,
        witness_id: str,
        organization_id: str,
        workspace_id: str,
        source_binding: WitnessSourceBinding,
        witness_kind: WitnessKind,
        discovered_source_role: SourceRoleKind,
        evidence_refs: Sequence[EvidenceRef],
        source_anchor_refs: Sequence[str],
        exact_text_hash: str,
        semantic_expectation: SemanticExpectation,
        review: WitnessReviewRecord,
    ) -> "SourceSemanticWitnessV1":
        payload = {
            "schemaVersion": "source_semantic_witness.v1",
            "witnessId": witness_id,
            "organizationId": organization_id,
            "workspaceId": workspace_id,
            "sourceBinding": source_binding,
            "witnessKind": witness_kind,
            "discoveredSourceRole": discovered_source_role,
            "evidenceRefs": list(evidence_refs),
            "sourceAnchorRefs": list(source_anchor_refs),
            "exactTextHash": exact_text_hash,
            "semanticExpectation": semantic_expectation,
            "review": review,
        }
        hash_payload = cls._hash_payload_from_input(payload)
        return cls.model_validate(
            {**payload, "witnessHash": "sha256:" + content_hash(hash_payload)}
        )

    @staticmethod
    def _hash_payload_from_input(payload: dict[str, Any]) -> dict[str, Any]:
        value = {
            key: (
                item.model_dump(mode="json", by_alias=True)
                if isinstance(item, BaseModel)
                else [entry.model_dump(mode="json", by_alias=True) if isinstance(entry, BaseModel) else entry for entry in item]
                if isinstance(item, (list, tuple))
                else item
            )
            for key, item in payload.items()
            if key not in {"witnessHash", "sourceAnchorRefs"}
        }
        review = dict(value["review"])
        review.pop("predecessorWitnessRef", None)
        review.pop("successorWitnessRef", None)
        value["review"] = review
        return value

    def create_successor(
        self,
        *,
        semantic_expectation: SemanticExpectation,
        review: WitnessReviewRecord,
    ) -> "SourceSemanticWitnessV1":
        if review.predecessor_witness_ref != self.witness_id:
            raise ValueError("successor review must reference predecessor witness")
        digest = sha256(
            canonical_json(
                {
                    "predecessor": self.witness_id,
                    "semantic_expectation": semantic_expectation.model_dump(mode="json", by_alias=True),
                    "review_ref": review.review_ref,
                }
            ).encode("utf-8")
        ).hexdigest()
        return self.create(
            witness_id=f"witness:{digest[:32]}",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            source_binding=self.source_binding,
            witness_kind=self.witness_kind,
            discovered_source_role=self.discovered_source_role,
            evidence_refs=self.evidence_refs,
            source_anchor_refs=self.source_anchor_refs,
            exact_text_hash=self.exact_text_hash,
            semantic_expectation=semantic_expectation,
            review=review,
        )


class SourceRoleFieldBinding(_Contract):
    role_kind: SourceRoleKind = Field(alias="roleKind")
    canonical_type_id: str = Field(alias="canonicalTypeId", min_length=1)
    header_evidence_ref: EvidenceRef = Field(alias="headerEvidenceRef")

    @model_validator(mode="after")
    def ascii_type(self) -> "SourceRoleFieldBinding":
        _require_machine_id(self.canonical_type_id, "canonicalTypeId")
        return self


class SourceRoleInterpretationEvidenceV1(_Contract):
    schema_version: Literal["source_role_interpretation_evidence.v1"] = Field(
        default="source_role_interpretation_evidence.v1", alias="schemaVersion"
    )
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    source_binding: WitnessSourceBinding = Field(alias="sourceBinding")
    field_bindings: tuple[SourceRoleFieldBinding, ...] = Field(alias="fieldBindings")
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_content(self) -> "SourceRoleInterpretationEvidenceV1":
        _require_machine_id(self.organization_id, "organizationId")
        _require_machine_id(self.workspace_id, "workspaceId")
        for binding in self.field_bindings:
            _validate_evidence_ref(
                binding.header_evidence_ref,
                organization_id=self.organization_id,
                workspace_id=self.workspace_id,
                source_binding=self.source_binding,
            )
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("contentHash")
        if self.content_hash != "sha256:" + content_hash(payload):
            raise ValueError("source-role evidence content hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        organization_id: str,
        workspace_id: str,
        source_identity_ref: str,
        source_hash: str,
        canonical_source_ir_ref: str,
        canonical_source_ir_hash: str,
        field_bindings: Sequence[SourceRoleFieldBinding],
    ) -> "SourceRoleInterpretationEvidenceV1":
        payload = {
            "schemaVersion": "source_role_interpretation_evidence.v1",
            "organizationId": organization_id,
            "workspaceId": workspace_id,
            "sourceBinding": WitnessSourceBinding(
                sourceIdentitySchemaVersion="semantier.source_identity.v1",
                sourceIdentityRef=source_identity_ref,
                sourceHash=source_hash,
                canonicalSourceIrRef=canonical_source_ir_ref,
                canonicalSourceIrHash=canonical_source_ir_hash,
            ),
            "fieldBindings": list(field_bindings),
        }
        normalized = {
            key: (
                item.model_dump(mode="json", by_alias=True)
                if isinstance(item, BaseModel)
                else [entry.model_dump(mode="json", by_alias=True) for entry in item]
                if isinstance(item, list)
                else item
            )
            for key, item in payload.items()
        }
        return cls.model_validate(
            {**payload, "contentHash": "sha256:" + content_hash(normalized)}
        )


class WitnessManifestEntry(_Contract):
    witness_ref: str = Field(alias="witnessRef")
    witness_hash: str = Field(alias="witnessHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def ascii_ref(self) -> "WitnessManifestEntry":
        _require_machine_id(self.witness_ref, "witnessRef")
        return self


class WitnessReviewManifestEntry(_Contract):
    witness_ref: str = Field(alias="witnessRef")
    review_ref: str = Field(alias="reviewRef")
    review_hash: str = Field(alias="reviewHash", pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def ascii_refs(self) -> "WitnessReviewManifestEntry":
        _require_machine_id(self.witness_ref, "witnessRef")
        _require_machine_id(self.review_ref, "reviewRef")
        return self


class SourceSemanticWitnessSetManifestV1(_Contract):
    schema_version: Literal["source_semantic_witness_set_manifest.v1"] = Field(
        default="source_semantic_witness_set_manifest.v1", alias="schemaVersion"
    )
    manifest_ref: str = Field(alias="manifestRef")
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)
    organization_id: str = Field(alias="organizationId")
    workspace_id: str = Field(alias="workspaceId")
    source_binding: WitnessSourceBinding = Field(alias="sourceBinding")
    witnesses: tuple[WitnessManifestEntry, ...]
    review_set_ref: str = Field(alias="reviewSetRef")
    ordered_reviews: tuple[WitnessReviewManifestEntry, ...] = Field(alias="orderedReviews")
    review_set_hash: str = Field(alias="reviewSetHash", pattern=SHA256_PATTERN)
    predecessor_manifest_ref: str | None = Field(default=None, alias="predecessorManifestRef")
    successor_manifest_ref: str | None = Field(default=None, alias="successorManifestRef")

    @model_validator(mode="after")
    def validate_manifest(self) -> "SourceSemanticWitnessSetManifestV1":
        for field in ("manifest_ref", "organization_id", "workspace_id", "review_set_ref"):
            _require_machine_id(getattr(self, field), field)
        for field in ("predecessor_manifest_ref", "successor_manifest_ref"):
            value = getattr(self, field)
            if value is not None:
                _require_machine_id(value, field)
        refs = [item.witness_ref for item in self.witnesses]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate witness refs are forbidden")
        review_refs = [item.witness_ref for item in self.ordered_reviews]
        if review_refs != refs:
            raise ValueError("review-set ordered witness refs mismatch")
        expected_review_set_hash = "sha256:" + content_hash(self._review_set_material())
        if self.review_set_hash != expected_review_set_hash:
            raise ValueError("review-set hash does not match ordered review material")
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("contentHash")
        if self.content_hash != "sha256:" + content_hash(payload):
            raise ValueError("manifest content hash mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        witnesses: Sequence[SourceSemanticWitnessV1],
        review_set_ref: str,
        predecessor_manifest_ref: str | None = None,
        successor_manifest_ref: str | None = None,
    ) -> "SourceSemanticWitnessSetManifestV1":
        if not witnesses:
            raise ValueError("manifest requires witnesses")
        first = witnesses[0]
        for witness in witnesses:
            if (
                witness.organization_id != first.organization_id
                or witness.workspace_id != first.workspace_id
                or witness.source_binding != first.source_binding
            ):
                raise ValueError("manifest witness scope/source mismatch")
        entries = [
            WitnessManifestEntry(witnessRef=item.witness_id, witnessHash=item.witness_hash)
            for item in witnesses
        ]
        review_entries = [
            WitnessReviewManifestEntry(
                witnessRef=item.witness_id,
                reviewRef=item.review.review_ref,
                reviewHash=item.review.review_hash,
            )
            for item in witnesses
        ]
        review_material = {
            "schemaVersion": "source_semantic_witness_review_set_material.v1",
            "organizationId": first.organization_id,
            "workspaceId": first.workspace_id,
            "sourceBinding": first.source_binding.model_dump(mode="json", by_alias=True),
            "reviewSetRef": review_set_ref,
            "orderedReviews": [
                item.model_dump(mode="json", by_alias=True) for item in review_entries
            ],
        }
        review_set_hash = "sha256:" + content_hash(review_material)
        ref_seed = content_hash(
            {
                "review_set_ref": review_set_ref,
                "review_set_hash": review_set_hash,
                "source_identity_ref": first.source_binding.source_identity_ref,
                "witnesses": [item.model_dump(mode="json", by_alias=True) for item in entries],
                "predecessor_manifest_ref": predecessor_manifest_ref,
            }
        )
        payload = {
            "schemaVersion": "source_semantic_witness_set_manifest.v1",
            "manifestRef": f"witness-set:{ref_seed[:32]}",
            "organizationId": first.organization_id,
            "workspaceId": first.workspace_id,
            "sourceBinding": first.source_binding,
            "witnesses": entries,
            "reviewSetRef": review_set_ref,
            "orderedReviews": review_entries,
            "reviewSetHash": review_set_hash,
            "predecessorManifestRef": predecessor_manifest_ref,
            "successorManifestRef": successor_manifest_ref,
        }
        normalized = {
            key: (
                item.model_dump(mode="json", by_alias=True)
                if isinstance(item, BaseModel)
                else [entry.model_dump(mode="json", by_alias=True) for entry in item]
                if isinstance(item, list)
                else item
            )
            for key, item in payload.items()
        }
        return cls.model_validate(
            {**payload, "contentHash": "sha256:" + content_hash(normalized)}
        )

    def verify_witnesses(self, witnesses: Sequence[SourceSemanticWitnessV1]) -> None:
        actual = tuple((item.witness_id, item.witness_hash) for item in witnesses)
        expected = tuple((item.witness_ref, item.witness_hash) for item in self.witnesses)
        if actual != expected:
            raise ValueError("witness ordered refs/hashes mismatch")
        actual_reviews = tuple(
            (item.witness_id, item.review.review_ref, item.review.review_hash)
            for item in witnesses
        )
        expected_reviews = tuple(
            (item.witness_ref, item.review_ref, item.review_hash)
            for item in self.ordered_reviews
        )
        if actual_reviews != expected_reviews:
            raise ValueError("review-set ordered refs/hashes mismatch")
        for witness in witnesses:
            if (
                witness.organization_id != self.organization_id
                or witness.workspace_id != self.workspace_id
                or witness.source_binding != self.source_binding
            ):
                raise ValueError("manifest witness scope/source mismatch")

    def _review_set_material(self) -> dict[str, Any]:
        return {
            "schemaVersion": "source_semantic_witness_review_set_material.v1",
            "organizationId": self.organization_id,
            "workspaceId": self.workspace_id,
            "sourceBinding": self.source_binding.model_dump(mode="json", by_alias=True),
            "reviewSetRef": self.review_set_ref,
            "orderedReviews": [
                item.model_dump(mode="json", by_alias=True)
                for item in self.ordered_reviews
            ],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))


__all__ = [
    "SemanticExpectation",
    "SourceRoleFieldBinding",
    "SourceRoleInterpretationEvidenceV1",
    "SourceSemanticWitnessSetManifestV1",
    "SourceSemanticWitnessV1",
    "WitnessManifestEntry",
    "WitnessReviewManifestEntry",
    "WitnessReviewRecord",
    "WitnessSourceBinding",
]
