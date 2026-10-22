"""Deterministic provenance records, separate from canonical domain relations."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.evidence_location import EvidenceRef
from contracts.knowledge_temporal_state import parse_utc
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_identity,
)


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ASCII_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")


class DerivationRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: Literal["derivation_record.v1"] = Field(
        default="derivation_record.v1", alias="schemaVersion"
    )
    derivation_ref: str = Field(alias="derivationRef")
    derivation_hash: str = Field(alias="derivationHash", pattern=SHA256_PATTERN)
    output_ref: str = Field(alias="outputRef")
    derivation_kind: Literal["extraction", "inference", "projection", "runtime_match"] = Field(
        alias="derivationKind"
    )
    rule_or_operation_ref: str = Field(alias="ruleOrOperationRef")
    premise_refs: tuple[str, ...] = Field(alias="premiseRefs")
    premise_order_matters: bool = Field(alias="premiseOrderMatters")
    evidence_refs: tuple[EvidenceRef, ...] = Field(alias="evidenceRefs")
    engine_version: str = Field(alias="engineVersion")
    derived_at: str = Field(alias="derivedAt")

    @field_validator("output_ref", "rule_or_operation_ref", "engine_version")
    @classmethod
    def stable_refs(cls, value: str, info) -> str:
        if _ASCII_REF.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be ASCII-stable")
        return value

    @model_validator(mode="before")
    @classmethod
    def canonicalize_arrays(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        aliases = "premiseRefs" if "premiseRefs" in value else "premise_refs"
        refs = tuple(value.get(aliases, ()))
        if not refs or any(
            not isinstance(ref, str) or _ASCII_REF.fullmatch(ref) is None for ref in refs
        ):
            raise ValueError("premiseRefs must contain ASCII-stable references")
        if len(refs) != len(set(refs)):
            raise ValueError("premiseRefs contains a duplicate reference")
        order_key = (
            "premiseOrderMatters"
            if "premiseOrderMatters" in value
            else "premise_order_matters"
        )
        if not value.get(order_key, False):
            refs = tuple(sorted(refs, key=lambda ref: ref.encode("utf-8")))
        value[aliases] = refs
        evidence_key = "evidenceRefs" if "evidenceRefs" in value else "evidence_refs"
        evidence = tuple(
            item if isinstance(item, EvidenceRef) else EvidenceRef.model_validate(item)
            for item in value.get(evidence_key, ())
        )
        value[evidence_key] = tuple(
            sorted(evidence, key=lambda item: item.evidence_ref.encode("utf-8"))
        )
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> "DerivationRecordV1":
        parse_utc(self.derived_at, "derived_at")
        identities: list[str] = []
        for evidence in self.evidence_refs:
            expected = evidence_ref_identity(
                organization_id=evidence.organization_id,
                workspace_id=evidence.workspace_id,
                source_identity_ref=evidence.source_identity_ref,
                selector_hash=evidence.selector_hash,
                evidence_role=evidence.evidence_role,
            )
            if evidence.evidence_ref != expected:
                raise ValueError("EvidenceRef identity mismatch")
            identities.append(evidence.evidence_ref)
        if len(identities) != len(set(identities)):
            raise ValueError("evidenceRefs contains a duplicate identity")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.derivation_hash != expected_hash:
            raise ValueError("derivation hash does not match canonical content")
        if self.derivation_ref != "derivation:" + expected_hash[7:31]:
            raise ValueError("derivation ref does not match derivation hash")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        payload.pop("derivationRef")
        payload.pop("derivationHash")
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))


def create_derivation_record(
    *,
    output_ref: str,
    derivation_kind: Literal["extraction", "inference", "projection", "runtime_match"],
    rule_or_operation_ref: str,
    premise_refs: list[str] | tuple[str, ...],
    premise_order_matters: bool,
    evidence_refs: list[EvidenceRef] | tuple[EvidenceRef, ...],
    engine_version: str,
    derived_at: str,
) -> DerivationRecordV1:
    canonical_premises = tuple(premise_refs)
    if not premise_order_matters:
        canonical_premises = tuple(
            sorted(canonical_premises, key=lambda ref: ref.encode("utf-8"))
        )
    canonical_evidence = tuple(
        sorted(evidence_refs, key=lambda item: item.evidence_ref.encode("utf-8"))
    )
    payload = {
        "schemaVersion": "derivation_record.v1",
        "outputRef": output_ref,
        "derivationKind": derivation_kind,
        "ruleOrOperationRef": rule_or_operation_ref,
        "premiseRefs": canonical_premises,
        "premiseOrderMatters": premise_order_matters,
        "evidenceRefs": canonical_evidence,
        "engineVersion": engine_version,
        "derivedAt": derived_at,
    }
    hash_payload = {
        **payload,
        "evidenceRefs": [
            item.model_dump(mode="json", by_alias=True) for item in canonical_evidence
        ],
    }
    derivation_hash = "sha256:" + content_hash(hash_payload)
    return DerivationRecordV1.model_validate(
        {
            **payload,
            "derivationRef": "derivation:" + derivation_hash[7:31],
            "derivationHash": derivation_hash,
        }
    )


__all__ = ["DerivationRecordV1", "create_derivation_record"]
