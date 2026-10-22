"""Replayable source-field observation for Tender document dates."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class TenderDocumentDateObservationV1(BaseModel):
    """Deterministic date observation; it carries source evidence, not authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tender_document_date_observation.v1"] = (
        "tender_document_date_observation.v1"
    )
    observation_ref: str = Field(min_length=1)
    observation_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    fact_ref: Literal["fact.document_date"] = "fact.document_date"
    status: Literal["OBSERVED", "UNKNOWN"]
    reason: Literal[
        "MISSING_OR_INCOMPLETE",
        "AMBIGUOUS_SOURCE_DATE",
        "INVALID_SOURCE_DATE",
    ] | None
    normalized_value: str | None
    raw_values: tuple[str, ...]
    source_identity: ArtifactPinV1
    canonical_source_ir: ArtifactPinV1
    normative_schema: ArtifactPinV1
    target_evidence: tuple[ArtifactPinV1, ...]
    fact_definition: ArtifactPinV1
    fact_definition_hash: str = Field(pattern=SHA256_PATTERN)
    selector: ArtifactPinV1
    normalizer: ArtifactPinV1
    producer: ArtifactPinV1
    observed_at: datetime

    @model_validator(mode="after")
    def exact_identity_and_status(self) -> "TenderDocumentDateObservationV1":
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() != timedelta(0):
            raise ValueError("observed_at must be timezone-aware UTC")
        if any(pin.artifact_kind != "evidence" for pin in self.target_evidence):
            raise ValueError("TARGET_EVIDENCE_REQUIRED")
        if self.source_identity.artifact_kind != "source_identity":
            raise ValueError("SOURCE_IDENTITY_REQUIRED")
        if self.canonical_source_ir.artifact_kind != "canonical_source_ir":
            raise ValueError("CANONICAL_SOURCE_IR_REQUIRED")
        if self.normative_schema.artifact_kind != "normative_schema_contract":
            raise ValueError("NORMATIVE_SCHEMA_REQUIRED")
        if self.fact_definition.artifact_kind != "normative_fact_definition":
            raise ValueError("NORMATIVE_FACT_DEFINITION_REQUIRED")
        if tuple(sorted(self.target_evidence, key=_pin_key)) != self.target_evidence:
            raise ValueError("TARGET_EVIDENCE_MUST_BE_CANONICAL")
        if len({_pin_key(pin) for pin in self.target_evidence}) != len(self.target_evidence):
            raise ValueError("TARGET_EVIDENCE_MUST_BE_UNIQUE")
        if self.status == "OBSERVED":
            if (
                self.reason is not None
                or self.normalized_value is None
                or len(self.raw_values) != 1
                or not self.target_evidence
            ):
                raise ValueError("OBSERVED_DATE_REQUIRES_ONE_NORMALIZED_VALUE")
            try:
                if date.fromisoformat(self.normalized_value).isoformat() != self.normalized_value:
                    raise ValueError("OBSERVED_DATE_MUST_BE_ISO")
            except ValueError as exc:
                raise ValueError("OBSERVED_DATE_MUST_BE_ISO") from exc
        elif self.reason is None or self.normalized_value is not None:
            raise ValueError("UNKNOWN_DATE_MUST_REMAIN_ABSENT")
        payload = self.model_dump(
            mode="json", exclude={"observation_ref", "observation_hash"}
        )
        digest = content_hash(payload)
        if (self.observation_ref, self.observation_hash) != (
            f"tender-document-date-observation:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("DOCUMENT_DATE_OBSERVATION_HASH_MISMATCH")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="tender_document_date_observation",
            artifact_ref=self.observation_ref,
            artifact_hash=self.observation_hash,
        )

    @classmethod
    def create(cls, **values) -> "TenderDocumentDateObservationV1":
        values = dict(values)
        values["target_evidence"] = tuple(
            sorted(tuple(values["target_evidence"]), key=_pin_key)
        )
        payload = {"schema_version": "tender_document_date_observation.v1", **values}
        probe = cls.model_construct(
            **payload,
            observation_ref="pending",
            observation_hash="sha256:" + "0" * 64,
        )
        digest = content_hash(
            probe.model_dump(
                mode="json", exclude={"observation_ref", "observation_hash"}
            )
        )
        return cls(
            **payload,
            observation_ref=f"tender-document-date-observation:{digest}",
            observation_hash=f"sha256:{digest}",
        )


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


__all__ = ["TenderDocumentDateObservationV1"]
