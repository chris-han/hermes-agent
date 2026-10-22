"""Non-authoritative target observations preceding policy retrieval."""
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from eos.artifact_hashing import content_hash


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PerceptionSpanV1(_Value):
    evidence_key: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    text: str = Field(min_length=1)


class RestrictionObservationV1(_Value):
    clause: PerceptionSpanV1
    classification: Literal["eligibility_condition", "other_condition", "description", "uncertain"]
    subject: tuple[PerceptionSpanV1, ...]
    restricted_population: tuple[PerceptionSpanV1, ...]
    criterion: tuple[PerceptionSpanV1, ...]
    scope: tuple[PerceptionSpanV1, ...]
    threshold: tuple[PerceptionSpanV1, ...]
    exception_cues: tuple[PerceptionSpanV1, ...]
    qualification_cues: tuple[PerceptionSpanV1, ...]
    temporal_cues: tuple[PerceptionSpanV1, ...]


class RestrictionPerceptionOutputV1(_Value):
    covered_evidence_keys: tuple[str, ...]
    observations: tuple[RestrictionObservationV1, ...]


class PerceptionQuoteOccurrenceV1(_Value):
    evidence_key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    occurrence: int = Field(ge=0, strict=True)


class RestrictionObservationProposalV1(_Value):
    clause: PerceptionQuoteOccurrenceV1
    classification: Literal["eligibility_condition", "other_condition", "description", "uncertain"]
    subject: tuple[PerceptionQuoteOccurrenceV1, ...]
    restricted_population: tuple[PerceptionQuoteOccurrenceV1, ...]
    criterion: tuple[PerceptionQuoteOccurrenceV1, ...]
    scope: tuple[PerceptionQuoteOccurrenceV1, ...]
    threshold: tuple[PerceptionQuoteOccurrenceV1, ...]
    exception_cues: tuple[PerceptionQuoteOccurrenceV1, ...]
    qualification_cues: tuple[PerceptionQuoteOccurrenceV1, ...]
    temporal_cues: tuple[PerceptionQuoteOccurrenceV1, ...]


class RestrictionPerceptionProposalV1(_Value):
    covered_evidence_keys: tuple[str, ...]
    observations: tuple[RestrictionObservationProposalV1, ...]


class _Artifact(_Value):
    artifact_ref: str
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: str
    created_at: datetime

    @model_validator(mode="after")
    def identity(self):
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("perception timestamps require aware UTC")
        digest = content_hash(self.model_dump(mode="json", exclude={"artifact_ref", "artifact_hash"}))
        artifact_stem = self.schema_version.rsplit(".v", 1)[0].replace("_", "-")
        if (self.artifact_ref, self.artifact_hash) != (artifact_stem + ":" + digest, "sha256:" + digest):
            raise ValueError("perception artifact hash mismatch")
        return self

    @classmethod
    def create(cls, **basis):
        basis = {"schema_version": cls.model_fields["schema_version"].default, **basis}
        digest = content_hash(basis)
        artifact_stem = basis["schema_version"].rsplit(".v", 1)[0].replace("_", "-")
        return cls.model_validate({**basis, "artifact_ref": artifact_stem + ":" + digest,
            "artifact_hash": "sha256:" + digest})


class RuntimeRestrictionPerceptionRequestV1(_Artifact):
    schema_version: Literal["runtime_restriction_perception_request.v1"] = "runtime_restriction_perception_request.v1"
    actor_ref: str
    source_region: RuntimeSemanticSourceRegionV1
    instructions: str
    input_text: str
    output_schema: dict
    execution_profile: dict


class RuntimeRestrictionPerceptionAttemptV1(_Artifact):
    schema_version: Literal["runtime_restriction_perception_attempt.v1"] = "runtime_restriction_perception_attempt.v1"
    request_pin: ArtifactPinV1
    compatibility: dict


class RuntimeRestrictionPerceptionResponseV1(_Artifact):
    schema_version: Literal["runtime_restriction_perception_response.v1"] = "runtime_restriction_perception_response.v1"
    request_pin: ArtifactPinV1
    attempt_pins: tuple[ArtifactPinV1, ...]
    result: dict | None
    error: str | None


class RuntimeRestrictionPerceptionResponseV2(RuntimeRestrictionPerceptionResponseV1):
    schema_version: Literal["runtime_restriction_perception_response.v2"] = "runtime_restriction_perception_response.v2"
    execution_elapsed_ns: int = Field(ge=0, strict=True)
