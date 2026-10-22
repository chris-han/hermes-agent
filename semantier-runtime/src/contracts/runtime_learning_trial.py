"""Minimal bounded empirical-episode contracts for AR-MVL."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta
from hashlib import sha256
import json
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.normative_semantics import NormativeFrameV1
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_paired_evaluation import (
    ArtifactPayloadPinV1,
    OntologyPairedObligationSuiteV1,
)
from contracts.runtime_context_reconstruction import (
    CapturedJsonV1,
    Sha256,
    UtcDateTime,
)
from eos.artifact_hashing import content_hash

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
ASCII_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$"
_JSON_POINTER = re.compile(r"^(?:/(?:[^~/]|~[01])*)*$")

CompilerResourceRoleV1 = Literal[
    "semantic_compiler",
    "validation_compiler",
    "rule_plan_compiler",
    "prompt",
    "configuration",
    "frame_schema",
    "profile",
]
TrialVariantKindV1 = Literal[
    "OMIT_EXCEPTION",
    "PROXY_THRESHOLD_20",
    "PROXY_THRESHOLD_10",
    "ADD_MISSING_FRAME",
]

COMPILER_RESOURCE_ROLES_V1: tuple[str, ...] = (
    "semantic_compiler",
    "validation_compiler",
    "rule_plan_compiler",
    "prompt",
    "configuration",
    "frame_schema",
    "profile",
)


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _aware_utc(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class _Addressed(_Contract):
    REF_FIELD: ClassVar[str]
    HASH_FIELD: ClassVar[str]
    PREFIX: ClassVar[str]

    @model_validator(mode="after")
    def _content_bound(self):
        digest = content_hash(
            self.model_dump(mode="json", exclude={self.REF_FIELD, self.HASH_FIELD})
        )
        if (
            getattr(self, self.REF_FIELD) != f"{self.PREFIX}:{digest}"
            or getattr(self, self.HASH_FIELD) != f"sha256:{digest}"
        ):
            raise ValueError("addressed artifact ref/hash does not bind exact content")
        return self

    @classmethod
    def create(cls, **values):
        probe = cls.model_construct(
            **{
                **values,
                cls.REF_FIELD: "pending",
                cls.HASH_FIELD: "sha256:" + "0" * 64,
            }
        )
        digest = content_hash(
            probe.model_dump(mode="json", exclude={cls.REF_FIELD, cls.HASH_FIELD})
        )
        return cls(
            **{
                **values,
                cls.REF_FIELD: f"{cls.PREFIX}:{digest}",
                cls.HASH_FIELD: f"sha256:{digest}",
            }
        )


class AdvisoryCompilerResourceV1(_Contract):
    role: CompilerResourceRoleV1
    resource_ref: str = Field(strict=True, pattern=ASCII_ID_PATTERN)
    bytes_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    bytes_base64: str = Field(strict=True)

    def decoded_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.bytes_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("bytes_base64 must use strict canonical base64") from exc

    @model_validator(mode="after")
    def _canonical_bytes_and_hash(self):
        raw = self.decoded_bytes()
        if base64.b64encode(raw).decode("ascii") != self.bytes_base64:
            raise ValueError("bytes_base64 must use strict canonical base64")
        if self.bytes_hash != "sha256:" + sha256(raw).hexdigest():
            raise ValueError("compiler resource hash must bind decoded bytes")
        return self


class AdvisoryCompilerInputsV1(_Addressed):
    REF_FIELD = "input_ref"
    HASH_FIELD = "input_hash"
    PREFIX = "advisory-compiler-inputs"

    schema_version: Literal["advisory_compiler_inputs.v1"] = (
        "advisory_compiler_inputs.v1"
    )
    input_ref: str = Field(strict=True, min_length=1)
    input_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    organization_id: str = Field(strict=True, min_length=1)
    workspace_id: str = Field(strict=True, min_length=1)
    source: ArtifactPinV1
    source_object_locator: str = Field(strict=True)
    competency: ArtifactPinV1
    boundary: ArtifactPinV1
    compiler_resources: tuple[AdvisoryCompilerResourceV1, ...]
    provider_ref: str = Field(strict=True, pattern=ASCII_ID_PATTERN)
    model_ref: str = Field(strict=True, pattern=ASCII_ID_PATTERN)
    oracle_locator: str = Field(strict=True)
    recorded_at: datetime

    @field_validator("source_object_locator", "oracle_locator")
    @classmethod
    def _rfc6901_pointer(cls, value: str) -> str:
        if _JSON_POINTER.fullmatch(value) is None:
            raise ValueError("locator must be an RFC6901 JSON pointer")
        return value

    @model_validator(mode="after")
    def _closed_native_inputs(self):
        if self.source.artifact_kind != "advisory_source_capture":
            raise ValueError("source must pin an advisory_source_capture")
        if self.competency.artifact_kind != "competency_contract":
            raise ValueError("competency must pin a competency_contract")
        if self.boundary.artifact_kind != "canonical_evidence_boundary":
            raise ValueError("boundary must pin a canonical_evidence_boundary")
        roles = tuple(resource.role for resource in self.compiler_resources)
        if roles != COMPILER_RESOURCE_ROLES_V1:
            raise ValueError("compiler resources must contain all seven roles in exact order")
        return self

    def resource(self, role: CompilerResourceRoleV1) -> AdvisoryCompilerResourceV1:
        return self.compiler_resources[COMPILER_RESOURCE_ROLES_V1.index(role)]


class AdvisoryTrialVariantV1(_Addressed):
    REF_FIELD = "variant_ref"
    HASH_FIELD = "variant_hash"
    PREFIX = "advisory-trial-variant"

    schema_version: Literal["advisory_trial_variant.v1"] = (
        "advisory_trial_variant.v1"
    )
    variant_ref: str = Field(strict=True, min_length=1)
    variant_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    organization_id: str = Field(strict=True, min_length=1)
    workspace_id: str = Field(strict=True, min_length=1)
    source: ArtifactPinV1
    inputs: ArtifactPinV1
    variant: TrialVariantKindV1
    candidate: ArtifactPinV1
    added_frame: NormativeFrameV1 | None = None
    recorded_at: datetime

    @model_validator(mode="after")
    def _closed_derivation(self):
        if self.source.artifact_kind != "normative_semantic_candidate_set":
            raise ValueError("source must pin a normative semantic candidate set")
        if self.inputs.artifact_kind != "advisory_compiler_inputs":
            raise ValueError("inputs must pin advisory compiler inputs")
        if self.candidate.artifact_kind != "normative_semantic_candidate_set":
            raise ValueError("candidate must pin a normative semantic candidate set")
        if self.source == self.candidate:
            raise ValueError("trial variant candidate must differ from its source")
        if (self.variant == "ADD_MISSING_FRAME") != (self.added_frame is not None):
            raise ValueError("added_frame is required only for ADD_MISSING_FRAME")
        return self


class AdvisoryTrialBuildInputsV1(_Addressed):
    REF_FIELD = "input_ref"
    HASH_FIELD = "input_hash"
    PREFIX = "advisory-trial-build-inputs"

    schema_version: Literal["advisory_trial_build_inputs.v1"] = (
        "advisory_trial_build_inputs.v1"
    )
    input_ref: str = Field(strict=True, min_length=1)
    input_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    organization_id: str = Field(strict=True, min_length=1)
    workspace_id: str = Field(strict=True, min_length=1)
    usage: Literal["CONTEXT_ONLY"] = "CONTEXT_ONLY"
    producer_ref: Literal[
        "repo:src/services/runtime_learning_trial_service.py#prepare-context-only-pair.v1"
    ] = "repo:src/services/runtime_learning_trial_service.py#prepare-context-only-pair.v1"
    producer_hash: Sha256
    side: Literal["BASELINE", "SUCCESSOR"]
    origin_manifest: ArtifactPayloadPinV1
    origin_snapshot: ArtifactPayloadPinV1
    predecessor_manifest: ArtifactPayloadPinV1
    predecessor_snapshot: ArtifactPayloadPinV1
    candidate: ArtifactPinV1
    derivation: ArtifactPinV1
    plan: ArtifactPinV1
    validation: ArtifactPinV1
    compiler_inputs: ArtifactPinV1
    recorded_at: UtcDateTime

    @model_validator(mode="after")
    def _closed_native_inputs(self):
        payload_prefixes = {
            "origin_manifest": "ontology-build:",
            "origin_snapshot": "exploratory-snapshot:",
            "predecessor_manifest": "ontology-build:",
            "predecessor_snapshot": "exploratory-snapshot:",
        }
        for field_name, prefix in payload_prefixes.items():
            if not getattr(self, field_name).artifact_ref.startswith(prefix):
                raise ValueError(f"{field_name} must pin its exact native artifact role")
        pin_kinds = {
            "candidate": "normative_semantic_candidate_set",
            "derivation": "advisory_trial_variant",
            "plan": "compiled_rule_plan",
            "validation": "validation_contract",
            "compiler_inputs": "advisory_compiler_inputs",
        }
        for field_name, artifact_kind in pin_kinds.items():
            if getattr(self, field_name).artifact_kind != artifact_kind:
                raise ValueError(f"{field_name} must pin {artifact_kind}")
        if self.side == "BASELINE" and (
            self.predecessor_manifest != self.origin_manifest
            or self.predecessor_snapshot != self.origin_snapshot
        ):
            raise ValueError("baseline predecessor must equal its origin")
        return self


class AdvisoryPairedSuiteReviewV1(_Addressed):
    """Nonauthorizing receipt for one exact native controlled-trial suite subject."""

    REF_FIELD = "decision_ref"
    HASH_FIELD = "decision_hash"
    PREFIX = "advisory-paired-suite-review"

    schema_version: Literal["advisory_paired_suite_review.v1"] = (
        "advisory_paired_suite_review.v1"
    )
    decision_ref: str = Field(strict=True, min_length=1)
    decision_hash: str = Field(strict=True, pattern=SHA256_PATTERN)
    organization_id: str = Field(strict=True, min_length=1)
    workspace_id: str = Field(strict=True, min_length=1)
    usage: Literal["CONTEXT_ONLY"] = "CONTEXT_ONLY"
    reviewer_kind: Literal["trial_preparation"] = "trial_preparation"
    producer_ref: Literal["runtime-learning-trial:prepare-suite:v1"] = (
        "runtime-learning-trial:prepare-suite:v1"
    )
    source: ArtifactPinV1
    suite_subject: CapturedJsonV1

    @model_validator(mode="after")
    def _closed_native_subject(self):
        if self.source.artifact_kind != "advisory_source_capture":
            raise ValueError("source must pin an advisory_source_capture")
        try:
            subject = json.loads(self.suite_subject.canonical_json)
            native_values = {
                **subject,
                "decision_ref": self.decision_ref,
                "decision_hash": self.decision_hash,
            }
            digest = content_hash(native_values)
            native = OntologyPairedObligationSuiteV1.model_validate(
                {
                    **native_values,
                    "suite_ref": f"ontology-paired-obligation-suite:{digest}",
                    "suite_hash": f"sha256:{digest}",
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("suite_subject must be an exact native suite projection") from exc
        projection = native.model_dump(
            mode="json",
            exclude={"suite_ref", "suite_hash", "decision_ref", "decision_hash"},
        )
        if projection != subject:
            raise ValueError("suite_subject must retain the complete native field set")
        if (
            native.reviewer_ref
            != "trial-preparation:controlled_internal_policy_v1"
            or native.review_state != "accepted"
        ):
            raise ValueError("suite_subject reviewer provenance is unsupported")
        return self


__all__ = [
    "AdvisoryCompilerInputsV1",
    "AdvisoryCompilerResourceV1",
    "AdvisoryPairedSuiteReviewV1",
    "AdvisoryTrialBuildInputsV1",
    "AdvisoryTrialVariantV1",
    "COMPILER_RESOURCE_ROLES_V1",
    "TrialVariantKindV1",
]
