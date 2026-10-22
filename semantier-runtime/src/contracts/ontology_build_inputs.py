"""Immutable source-first inputs for offline audit; never execute archived code."""
from __future__ import annotations

import base64
from datetime import timedelta
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from contracts.canonical_source_ir import CanonicalSourceDocument, content_hash as ir_hash
from contracts.evidence_location import SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_audit import OntologyAuditArtifactV1
from eos.artifact_hashing import content_hash


class OntologyBuildInputEnvelopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["ontology_build_inputs.v1"] = "ontology_build_inputs.v1"
    envelope_ref: str
    envelope_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    graph_intent: dict[str, Any]
    source_identity: dict[str, Any]
    canonical_source_ir: dict[str, Any]
    source_bytes_base64: str
    compiler_version: str = Field(min_length=1)
    compiler_bytes_base64: str

    @model_validator(mode="after")
    def verify_content(self) -> "OntologyBuildInputEnvelopeV1":
        brief = GraphIntentBrief.from_dict(self.graph_intent)
        identity = SourceIdentity.model_validate(self.source_identity)
        canonical = CanonicalSourceDocument.model_validate(self.canonical_source_ir)
        binding = brief.source_binding
        if not brief.compiler_eligible:
            raise ValueError("archived GraphIntent must be compiler eligible")
        scope = (self.organization_id, self.workspace_id)
        if scope != (binding.organization_id, binding.workspace_id) or scope != (identity.organization_id, identity.workspace_id):
            raise ValueError("input envelope tenant scope mismatch")
        if identity.created_at.utcoffset() != timedelta(0):
            raise ValueError("archived source identity timestamp must be UTC")
        if (binding.source_identity_ref, binding.source_hash, binding.source_version) != (
            identity.source_identity_ref, identity.source_hash, identity.source_version
        ):
            raise ValueError("archived source identity does not match GraphIntent pins")
        source_bytes = base64.b64decode(self.source_bytes_base64, validate=True)
        if "sha256:" + sha256(source_bytes).hexdigest() != identity.source_hash:
            raise ValueError("archived source bytes hash mismatch")
        base64.b64decode(self.compiler_bytes_base64, validate=True)
        if canonical.source_identity_ref != identity.source_identity_ref:
            raise ValueError("archived canonical IR source identity mismatch")
        if ir_hash(self.canonical_source_ir) != binding.canonical_source_ir_hash:
            raise ValueError("archived canonical IR hash mismatch")
        locators = [locator for mapping in canonical.source_map.mappings for locator in mapping.source_locators]
        if not locators or any((locator.source_identity_ref, locator.source_hash) != (
            identity.source_identity_ref, identity.source_hash
        ) for locator in locators):
            raise ValueError("archived canonical IR locator binding mismatch")
        expected = "sha256:" + content_hash(self.model_dump(mode="json", exclude={"envelope_ref", "envelope_hash"}))
        if self.envelope_hash != expected or self.envelope_ref != f"ontology-build-inputs:{expected[7:]}":
            raise ValueError("input envelope ref/hash does not bind exact content")
        return self

    @classmethod
    def create(cls, **values: Any) -> "OntologyBuildInputEnvelopeV1":
        material = to_jsonable_python({"schema_version": "ontology_build_inputs.v1", **values})
        digest = content_hash(material)
        return cls(envelope_ref=f"ontology-build-inputs:{digest}", envelope_hash=f"sha256:{digest}", **values)

    def audit_inputs(self) -> tuple[OntologyAuditArtifactV1, ...]:
        current = type(self).model_validate(self.model_dump(mode="json"))
        brief = GraphIntentBrief.from_dict(current.graph_intent)
        binding = brief.source_binding
        compiler_bytes = base64.b64decode(current.compiler_bytes_base64, validate=True)
        # Ref/hash semantics belong to each original contract, not the wrapper.
        entries = (
            ("graph_intent_brief", brief.graph_intent_id, brief.graph_intent_hash,
             brief.schema_version, current.graph_intent),
            ("source_identity", binding.source_identity_ref,
             "sha256:" + content_hash(current.source_identity),
             "semantier.source_identity.v1", current.source_identity),
            ("canonical_source_ir", binding.canonical_source_ir_ref,
             binding.canonical_source_ir_hash, "canonical_source_ir.v1", current.canonical_source_ir),
            ("ontology_source_first_compiler", current.compiler_version,
             "sha256:" + sha256(compiler_bytes).hexdigest(), "ontology_source_first_compiler_source.v1",
             {"compiler_version": current.compiler_version, "source_bytes_base64": current.compiler_bytes_base64}),
        )
        return tuple(OntologyAuditArtifactV1(
            artifact_kind=kind, artifact_ref=ref, artifact_hash=digest, schema_version=schema, payload=payload
        ) for kind, ref, digest, schema, payload in entries)
