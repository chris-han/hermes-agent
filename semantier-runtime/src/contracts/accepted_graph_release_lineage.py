"""Exact origin lineage for AcceptedGraphRelease records from the v3 journey.

The immutable ``AcceptedGraphRelease`` payload remains unchanged.  This
content-addressed contract is persisted on the existing artifact-origin row.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from contracts.corpus_graph_schema import (
    V3_SCHEMA_VERSION,
    V4_SCHEMA_VERSION,
    CorpusGraphSchemaCandidate,
)
from contracts.accepted_graph_release_v2 import (
    AcceptedGraphReleaseV2,
    CanonicalPatternManifestV1,
    OccurrenceGroundingManifestV1,
)
from contracts.governed_extract import ExtractResult
from contracts.knowledge_graph_phase1 import (
    AcceptedGraphRelease,
    SensitiveTermCandidateGraph,
)
from contracts.schema_decision import (
    V2_SCHEMA_VERSION as SCHEMA_DECISION_V2,
    V3_SCHEMA_VERSION as SCHEMA_DECISION_V3,
    SchemaDecision,
)
from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "accepted_graph_release_lineage.v1"
V2_SCHEMA_VERSION = "accepted_graph_release_lineage.v2"
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIELDS = {
    "schema_version",
    "organization_id",
    "workspace_id",
    "lineage_id",
    "source_identity_ref",
    "source_hash",
    "source_version",
    "canonical_source_ir_ref",
    "canonical_source_ir_hash",
    "graph_intent_ref",
    "graph_intent_hash",
    "graph_intent_review_state",
    "schema_candidate_ref",
    "schema_candidate_hash",
    "schema_candidate_version",
    "schema_decision_ref",
    "schema_decision_hash",
    "schema_decision_version",
    "schema_validation_extract_run_ref",
    "schema_validation_extract_result_hash",
    "candidate_graph_materialization_extract_run_ref",
    "candidate_graph_id",
    "candidate_graph_hash",
    "accepted_graph_release_ref",
    "accepted_graph_release_hash",
    "orchestration_skill_ref",
    "compiler_capability_ref",
    "compiler_semantica_commit",
    "lineage_hash",
}
_V2_FIELDS = _FIELDS | {
    "occurrence_grounding_manifest_ref",
    "occurrence_grounding_manifest_hash",
    "canonical_pattern_manifest_ref",
    "canonical_pattern_manifest_hash",
}


def _required(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _machine_ref(value: Any, field: str) -> str:
    text = _required(value, field)
    if not text.isascii():
        raise ValueError(f"{field} must be ASCII-stable")
    return text


def _sha256_ref(value: Any, field: str) -> str:
    text = str(value or "")
    if _SHA256_REF.fullmatch(text) is None:
        raise ValueError(f"{field} must be a sha256 ref")
    return text


@dataclass(frozen=True)
class AcceptedGraphReleaseLineage:
    organization_id: str
    workspace_id: str
    lineage_id: str
    source_identity_ref: str
    source_hash: str
    source_version: str | None
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str
    graph_intent_ref: str
    graph_intent_hash: str
    graph_intent_review_state: str
    schema_candidate_ref: str
    schema_candidate_hash: str
    schema_candidate_version: str
    schema_decision_ref: str
    schema_decision_hash: str
    schema_decision_version: str
    schema_validation_extract_run_ref: str
    schema_validation_extract_result_hash: str
    candidate_graph_materialization_extract_run_ref: str
    candidate_graph_id: str
    candidate_graph_hash: str
    accepted_graph_release_ref: str
    accepted_graph_release_hash: str
    orchestration_skill_ref: str
    compiler_capability_ref: str
    compiler_semantica_commit: str
    lineage_hash: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for field in (
            "organization_id",
            "workspace_id",
            "lineage_id",
            "source_identity_ref",
            "canonical_source_ir_ref",
            "graph_intent_ref",
            "schema_candidate_ref",
            "schema_decision_ref",
            "schema_validation_extract_run_ref",
            "candidate_graph_materialization_extract_run_ref",
            "candidate_graph_id",
            "accepted_graph_release_ref",
            "compiler_semantica_commit",
        ):
            _machine_ref(getattr(self, field), field)
        if self.source_version is not None:
            _machine_ref(self.source_version, "source_version")
        for field in (
            "source_hash",
            "canonical_source_ir_hash",
            "graph_intent_hash",
            "schema_candidate_hash",
            "schema_decision_hash",
            "schema_validation_extract_result_hash",
            "candidate_graph_hash",
            "accepted_graph_release_hash",
            "lineage_hash",
        ):
            _sha256_ref(getattr(self, field), field)
        if self.graph_intent_review_state not in {"accepted", "edited"}:
            raise ValueError("graph_intent_review_state must be accepted or edited")
        if self.schema_candidate_version != V3_SCHEMA_VERSION:
            raise ValueError("schema_candidate_version must be v3")
        if self.schema_decision_version != SCHEMA_DECISION_V2:
            raise ValueError("schema_decision_version must be v2")
        if self.orchestration_skill_ref != "knowledge-graph-building@1.0.0":
            raise ValueError("orchestration_skill_ref is invalid")
        if self.compiler_capability_ref != "corpus-graph-schema-induction":
            raise ValueError("compiler_capability_ref is invalid")
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.lineage_hash != expected:
            raise ValueError("lineage_hash does not match release lineage content")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "lineage_id": self.lineage_id,
            "source_identity_ref": self.source_identity_ref,
            "source_hash": self.source_hash,
            "source_version": self.source_version,
            "canonical_source_ir_ref": self.canonical_source_ir_ref,
            "canonical_source_ir_hash": self.canonical_source_ir_hash,
            "graph_intent_ref": self.graph_intent_ref,
            "graph_intent_hash": self.graph_intent_hash,
            "graph_intent_review_state": self.graph_intent_review_state,
            "schema_candidate_ref": self.schema_candidate_ref,
            "schema_candidate_hash": self.schema_candidate_hash,
            "schema_candidate_version": self.schema_candidate_version,
            "schema_decision_ref": self.schema_decision_ref,
            "schema_decision_hash": self.schema_decision_hash,
            "schema_decision_version": self.schema_decision_version,
            "schema_validation_extract_run_ref": self.schema_validation_extract_run_ref,
            "schema_validation_extract_result_hash": self.schema_validation_extract_result_hash,
            "candidate_graph_materialization_extract_run_ref": (
                self.candidate_graph_materialization_extract_run_ref
            ),
            "candidate_graph_id": self.candidate_graph_id,
            "candidate_graph_hash": self.candidate_graph_hash,
            "accepted_graph_release_ref": self.accepted_graph_release_ref,
            "accepted_graph_release_hash": self.accepted_graph_release_hash,
            "orchestration_skill_ref": self.orchestration_skill_ref,
            "compiler_capability_ref": self.compiler_capability_ref,
            "compiler_semantica_commit": self.compiler_semantica_commit,
            "lineage_hash": self.lineage_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("lineage_hash")
        return payload

    @classmethod
    def create(
        cls,
        *,
        candidate: CorpusGraphSchemaCandidate,
        decision: SchemaDecision,
        extract_result: ExtractResult,
        candidate_graph: SensitiveTermCandidateGraph,
        accepted_release: dict[str, Any],
        candidate_graph_id: str | None = None,
    ) -> "AcceptedGraphReleaseLineage":
        source, intent, compiler = cls.validate_pre_release_inputs(
            candidate=candidate,
            decision=decision,
            extract_result=extract_result,
            candidate_graph=candidate_graph,
            candidate_graph_id=candidate_graph_id,
        )

        release_payload = dict(accepted_release)
        release_hash = _sha256_ref(
            release_payload.pop("release_hash", None),
            "accepted_graph_release_hash",
        )
        release = AcceptedGraphRelease.model_validate(release_payload)
        if release.release_hash() != release_hash:
            raise ValueError("accepted release hash mismatch")
        if release.candidate_graph_id != candidate_graph.candidate_graph_id:
            raise ValueError("accepted release candidate graph mismatch")

        payload = {
            "schema_version": SCHEMA_VERSION,
            "organization_id": source.get("organization_id"),
            "workspace_id": source.get("workspace_id"),
            "lineage_id": source.get("lineage_id"),
            "source_identity_ref": source.get("source_identity_ref"),
            "source_hash": source.get("source_hash"),
            "source_version": source.get("source_version"),
            "canonical_source_ir_ref": source.get("canonical_source_ir_ref"),
            "canonical_source_ir_hash": source.get("canonical_source_ir_hash"),
            "graph_intent_ref": intent.get("graph_intent_ref"),
            "graph_intent_hash": intent.get("graph_intent_hash"),
            "graph_intent_review_state": intent.get("review_state"),
            "schema_candidate_ref": candidate.candidate_id,
            "schema_candidate_hash": candidate.candidate_hash,
            "schema_candidate_version": candidate.schema_version,
            "schema_decision_ref": decision.schema_decision_id,
            "schema_decision_hash": decision.schema_decision_hash,
            "schema_decision_version": decision.schema_version,
            "schema_validation_extract_run_ref": extract_result.extract_run_id,
            "schema_validation_extract_result_hash": (
                extract_result.extract_result_hash
            ),
            "candidate_graph_materialization_extract_run_ref": (
                candidate_graph.extraction_run_id
            ),
            "candidate_graph_id": candidate_graph.candidate_graph_id,
            "candidate_graph_hash": candidate_graph.graph_hash(),
            "accepted_graph_release_ref": release.graph_version,
            "accepted_graph_release_hash": release_hash,
            "orchestration_skill_ref": compiler.get("orchestration_skill_ref"),
            "compiler_capability_ref": compiler.get("compiler_capability_ref"),
            "compiler_semantica_commit": compiler.get("semantica_commit"),
        }
        lineage_hash = f"sha256:{content_hash(payload)}"
        return cls.from_dict({**payload, "lineage_hash": lineage_hash})

    @classmethod
    def validate_pre_release_inputs(
        cls,
        *,
        candidate: CorpusGraphSchemaCandidate,
        decision: SchemaDecision,
        extract_result: ExtractResult,
        candidate_graph: SensitiveTermCandidateGraph,
        candidate_graph_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Validate the complete immutable chain before authority is materialized."""
        if candidate.schema_version != V3_SCHEMA_VERSION:
            raise ValueError("release lineage requires candidate v3")
        if decision.schema_version != SCHEMA_DECISION_V2:
            raise ValueError("release lineage requires schema decision v2")
        if extract_result.status != "succeeded":
            raise ValueError("release lineage requires a succeeded ExtractResult")

        source = dict(candidate.source_binding or {})
        intent = dict(candidate.graph_intent_binding or {})
        compiler = dict(candidate.compiler_provenance or {})
        if decision.source_binding != source:
            raise ValueError("SchemaDecision source pin mismatch")
        if decision.graph_intent_binding != intent:
            raise ValueError("SchemaDecision GraphIntent pin mismatch")
        if (
            decision.candidate_binding.get("candidate_id") != candidate.candidate_id
            or decision.candidate_binding.get("candidate_hash")
            != candidate.candidate_hash
        ):
            raise ValueError("SchemaDecision candidate pin mismatch")

        extract_source = extract_result.source_binding
        if (
            extract_source.get("lineage_id") != source.get("lineage_id")
            or extract_source.get("canonical_source_ir_ref")
            != source.get("canonical_source_ir_ref")
            or extract_source.get("canonical_source_hash") != source.get("source_hash")
        ):
            raise ValueError("Extract source pin mismatch")
        extract_schema = extract_result.schema_binding
        if (
            extract_schema.get("schema_decision_ref") != decision.schema_decision_id
            or extract_schema.get("schema_decision_hash")
            != decision.schema_decision_hash
        ):
            raise ValueError("Extract SchemaDecision pin mismatch")
        if (
            extract_schema.get("candidate_ref") != candidate.candidate_id
            or extract_schema.get("candidate_hash") != candidate.candidate_hash
        ):
            raise ValueError("Extract candidate pin mismatch")
        if extract_schema.get("graph_intent_binding") != intent:
            raise ValueError("Extract GraphIntent pin mismatch")
        if extract_schema.get("compiler_provenance") != compiler:
            raise ValueError("Extract compiler pin mismatch")
        exact_candidate_graph_id = (
            candidate_graph_id or candidate_graph.candidate_graph_id
        )
        if candidate_graph.candidate_graph_id != exact_candidate_graph_id:
            raise ValueError("candidate graph exact pin mismatch")
        return source, intent, compiler

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AcceptedGraphReleaseLineage":
        if set(payload) != _FIELDS:
            raise ValueError("release lineage payload must contain exact fields")
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            organization_id=str(payload.get("organization_id") or ""),
            workspace_id=str(payload.get("workspace_id") or ""),
            lineage_id=str(payload.get("lineage_id") or ""),
            source_identity_ref=str(payload.get("source_identity_ref") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            source_version=(
                str(payload["source_version"])
                if payload.get("source_version") is not None
                else None
            ),
            canonical_source_ir_ref=str(payload.get("canonical_source_ir_ref") or ""),
            canonical_source_ir_hash=str(payload.get("canonical_source_ir_hash") or ""),
            graph_intent_ref=str(payload.get("graph_intent_ref") or ""),
            graph_intent_hash=str(payload.get("graph_intent_hash") or ""),
            graph_intent_review_state=str(
                payload.get("graph_intent_review_state") or ""
            ),
            schema_candidate_ref=str(payload.get("schema_candidate_ref") or ""),
            schema_candidate_hash=str(payload.get("schema_candidate_hash") or ""),
            schema_candidate_version=str(payload.get("schema_candidate_version") or ""),
            schema_decision_ref=str(payload.get("schema_decision_ref") or ""),
            schema_decision_hash=str(payload.get("schema_decision_hash") or ""),
            schema_decision_version=str(payload.get("schema_decision_version") or ""),
            schema_validation_extract_run_ref=str(
                payload.get("schema_validation_extract_run_ref") or ""
            ),
            schema_validation_extract_result_hash=str(
                payload.get("schema_validation_extract_result_hash") or ""
            ),
            candidate_graph_materialization_extract_run_ref=str(
                payload.get("candidate_graph_materialization_extract_run_ref") or ""
            ),
            candidate_graph_id=str(payload.get("candidate_graph_id") or ""),
            candidate_graph_hash=str(payload.get("candidate_graph_hash") or ""),
            accepted_graph_release_ref=str(
                payload.get("accepted_graph_release_ref") or ""
            ),
            accepted_graph_release_hash=str(
                payload.get("accepted_graph_release_hash") or ""
            ),
            orchestration_skill_ref=str(payload.get("orchestration_skill_ref") or ""),
            compiler_capability_ref=str(payload.get("compiler_capability_ref") or ""),
            compiler_semantica_commit=str(
                payload.get("compiler_semantica_commit") or ""
            ),
            lineage_hash=str(payload.get("lineage_hash") or ""),
        )


@dataclass(frozen=True)
class AcceptedGraphReleaseLineageV2:
    """Unverified canonical v4/v3 lineage DTO for a relation-aware release."""

    organization_id: str
    workspace_id: str
    lineage_id: str
    source_identity_ref: str
    source_hash: str
    source_version: str | None
    canonical_source_ir_ref: str
    canonical_source_ir_hash: str
    graph_intent_ref: str
    graph_intent_hash: str
    graph_intent_review_state: str
    schema_candidate_ref: str
    schema_candidate_hash: str
    schema_candidate_version: str
    schema_decision_ref: str
    schema_decision_hash: str
    schema_decision_version: str
    schema_validation_extract_run_ref: str
    schema_validation_extract_result_hash: str
    candidate_graph_materialization_extract_run_ref: str
    candidate_graph_id: str
    candidate_graph_hash: str
    accepted_graph_release_ref: str
    accepted_graph_release_hash: str
    orchestration_skill_ref: str
    compiler_capability_ref: str
    compiler_semantica_commit: str
    occurrence_grounding_manifest_ref: str
    occurrence_grounding_manifest_hash: str
    canonical_pattern_manifest_ref: str
    canonical_pattern_manifest_hash: str
    lineage_hash: str
    schema_version: str = V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V2_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {V2_SCHEMA_VERSION}")
        for field in (
            "organization_id", "workspace_id", "lineage_id", "source_identity_ref",
            "canonical_source_ir_ref", "graph_intent_ref", "schema_candidate_ref",
            "schema_decision_ref", "schema_validation_extract_run_ref",
            "candidate_graph_materialization_extract_run_ref", "candidate_graph_id",
            "accepted_graph_release_ref", "compiler_semantica_commit",
            "occurrence_grounding_manifest_ref", "canonical_pattern_manifest_ref",
        ):
            _machine_ref(getattr(self, field), field)
        if self.source_version is not None:
            _machine_ref(self.source_version, "source_version")
        for field in (
            "source_hash", "canonical_source_ir_hash", "graph_intent_hash",
            "schema_candidate_hash", "schema_decision_hash",
            "schema_validation_extract_result_hash", "candidate_graph_hash",
            "accepted_graph_release_hash", "occurrence_grounding_manifest_hash",
            "canonical_pattern_manifest_hash", "lineage_hash",
        ):
            _sha256_ref(getattr(self, field), field)
        if self.graph_intent_review_state not in {"accepted", "edited"}:
            raise ValueError("graph_intent_review_state must be accepted or edited")
        if self.schema_candidate_version != V4_SCHEMA_VERSION:
            raise ValueError("release lineage v2 requires candidate v4")
        if self.schema_decision_version != SCHEMA_DECISION_V3:
            raise ValueError("release lineage v2 requires schema decision v3")
        if self.orchestration_skill_ref != "knowledge-graph-building@1.0.0":
            raise ValueError("orchestration_skill_ref is invalid")
        if self.compiler_capability_ref != "corpus-graph-schema-induction":
            raise ValueError("compiler_capability_ref is invalid")
        if not re.fullmatch(r"KG_v[1-9]\d*", self.accepted_graph_release_ref):
            raise ValueError("accepted_graph_release_ref must be an exact KG_vN ref")
        if self.lineage_hash != f"sha256:{content_hash(self._hash_payload())}":
            raise ValueError("lineage_hash does not match release lineage content")

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in (
            "schema_version", "organization_id", "workspace_id", "lineage_id",
            "source_identity_ref", "source_hash", "source_version",
            "canonical_source_ir_ref", "canonical_source_ir_hash", "graph_intent_ref",
            "graph_intent_hash", "graph_intent_review_state", "schema_candidate_ref",
            "schema_candidate_hash", "schema_candidate_version", "schema_decision_ref",
            "schema_decision_hash", "schema_decision_version",
            "schema_validation_extract_run_ref", "schema_validation_extract_result_hash",
            "candidate_graph_materialization_extract_run_ref", "candidate_graph_id",
            "candidate_graph_hash", "accepted_graph_release_ref",
            "accepted_graph_release_hash", "orchestration_skill_ref",
            "compiler_capability_ref", "compiler_semantica_commit",
            "occurrence_grounding_manifest_ref", "occurrence_grounding_manifest_hash",
            "canonical_pattern_manifest_ref", "canonical_pattern_manifest_hash",
            "lineage_hash",
        )}

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict(); payload.pop("lineage_hash"); return payload

    @classmethod
    def create_unverified(
        cls, *, candidate: CorpusGraphSchemaCandidate, decision: SchemaDecision,
        extract_result: ExtractResult, candidate_graph: SensitiveTermCandidateGraph,
        accepted_release: AcceptedGraphReleaseV2,
        occurrence_grounding_manifest: OccurrenceGroundingManifestV1,
        canonical_pattern_manifest: CanonicalPatternManifestV1,
        candidate_graph_id: str | None = None,
    ) -> "AcceptedGraphReleaseLineageV2":
        source, intent, compiler = cls.validate_pre_release_inputs(
            candidate=candidate, decision=decision, extract_result=extract_result,
            candidate_graph=candidate_graph, candidate_graph_id=candidate_graph_id,
        )
        if accepted_release.candidate_graph_id != candidate_graph.candidate_graph_id:
            raise ValueError("accepted release candidate graph mismatch")
        occurrence_grounding_manifest.assert_matches_release(accepted_release)
        canonical_pattern_manifest.assert_matches_release(accepted_release)
        if occurrence_grounding_manifest.organization_id != source.get("organization_id") or occurrence_grounding_manifest.workspace_id != source.get("workspace_id"):
            raise ValueError("occurrence Grounding manifest authority-context mismatch")
        if canonical_pattern_manifest.organization_id != source.get("organization_id") or canonical_pattern_manifest.workspace_id != source.get("workspace_id"):
            raise ValueError("canonical pattern manifest authority-context mismatch")
        if occurrence_grounding_manifest.schema_decision_ref != decision.schema_decision_id or occurrence_grounding_manifest.schema_decision_hash != decision.schema_decision_hash:
            raise ValueError("occurrence Grounding manifest SchemaDecision mismatch")
        if occurrence_grounding_manifest.extract_result_ref != extract_result.extract_run_id or occurrence_grounding_manifest.extract_result_hash != extract_result.extract_result_hash:
            raise ValueError("occurrence Grounding manifest ExtractResult mismatch")
        payload = {
            "schema_version": V2_SCHEMA_VERSION,
            "organization_id": source.get("organization_id"), "workspace_id": source.get("workspace_id"),
            "lineage_id": source.get("lineage_id"), "source_identity_ref": source.get("source_identity_ref"),
            "source_hash": source.get("source_hash"), "source_version": source.get("source_version"),
            "canonical_source_ir_ref": source.get("canonical_source_ir_ref"),
            "canonical_source_ir_hash": source.get("canonical_source_ir_hash"),
            "graph_intent_ref": intent.get("graph_intent_ref"), "graph_intent_hash": intent.get("graph_intent_hash"),
            "graph_intent_review_state": intent.get("review_state"),
            "schema_candidate_ref": candidate.candidate_id, "schema_candidate_hash": candidate.candidate_hash,
            "schema_candidate_version": candidate.schema_version, "schema_decision_ref": decision.schema_decision_id,
            "schema_decision_hash": decision.schema_decision_hash, "schema_decision_version": decision.schema_version,
            "schema_validation_extract_run_ref": extract_result.extract_run_id,
            "schema_validation_extract_result_hash": extract_result.extract_result_hash,
            "candidate_graph_materialization_extract_run_ref": candidate_graph.extraction_run_id,
            "candidate_graph_id": candidate_graph.candidate_graph_id, "candidate_graph_hash": candidate_graph.graph_hash(),
            "accepted_graph_release_ref": accepted_release.accepted_graph_release_ref,
            "accepted_graph_release_hash": accepted_release.release_hash,
            "orchestration_skill_ref": compiler.get("orchestration_skill_ref"),
            "compiler_capability_ref": compiler.get("compiler_capability_ref"),
            "compiler_semantica_commit": compiler.get("semantica_commit"),
            "occurrence_grounding_manifest_ref": occurrence_grounding_manifest.manifest_ref,
            "occurrence_grounding_manifest_hash": occurrence_grounding_manifest.manifest_hash,
            "canonical_pattern_manifest_ref": canonical_pattern_manifest.manifest_ref,
            "canonical_pattern_manifest_hash": canonical_pattern_manifest.manifest_hash,
        }
        return cls.parse_unverified(
            {**payload, "lineage_hash": f"sha256:{content_hash(payload)}"}
        )

    @classmethod
    def validate_pre_release_inputs(
        cls, *, candidate: CorpusGraphSchemaCandidate, decision: SchemaDecision,
        extract_result: ExtractResult, candidate_graph: SensitiveTermCandidateGraph,
        candidate_graph_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if candidate.schema_version != V4_SCHEMA_VERSION:
            raise ValueError("release lineage v2 requires candidate v4")
        if decision.schema_version != SCHEMA_DECISION_V3:
            raise ValueError("release lineage v2 requires schema decision v3")
        if extract_result.status != "succeeded":
            raise ValueError("release lineage requires a succeeded ExtractResult")
        source = dict(candidate.source_binding or {}); intent = dict(candidate.graph_intent_binding or {}); compiler = dict(candidate.compiler_provenance or {})
        if decision.source_binding != source: raise ValueError("SchemaDecision source pin mismatch")
        if decision.graph_intent_binding != intent: raise ValueError("SchemaDecision GraphIntent pin mismatch")
        if decision.candidate_binding.get("candidate_id") != candidate.candidate_id or decision.candidate_binding.get("candidate_hash") != candidate.candidate_hash: raise ValueError("SchemaDecision candidate pin mismatch")
        extract_source = extract_result.source_binding; extract_schema = extract_result.schema_binding
        if extract_source.get("lineage_id") != source.get("lineage_id") or extract_source.get("canonical_source_ir_ref") != source.get("canonical_source_ir_ref") or extract_source.get("canonical_source_hash") != source.get("source_hash"): raise ValueError("Extract source pin mismatch")
        if extract_schema.get("schema_decision_ref") != decision.schema_decision_id or extract_schema.get("schema_decision_hash") != decision.schema_decision_hash: raise ValueError("Extract SchemaDecision pin mismatch")
        if extract_schema.get("candidate_ref") != candidate.candidate_id or extract_schema.get("candidate_hash") != candidate.candidate_hash: raise ValueError("Extract candidate pin mismatch")
        extract_intent = extract_schema.get("graph_intent_binding")
        if extract_intent is not None and extract_intent != intent:
            raise ValueError("Extract GraphIntent pin mismatch")
        extract_compiler = extract_schema.get("compiler_provenance")
        if extract_compiler is not None:
            if extract_compiler != compiler:
                raise ValueError("Extract compiler pin mismatch")
        elif extract_schema.get("semantica_commit") != compiler.get("semantica_commit"):
            raise ValueError("Extract compiler pin mismatch")
        if candidate_graph.candidate_graph_id != (candidate_graph_id or candidate_graph.candidate_graph_id): raise ValueError("candidate graph exact pin mismatch")
        return source, intent, compiler

    @classmethod
    def parse_unverified(
        cls, payload: dict[str, Any]
    ) -> "AcceptedGraphReleaseLineageV2":
        if set(payload) != _V2_FIELDS:
            raise ValueError("release lineage v2 payload must contain exact fields")
        return cls(**payload)


def parse_accepted_graph_release_lineage_unverified(
    payload: dict[str, Any], lineage_contract: str
) -> AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2:
    """Dispatch canonical DTO parsing; this does not grant release authority."""
    if lineage_contract == "exact_v3":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("lineage discriminator does not match v1 payload")
        return AcceptedGraphReleaseLineage.from_dict(payload)
    if lineage_contract == "exact_v4_v3":
        if payload.get("schema_version") != V2_SCHEMA_VERSION:
            raise ValueError("lineage discriminator does not match v2 payload")
        return AcceptedGraphReleaseLineageV2.parse_unverified(payload)
    raise ValueError(f"unsupported release lineage discriminator: {lineage_contract}")


__all__ = [
    "SCHEMA_VERSION", "V2_SCHEMA_VERSION", "AcceptedGraphReleaseLineage",
    "AcceptedGraphReleaseLineageV2",
    "parse_accepted_graph_release_lineage_unverified",
]
