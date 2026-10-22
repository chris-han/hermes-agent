"""Canonical advisory graph-coherence analysis contracts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
DiagnosticCode = Literal[
    "POSSIBLE_ENTITY_FRAGMENTATION",
    "HIGH_SIGNIFICANCE_ISOLATED_COMPONENT",
    "HIGH_SIGNIFICANCE_ORPHAN_NODE",
    "FRAME_ROLE_INCOMPLETENESS",
    "RECURRING_UNMODELED_RELATION_PATTERN",
    "WEAK_EVIDENCE_NEIGHBORHOOD",
    "CONTRADICTORY_EVIDENCE_NEIGHBORHOOD",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _ascii(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError(f"{field} must be a non-empty ASCII-stable identifier")
    return value


def _sorted_unique(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    items = tuple(_ascii(item, f"{field}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{field} contains a duplicate identity")
    return tuple(sorted(items, key=lambda item: item.encode("utf-8")))


class GraphCoherenceDiagnosticV1(_Contract):
    diagnostic_ref: str
    diagnostic_code: DiagnosticCode
    affected_node_ids: tuple[str, ...]
    affected_entity_refs: tuple[str, ...]
    affected_relation_instance_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    diagnostic_basis: dict[str, Any]

    @field_validator("diagnostic_ref")
    @classmethod
    def stable_ref(cls, value: str) -> str:
        return _ascii(value, "diagnostic_ref")

    @field_validator(
        "affected_node_ids", "affected_entity_refs",
        "affected_relation_instance_refs", "evidence_refs", mode="before",
    )
    @classmethod
    def canonical_refs(cls, value: Any, info) -> tuple[str, ...]:
        return _sorted_unique(value, info.field_name)


class GraphRepairCandidateV1(_Contract):
    candidate_ref: str
    candidate_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_kind: Literal[
        "entity_resolution", "missing_relation", "schema_gap", "evidence_gap"
    ]
    diagnostic_code: DiagnosticCode
    affected_node_ids: tuple[str, ...]
    affected_entity_refs: tuple[str, ...]
    affected_relation_instance_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    diagnostic_basis: dict[str, Any]
    confidence: float | None
    authority: Literal["advisory"]

    @field_validator("candidate_ref")
    @classmethod
    def stable_ref(cls, value: str) -> str:
        return _ascii(value, "candidate_ref")

    @field_validator(
        "affected_node_ids", "affected_entity_refs",
        "affected_relation_instance_refs", "evidence_refs", mode="before",
    )
    @classmethod
    def canonical_refs(cls, value: Any, info) -> tuple[str, ...]:
        return _sorted_unique(value, info.field_name)

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("confidence must be null or between zero and one")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "GraphRepairCandidateV1":
        expected = "sha256:" + content_hash(self._hash_payload())
        if self.candidate_hash != expected:
            raise ValueError("candidate_hash does not match canonical candidate content")
        if self.candidate_ref != "grc_" + expected[7:31]:
            raise ValueError("candidate_ref does not match candidate_hash")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("candidate_ref")
        payload.pop("candidate_hash")
        return payload


class GraphCoherenceDiagnosticsV1(_Contract):
    identity_coherence: tuple[GraphCoherenceDiagnosticV1, ...]
    structural_coherence: tuple[GraphCoherenceDiagnosticV1, ...]
    schema_coherence: tuple[GraphCoherenceDiagnosticV1, ...]
    evidence_coherence: tuple[GraphCoherenceDiagnosticV1, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        return {
            key: tuple(sorted(payload.get(key, ()), key=_diagnostic_sort_key))
            for key in (
                "identity_coherence", "structural_coherence",
                "schema_coherence", "evidence_coherence",
            )
        }


class GraphCoherenceAnalysisV1(_Contract):
    schema_version: Literal["graph_coherence_analysis.v1"] = "graph_coherence_analysis.v1"
    analysis_version: Literal[1] = 1
    analysis_ref: str
    analysis_hash: str = Field(pattern=SHA256_PATTERN)
    graph_release_ref: str
    graph_release_hash: str = Field(pattern=SHA256_PATTERN)
    projection_ref: str
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    analyzer_version: str
    analyzer_policy_ref: str
    analyzer_policy_hash: str = Field(pattern=SHA256_PATTERN)
    diagnostics: GraphCoherenceDiagnosticsV1
    repair_candidates: tuple[GraphRepairCandidateV1, ...]

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        value = dict(payload)
        value["repair_candidates"] = tuple(sorted(
            value.get("repair_candidates", ()), key=_candidate_sort_key
        ))
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> "GraphCoherenceAnalysisV1":
        for field in (
            "analysis_ref", "graph_release_ref", "projection_ref",
            "analyzer_version", "analyzer_policy_ref",
        ):
            _ascii(getattr(self, field), field)
        expected = "sha256:" + content_hash(self._hash_payload())
        if self.analysis_hash != expected:
            raise ValueError("analysis_hash does not match canonical analysis content")
        if self.analysis_ref != "gca_" + expected[7:31]:
            raise ValueError("analysis_ref does not match analysis_hash")
        return self

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("analysis_ref")
        payload.pop("analysis_hash")
        return payload

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _diagnostic_sort_key(item: Any) -> bytes:
    ref = item.get("diagnostic_ref") if isinstance(item, dict) else item.diagnostic_ref
    return _ascii(ref, "diagnostic_ref").encode("utf-8")


def _candidate_sort_key(item: Any) -> bytes:
    ref = item.get("candidate_ref") if isinstance(item, dict) else item.candidate_ref
    return _ascii(ref, "candidate_ref").encode("utf-8")


def _create_candidate(payload: dict[str, Any]) -> GraphRepairCandidateV1:
    raw = dict(payload)
    raw.pop("candidate_ref", None)
    raw.pop("candidate_hash", None)
    for field in (
        "affected_node_ids", "affected_entity_refs",
        "affected_relation_instance_refs", "evidence_refs",
    ):
        raw[field] = _sorted_unique(raw.get(field, ()), field)
    provisional = GraphRepairCandidateV1.model_construct(
        **raw, candidate_ref="pending", candidate_hash="sha256:" + "0" * 64
    )
    candidate_hash = "sha256:" + content_hash(provisional._hash_payload())
    return GraphRepairCandidateV1.model_validate({
        **raw,
        "candidate_ref": "grc_" + candidate_hash[7:31],
        "candidate_hash": candidate_hash,
    })


def create_graph_coherence_analysis(content: dict[str, Any]) -> GraphCoherenceAnalysisV1:
    """Create a canonical content-addressed advisory analysis."""
    raw = {
        "schema_version": "graph_coherence_analysis.v1",
        "analysis_version": 1,
        **dict(content),
    }
    raw.pop("analysis_ref", None)
    raw.pop("analysis_hash", None)
    raw["repair_candidates"] = [
        _create_candidate(dict(item)) for item in raw.get("repair_candidates", ())
    ]
    raw["diagnostics"] = GraphCoherenceDiagnosticsV1.model_validate(raw["diagnostics"])
    raw["repair_candidates"] = tuple(
        sorted(raw["repair_candidates"], key=_candidate_sort_key)
    )
    provisional = GraphCoherenceAnalysisV1.model_construct(
        **raw,
        analysis_ref="pending",
        analysis_hash="sha256:" + "0" * 64,
    )
    analysis_hash = "sha256:" + content_hash(provisional._hash_payload())
    return GraphCoherenceAnalysisV1.model_validate({
        **provisional.as_dict(),
        "analysis_ref": "gca_" + analysis_hash[7:31],
        "analysis_hash": analysis_hash,
    })
