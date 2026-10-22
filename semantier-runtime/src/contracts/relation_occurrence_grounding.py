"""Release-selection Grounding for one exact governed relation occurrence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from contracts.governed_extract import ExtractResult
from contracts.extracted_relation_occurrence import (
    ExtractedRelationOccurrenceV1,
    create_extracted_relation_occurrence,
)
from contracts.relation_usage_grounding import RelationUsageGroundingDecision
from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "relation_occurrence_grounding.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIELDS = {
    "schema_version",
    "grounding_record_id",
    "grounding_record_hash",
    "organization_id",
    "workspace_id",
    "extract_result_ref",
    "extract_result_hash",
    "schema_decision_ref",
    "schema_decision_hash",
    "extracted_occurrence_ref",
    "extracted_occurrence_hash",
    "outcome",
    "corrected_occurrence_ref",
    "corrected_occurrence_hash",
    "supersedes_occurrence_ref",
    "supersedes_occurrence_hash",
    "relation_usage_grounding_decision_ref",
    "relation_usage_grounding_decision_hash",
    "actor_ref",
    "decided_at",
}
_CORRECTION_FIELDS = (
    "corrected_occurrence_ref",
    "corrected_occurrence_hash",
    "supersedes_occurrence_ref",
    "supersedes_occurrence_hash",
    "relation_usage_grounding_decision_ref",
    "relation_usage_grounding_decision_hash",
)


def _ascii(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty ASCII-stable string")
    if not value.isascii():
        raise ValueError(f"{field} must be ASCII-stable")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact sha256 ref")
    return value


def _utc(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("decided_at must be timezone-aware UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("decided_at must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("decided_at must be timezone-aware UTC ISO-8601")
    return value


def _exact_occurrence(result: ExtractResult, occurrence_ref: str) -> dict[str, Any]:
    if result.status != "succeeded":
        raise ValueError("occurrence Grounding requires a succeeded ExtractResult")
    raw = result.output.get("relation_occurrences")
    if not isinstance(raw, list):
        raise ValueError(
            "ExtractResult lacks required relation_occurrences authority surface"
        )
    occurrences = [
        ExtractedRelationOccurrenceV1.model_validate(item) for item in raw
    ]
    matches = [
        item.model_dump(mode="json", by_alias=True)
        for item in occurrences
        if item.occurrence_ref == occurrence_ref
    ]
    if len(matches) != 1:
        raise ValueError("ExtractResult must contain exactly one extracted occurrence")
    return matches[0]


def _semantic_edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    source = str(edge.get("source_node_id") or "")
    target = str(edge.get("target_node_id") or "")
    if edge.get("direction") == "target_to_source":
        return target, source
    if edge.get("direction") == "undirected" and source.encode("utf-8") > target.encode("utf-8"):
        return target, source
    return source, target


@dataclass(frozen=True)
class RelationOccurrenceGroundingV1:
    grounding_record_id: str
    grounding_record_hash: str
    organization_id: str
    workspace_id: str
    extract_result_ref: str
    extract_result_hash: str
    schema_decision_ref: str
    schema_decision_hash: str
    extracted_occurrence_ref: str
    extracted_occurrence_hash: str
    outcome: str
    corrected_occurrence_ref: str | None
    corrected_occurrence_hash: str | None
    supersedes_occurrence_ref: str | None
    supersedes_occurrence_hash: str | None
    relation_usage_grounding_decision_ref: str | None
    relation_usage_grounding_decision_hash: str | None
    actor_ref: str
    decided_at: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        for field in (
            "organization_id",
            "workspace_id",
            "extract_result_ref",
            "schema_decision_ref",
            "extracted_occurrence_ref",
            "actor_ref",
        ):
            _ascii(getattr(self, field), field)
        for field in (
            "grounding_record_hash",
            "extract_result_hash",
            "schema_decision_hash",
            "extracted_occurrence_hash",
        ):
            _sha256(getattr(self, field), field)
        _utc(self.decided_at)
        if self.outcome not in {"accepted", "corrected", "rejected"}:
            raise ValueError("outcome must be accepted, corrected, or rejected")
        corrections = [getattr(self, field) for field in _CORRECTION_FIELDS]
        if self.outcome == "corrected":
            if any(value is None for value in corrections):
                raise ValueError("corrected outcome requires every correction field")
            for field in (
                "corrected_occurrence_ref",
                "supersedes_occurrence_ref",
                "relation_usage_grounding_decision_ref",
            ):
                _ascii(getattr(self, field), field)
            for field in (
                "corrected_occurrence_hash",
                "supersedes_occurrence_hash",
                "relation_usage_grounding_decision_hash",
            ):
                _sha256(getattr(self, field), field)
        elif any(value is not None for value in corrections):
            raise ValueError(
                "accepted and rejected outcomes require null correction fields"
            )
        expected_hash = f"sha256:{content_hash(self._hash_payload())}"
        if self.grounding_record_hash != expected_hash:
            raise ValueError("grounding_record_hash does not match record content")
        if self.grounding_record_id != f"rog_{expected_hash[7:31]}":
            raise ValueError("grounding_record_id does not match record content")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grounding_record_id": self.grounding_record_id,
            "grounding_record_hash": self.grounding_record_hash,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "extract_result_ref": self.extract_result_ref,
            "extract_result_hash": self.extract_result_hash,
            "schema_decision_ref": self.schema_decision_ref,
            "schema_decision_hash": self.schema_decision_hash,
            "extracted_occurrence_ref": self.extracted_occurrence_ref,
            "extracted_occurrence_hash": self.extracted_occurrence_hash,
            "outcome": self.outcome,
            "corrected_occurrence_ref": self.corrected_occurrence_ref,
            "corrected_occurrence_hash": self.corrected_occurrence_hash,
            "supersedes_occurrence_ref": self.supersedes_occurrence_ref,
            "supersedes_occurrence_hash": self.supersedes_occurrence_hash,
            "relation_usage_grounding_decision_ref": self.relation_usage_grounding_decision_ref,
            "relation_usage_grounding_decision_hash": self.relation_usage_grounding_decision_hash,
            "actor_ref": self.actor_ref,
            "decided_at": self.decided_at,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("grounding_record_id")
        payload.pop("grounding_record_hash")
        return payload

    @classmethod
    def create(
        cls,
        content: dict[str, Any],
        *,
        extract_result: ExtractResult,
        relation_usage_grounding_decision: (
            RelationUsageGroundingDecision | dict[str, Any] | None
        ) = None,
        relation_usage_authority: object | None = None,
    ) -> "RelationOccurrenceGroundingV1":
        payload = dict(content)
        payload["schema_version"] = SCHEMA_VERSION
        payload.pop("grounding_record_id", None)
        payload.pop("grounding_record_hash", None)
        payload.pop("extracted_occurrence_hash", None)
        for field in ("extract_result_hash", "schema_decision_hash"):
            _sha256(payload.get(field), field)
        occurrence_ref = _ascii(
            payload.get("extracted_occurrence_ref"), "extracted_occurrence_ref"
        )
        occurrence = _exact_occurrence(extract_result, occurrence_ref)
        payload["extracted_occurrence_hash"] = occurrence["occurrence_hash"]
        cls._validate_governed_inputs(
            payload,
            extract_result=extract_result,
            relation_usage_grounding_decision=relation_usage_grounding_decision,
            relation_usage_authority=relation_usage_authority,
            for_new_write=True,
        )
        digest = content_hash(payload)
        return cls._from_dict_structural(
            {
                **payload,
                "grounding_record_id": f"rog_{digest[:24]}",
                "grounding_record_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        extract_result: ExtractResult,
        relation_usage_grounding_decision: (
            RelationUsageGroundingDecision | dict[str, Any] | None
        ) = None,
        relation_usage_authority: object | None = None,
    ) -> "RelationOccurrenceGroundingV1":
        record = cls._from_dict_structural(payload)
        cls._validate_governed_inputs(
            record.as_dict(),
            extract_result=extract_result,
            relation_usage_grounding_decision=relation_usage_grounding_decision,
            relation_usage_authority=relation_usage_authority,
            for_new_write=False,
        )
        return record

    @staticmethod
    def _validate_governed_inputs(
        payload: dict[str, Any],
        *,
        extract_result: ExtractResult,
        relation_usage_grounding_decision: (
            RelationUsageGroundingDecision | dict[str, Any] | None
        ),
        relation_usage_authority: object | None,
        for_new_write: bool,
    ) -> None:
        if type(extract_result) is not ExtractResult:
            raise ValueError("occurrence Grounding requires exact ExtractResult")
        # Frozen dataclasses do not freeze nested dictionaries. Replay through the
        # existing parser so a caller cannot mutate output after initial parsing.
        extract_result = ExtractResult.from_dict(extract_result.as_dict())
        if (
            payload.get("extract_result_ref") != extract_result.extract_run_id
            or payload.get("extract_result_hash")
            != extract_result.extract_result_hash
        ):
            raise ValueError("occurrence Grounding ExtractResult pin mismatch")
        if (
            payload.get("schema_decision_ref")
            != extract_result.schema_binding.get("schema_decision_ref")
            or payload.get("schema_decision_hash")
            != extract_result.schema_binding.get("schema_decision_hash")
        ):
            raise ValueError("occurrence Grounding SchemaDecision pin mismatch")
        occurrence = _exact_occurrence(
            extract_result, str(payload.get("extracted_occurrence_ref") or "")
        )
        occurrence_hash = occurrence["occurrence_hash"]
        if payload.get("extracted_occurrence_hash") != occurrence_hash:
            raise ValueError("extracted occurrence hash mismatch")

        from services.relation_usage_grounding_service import (
            _resolve_relation_occurrence_grounding_inputs,
        )

        governed_result, governed_occurrence = (
            _resolve_relation_occurrence_grounding_inputs(
                relation_usage_authority,
                organization_id=str(payload.get("organization_id") or ""),
                workspace_id=str(payload.get("workspace_id") or ""),
                extract_result_ref=str(payload.get("extract_result_ref") or ""),
                extract_result_hash=str(payload.get("extract_result_hash") or ""),
                schema_decision_ref=str(payload.get("schema_decision_ref") or ""),
                schema_decision_hash=str(payload.get("schema_decision_hash") or ""),
                occurrence_ref=str(payload.get("extracted_occurrence_ref") or ""),
                occurrence_hash=str(payload.get("extracted_occurrence_hash") or ""),
                for_new_write=for_new_write,
            )
        )
        if governed_result.as_dict() != extract_result.as_dict():
            raise ValueError("caller ExtractResult differs from governed ExtractResult")
        if governed_occurrence != occurrence:
            raise ValueError("caller occurrence differs from governed occurrence")

        if payload.get("outcome") != "corrected":
            if relation_usage_grounding_decision is not None:
                raise ValueError("noncorrective outcome cannot bind a W2e decision")
            return
        if relation_usage_grounding_decision is None:
            raise ValueError("corrected outcome requires exact W2e decision")
        decision_payload = (
            relation_usage_grounding_decision.as_dict()
            if type(relation_usage_grounding_decision)
            is RelationUsageGroundingDecision
            else dict(relation_usage_grounding_decision)
        )
        decision = RelationUsageGroundingDecision.from_dict(
            decision_payload, authority=relation_usage_authority
        )
        corrected = decision.corrected_relation_instance
        corrected_binding = decision.corrected_relation_binding
        if corrected is None or corrected_binding is None:
            raise ValueError("corrected occurrence requires W2e occurrence correction")
        if corrected_binding["representation_kind"] != "binary_edge":
            raise ValueError(
                "corrected frame/derived occurrence requires a governed successor"
            )
        compatibility_edges = [
            dict(edge)
            for edge in (governed_result.output.get("edges") or [])
            if edge.get("relation_type_id") == occurrence["relation_type_id"]
            and _semantic_edge_endpoints(edge)
            == (occurrence["source_node_id"], occurrence["target_node_id"])
        ]
        if len(compatibility_edges) != 1:
            raise ValueError("corrected occurrence lacks exact compatibility edge")
        compatibility_edge = compatibility_edges[0]
        if (
            decision.subject_binding["subject_ref"]
            != compatibility_edge.get("edge_id")
            or decision.subject_binding["subject_hash"]
            != f"sha256:{content_hash(compatibility_edge)}"
        ):
            raise ValueError("W2e decision does not bind the occurrence projection")
        corrected_source = corrected.get("source_node_id")
        corrected_target = corrected.get("target_node_id")
        if corrected.get("direction") == "target_to_source":
            corrected_source, corrected_target = corrected_target, corrected_source
        corrected_occurrence = create_extracted_relation_occurrence(
            organization_id=decision.organization_id,
            workspace_id=decision.workspace_id,
            relation_type_id=str(corrected["relation_type_id"]),
            representation_kind=corrected_binding["representation_kind"],
            projection_shape=str(corrected_binding["projection_shape"]),
            source_node_id=corrected_source,
            target_node_id=corrected_target,
            frame_ref=None,
            projection_source_refs=[],
            evidence_refs=occurrence["evidence_refs"],
        )
        expected = {
            "organization_id": decision.organization_id,
            "workspace_id": decision.workspace_id,
            "extract_result_ref": decision.extract_run_binding["extract_run_ref"],
            "extract_result_hash": decision.extract_run_binding["extract_result_hash"],
            "schema_decision_ref": decision.schema_decision_binding[
                "schema_decision_ref"
            ],
            "schema_decision_hash": decision.schema_decision_binding[
                "schema_decision_hash"
            ],
            "extracted_occurrence_ref": occurrence["occurrence_ref"],
            "extracted_occurrence_hash": occurrence["occurrence_hash"],
            "corrected_occurrence_ref": corrected_occurrence.occurrence_ref,
            "corrected_occurrence_hash": corrected_occurrence.occurrence_hash,
            "supersedes_occurrence_ref": occurrence["occurrence_ref"],
            "supersedes_occurrence_hash": occurrence["occurrence_hash"],
            "relation_usage_grounding_decision_ref": decision.grounding_decision_ref,
            "relation_usage_grounding_decision_hash": decision.grounding_decision_hash,
        }
        for field, expected_value in expected.items():
            if payload.get(field) != expected_value:
                raise ValueError(
                    f"corrected occurrence {field} differs from exact W2e decision"
                )
        if decision.finding_kind != "occurrence_error":
            raise ValueError("corrected occurrence requires W2e occurrence correction")

    @classmethod
    def _from_dict_structural(
        cls, payload: dict[str, Any]
    ) -> "RelationOccurrenceGroundingV1":
        if not isinstance(payload, dict) or set(payload) != _FIELDS:
            raise ValueError(
                "relation occurrence grounding payload must contain exact fields"
            )
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


__all__ = ["RelationOccurrenceGroundingV1", "SCHEMA_VERSION"]
