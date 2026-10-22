"""Content-addressed Human Grounding evidence for relation-usage routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "relation_usage_grounding_decision.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_FINDING_ROUTES = {
    "occurrence_error": "occurrence_supersession",
    "repeated_type_selection_error": "schema_successor_required",
    "missing_predicate": "profile_catalog_proposal_required",
}
_FIELDS = {
    "schema_version",
    "grounding_decision_ref",
    "grounding_decision_hash",
    "organization_id",
    "workspace_id",
    "source_binding",
    "extract_run_binding",
    "schema_decision_binding",
    "finding_kind",
    "required_route",
    "subject_binding",
    "subject_relation_binding",
    "corrected_relation_binding",
    "original_relation_instance",
    "corrected_relation_instance",
    "schema_learning_feedback_ref",
    "reviewed_by",
    "occurred_at",
}
_SUBJECT_FIELDS = {"subject_kind", "subject_ref", "subject_hash"}
_SOURCE_BINDING_FIELDS = {
    "organization_id",
    "workspace_id",
    "lineage_id",
    "source_identity_ref",
    "source_hash",
    "source_version",
    "canonical_source_ir_ref",
    "canonical_source_ir_hash",
}
_RELATION_BINDING_FIELDS = {
    "relation_type_id",
    "family_id",
    "definition_scope",
    "namespace",
    "representation_kind",
    "projection_shape",
    "directed",
    "taxonomy_ref",
    "taxonomy_hash",
    "profile_ref",
    "profile_hash",
    "profile_relation_catalog_hash",
}
_CORRECTED_INSTANCE_FIELDS = {
    "relation_instance_ref",
    "relation_instance_hash",
    "relation_type_id",
    "source_node_id",
    "target_node_id",
    "direction",
    "evidence_refs",
    "authority_state",
    "supersedes_relation_instance_ref",
    "supersedes_relation_instance_hash",
}
_ORIGINAL_INSTANCE_FIELDS = {
    "edge_id",
    "relation_type_id",
    "source_node_id",
    "target_node_id",
    "direction",
    "evidence_refs",
    "authority_state",
}
_DIRECTED_ORIGINAL_INSTANCE_FIELDS = _ORIGINAL_INSTANCE_FIELDS | {
    "declared_source_type",
    "declared_target_type",
}
_OCCURRENCE_REPRESENTATIONS = {
    ("binary_edge", "binary"),
    ("frame_role", "frame"),
}


def _required(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    if not value.isascii():
        raise ValueError(f"{field} must be ASCII-stable")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be an exact sha256 ref")
    return value


def _utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("occurred_at must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("occurred_at must be timezone-aware UTC ISO-8601")
    return value


def _evidence_refs(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    refs = tuple(_required(item, f"{field}[]") for item in value)
    if len(refs) != len(set(refs)) or list(refs) != sorted(refs):
        raise ValueError(f"{field} must be sorted and unique")
    return refs


def _validate_source_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_BINDING_FIELDS:
        raise ValueError("source_binding must contain exact fields")
    source = dict(value)
    for field in (
        "organization_id",
        "workspace_id",
        "lineage_id",
        "source_identity_ref",
        "canonical_source_ir_ref",
    ):
        _required(source.get(field), f"source_binding.{field}")
    if source.get("source_version") is not None:
        _required(source.get("source_version"), "source_binding.source_version")
    for field in ("source_hash", "canonical_source_ir_hash"):
        _sha256(source.get(field), f"source_binding.{field}")
    return source


def _validate_original_instance(
    value: Any, relation_binding: dict[str, Any]
) -> dict[str, Any]:
    expected_fields = (
        _DIRECTED_ORIGINAL_INSTANCE_FIELDS
        if relation_binding["directed"]
        else _ORIGINAL_INSTANCE_FIELDS
    )
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("original relation instance must contain exact fields")
    original = dict(value)
    for field in (
        "edge_id",
        "relation_type_id",
        "source_node_id",
        "target_node_id",
        "direction",
    ):
        _required(original.get(field), f"original_relation_instance.{field}")
    if relation_binding["directed"]:
        for field in ("declared_source_type", "declared_target_type"):
            _required(original.get(field), f"original_relation_instance.{field}")
        if original["direction"] not in {"source_to_target", "target_to_source"}:
            raise ValueError("original relation instance direction is invalid")
    elif original["direction"] != "undirected":
        raise ValueError("original relation instance direction is invalid")
    _evidence_refs(original.get("evidence_refs"), "original_relation_instance.evidence_refs")
    if original.get("authority_state") != "candidate":
        raise ValueError("original relation instance must remain candidate")
    if (
        relation_binding["representation_kind"],
        relation_binding["projection_shape"],
    ) not in _OCCURRENCE_REPRESENTATIONS:
        raise ValueError("original relation instance representation is unsupported")
    if (
        not relation_binding["directed"]
        and original["source_node_id"].encode("utf-8")
        > original["target_node_id"].encode("utf-8")
    ):
        raise ValueError("undirected original relation endpoints are not canonical")
    return original


def _validate_corrected_instance(
    value: Any,
    relation_binding: dict[str, Any],
    subject_binding: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CORRECTED_INSTANCE_FIELDS:
        raise ValueError("corrected relation instance must contain exact fields")
    corrected = dict(value)
    for field in (
        "relation_instance_ref",
        "relation_type_id",
        "source_node_id",
        "target_node_id",
        "direction",
        "supersedes_relation_instance_ref",
    ):
        _required(corrected.get(field), f"corrected_relation_instance.{field}")
    for field in (
        "relation_instance_hash",
        "supersedes_relation_instance_hash",
    ):
        _sha256(corrected.get(field), f"corrected_relation_instance.{field}")
    _evidence_refs(
        corrected.get("evidence_refs"),
        "corrected_relation_instance.evidence_refs",
    )
    if corrected.get("authority_state") != "candidate":
        raise ValueError("corrected relation instance must remain candidate")
    if (
        relation_binding["representation_kind"],
        relation_binding["projection_shape"],
    ) not in _OCCURRENCE_REPRESENTATIONS:
        raise ValueError("corrected relation instance representation is unsupported")
    if relation_binding["directed"]:
        if corrected["direction"] not in {"source_to_target", "target_to_source"}:
            raise ValueError(
                "corrected relation instance direction does not match admitted type"
            )
    elif (
        corrected["direction"] != "undirected"
        or corrected["source_node_id"].encode("utf-8")
        > corrected["target_node_id"].encode("utf-8")
    ):
        raise ValueError(
            "corrected relation instance direction does not match admitted type"
        )
    if corrected["supersedes_relation_instance_ref"] != subject_binding.get(
        "subject_ref"
    ) or corrected["supersedes_relation_instance_hash"] != subject_binding.get(
        "subject_hash"
    ):
        raise ValueError("corrected relation instance must supersede exact subject")
    if corrected["relation_type_id"] != relation_binding.get("relation_type_id"):
        raise ValueError("corrected instance relation type binding mismatch")
    return corrected


def _relation_binding(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _RELATION_BINDING_FIELDS:
        raise ValueError(f"{field} must contain exact relation admission fields")
    result = dict(value)
    for name in (
        "relation_type_id",
        "family_id",
        "definition_scope",
        "namespace",
        "representation_kind",
        "projection_shape",
        "taxonomy_ref",
    ):
        _required(result.get(name), f"{field}.{name}")
    if type(result.get("directed")) is not bool:
        raise ValueError(f"{field}.directed must be a boolean")
    _sha256(result.get("taxonomy_hash"), f"{field}.taxonomy_hash")
    profile_pins = (
        result.get("profile_ref"),
        result.get("profile_hash"),
        result.get("profile_relation_catalog_hash"),
    )
    if result["definition_scope"] == "canonical":
        if any(pin is not None for pin in profile_pins):
            raise ValueError(f"{field} canonical relation requires null profile pins")
    elif result["definition_scope"] == "profile_defined":
        if any(pin is None for pin in profile_pins):
            raise ValueError(f"{field} profile-defined relation requires exact profile pins")
        _required(result["profile_ref"], f"{field}.profile_ref")
        _sha256(result["profile_hash"], f"{field}.profile_hash")
        _sha256(
            result["profile_relation_catalog_hash"],
            f"{field}.profile_relation_catalog_hash",
        )
    else:
        raise ValueError(f"{field}.definition_scope is invalid")
    return result


@dataclass(frozen=True)
class RelationUsageGroundingDecision:
    grounding_decision_ref: str
    grounding_decision_hash: str
    organization_id: str
    workspace_id: str
    source_binding: dict[str, Any]
    extract_run_binding: dict[str, Any]
    schema_decision_binding: dict[str, Any]
    finding_kind: str
    required_route: str
    subject_binding: dict[str, Any]
    subject_relation_binding: dict[str, Any] | None
    corrected_relation_binding: dict[str, Any] | None
    original_relation_instance: dict[str, Any] | None
    corrected_relation_instance: dict[str, Any] | None
    schema_learning_feedback_ref: str
    reviewed_by: str
    occurred_at: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        _required(self.organization_id, "organization_id")
        _required(self.workspace_id, "workspace_id")
        _required(self.reviewed_by, "reviewed_by")
        _required(self.schema_learning_feedback_ref, "schema_learning_feedback_ref")
        if not self.schema_learning_feedback_ref.startswith("schema-feedback-"):
            raise ValueError("schema_learning_feedback_ref is invalid")
        _utc(self.occurred_at)
        source_binding = _validate_source_binding(self.source_binding)
        if source_binding.get("organization_id") != self.organization_id:
            raise ValueError("source binding organization mismatch")
        if source_binding.get("workspace_id") != self.workspace_id:
            raise ValueError("source binding workspace mismatch")
        if set(self.extract_run_binding) != {"extract_run_ref", "extract_result_hash"}:
            raise ValueError("extract_run_binding must contain exact fields")
        _required(self.extract_run_binding.get("extract_run_ref"), "extract_run_ref")
        _sha256(self.extract_run_binding.get("extract_result_hash"), "extract_result_hash")
        if set(self.schema_decision_binding) != {
            "schema_decision_ref",
            "schema_decision_hash",
        }:
            raise ValueError("schema_decision_binding must contain exact fields")
        _required(
            self.schema_decision_binding.get("schema_decision_ref"),
            "schema_decision_ref",
        )
        _sha256(
            self.schema_decision_binding.get("schema_decision_hash"),
            "schema_decision_hash",
        )
        if set(self.subject_binding) != _SUBJECT_FIELDS:
            raise ValueError("subject_binding must contain exact fields")
        if self.subject_binding.get("subject_kind") not in {
            "candidate_relation_instance",
            "unresolved_relation_evidence",
        }:
            raise ValueError("subject_kind is invalid")
        _required(self.subject_binding.get("subject_ref"), "subject_ref")
        _sha256(self.subject_binding.get("subject_hash"), "subject_hash")
        subject_relation_binding = _relation_binding(
            self.subject_relation_binding, "subject_relation_binding"
        )
        corrected_relation_binding = _relation_binding(
            self.corrected_relation_binding, "corrected_relation_binding"
        )
        expected_route = _FINDING_ROUTES.get(self.finding_kind)
        if self.finding_kind == "family_definition_error":
            if subject_relation_binding is None:
                raise ValueError("family definition error requires admitted relation binding")
            expected_route = (
                "taxonomy_definition_successor_required"
                if subject_relation_binding["definition_scope"] == "canonical"
                else "profile_definition_successor_required"
            )
        if expected_route is None or self.required_route != expected_route:
            raise ValueError("finding kind and required route do not match")
        if self.finding_kind == "missing_predicate":
            if self.subject_binding["subject_kind"] != "unresolved_relation_evidence":
                raise ValueError("missing predicate requires unresolved relation evidence")
            if subject_relation_binding is not None:
                raise ValueError("missing predicate cannot claim an admitted relation binding")
        else:
            if self.subject_binding["subject_kind"] != "candidate_relation_instance":
                raise ValueError("admitted relation finding requires candidate instance")
            if subject_relation_binding is None or self.original_relation_instance is None:
                raise ValueError("admitted relation finding requires exact original instance")
            original_relation_instance = _validate_original_instance(
                self.original_relation_instance, subject_relation_binding
            )
            if original_relation_instance.get("edge_id") != self.subject_binding.get(
                "subject_ref"
            ):
                raise ValueError("original relation instance ref mismatch")
            if f"sha256:{content_hash(original_relation_instance)}" != (
                self.subject_binding.get("subject_hash")
            ):
                raise ValueError("original relation instance hash mismatch")
            if original_relation_instance.get("relation_type_id") != (
                subject_relation_binding.get("relation_type_id")
            ):
                raise ValueError("original relation instance admission mismatch")
        if self.finding_kind == "occurrence_error":
            if corrected_relation_binding is None or self.corrected_relation_instance is None:
                raise ValueError("occurrence error requires corrected candidate instance")
            if (
                corrected_relation_binding["representation_kind"],
                corrected_relation_binding["projection_shape"],
            ) != (
                subject_relation_binding["representation_kind"],
                subject_relation_binding["projection_shape"],
            ):
                raise ValueError(
                    "corrected relation instance requires unchanged representation"
                )
            corrected = _validate_corrected_instance(
                self.corrected_relation_instance,
                corrected_relation_binding,
                self.subject_binding,
            )
            relation_hash = _sha256(
                corrected.get("relation_instance_hash"), "relation_instance_hash"
            )
            identity_payload = dict(corrected)
            identity_payload.pop("relation_instance_ref")
            identity_payload.pop("relation_instance_hash")
            expected_hash = f"sha256:{content_hash(identity_payload)}"
            if relation_hash != expected_hash:
                raise ValueError("corrected relation instance hash mismatch")
            if corrected.get("relation_instance_ref") != (
                f"grounded-relation-candidate-{expected_hash[7:31]}"
            ):
                raise ValueError("corrected relation instance ref mismatch")
        elif (
            self.corrected_relation_binding is not None
            or self.corrected_relation_instance is not None
        ):
            raise ValueError("systematic route cannot create a corrected occurrence")
        expected_hash = f"sha256:{content_hash(self._hash_payload())}"
        if self.grounding_decision_hash != expected_hash:
            raise ValueError("grounding_decision_hash does not match decision content")
        if self.grounding_decision_ref != f"relation-usage-grounding-{expected_hash[7:31]}":
            raise ValueError("grounding_decision_ref does not match decision content")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grounding_decision_ref": self.grounding_decision_ref,
            "grounding_decision_hash": self.grounding_decision_hash,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "source_binding": dict(self.source_binding),
            "extract_run_binding": dict(self.extract_run_binding),
            "schema_decision_binding": dict(self.schema_decision_binding),
            "finding_kind": self.finding_kind,
            "required_route": self.required_route,
            "subject_binding": dict(self.subject_binding),
            "subject_relation_binding": (
                dict(self.subject_relation_binding)
                if self.subject_relation_binding is not None
                else None
            ),
            "corrected_relation_binding": (
                dict(self.corrected_relation_binding)
                if self.corrected_relation_binding is not None
                else None
            ),
            "original_relation_instance": (
                dict(self.original_relation_instance)
                if self.original_relation_instance is not None
                else None
            ),
            "corrected_relation_instance": (
                dict(self.corrected_relation_instance)
                if self.corrected_relation_instance is not None
                else None
            ),
            "schema_learning_feedback_ref": self.schema_learning_feedback_ref,
            "reviewed_by": self.reviewed_by,
            "occurred_at": self.occurred_at,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("grounding_decision_ref")
        payload.pop("grounding_decision_hash")
        return payload

    @staticmethod
    def validate_governed_semantics(
        decision: "RelationUsageGroundingDecision",
        *,
        authority: object,
        for_new_write: bool,
    ) -> None:
        """Validate through the service-owned Decision/Extract authority."""

        from services.relation_usage_grounding_service import (
            _validate_relation_usage_grounding_decision,
        )

        _validate_relation_usage_grounding_decision(
            authority, decision, for_new_write=for_new_write
        )

    @classmethod
    def create(
        cls,
        content: dict[str, Any],
        *,
        authority: object | None = None,
    ) -> "RelationUsageGroundingDecision":
        payload = dict(content)
        payload["schema_version"] = SCHEMA_VERSION
        payload.pop("grounding_decision_ref", None)
        payload.pop("grounding_decision_hash", None)
        digest = content_hash(payload)
        decision = RelationUsageGroundingDecision._from_dict_structural(
            {
                **payload,
                "grounding_decision_ref": f"relation-usage-grounding-{digest[:24]}",
                "grounding_decision_hash": f"sha256:{digest}",
            }
        )
        RelationUsageGroundingDecision.validate_governed_semantics(
            decision, authority=authority, for_new_write=True
        )
        return decision

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        authority: object | None = None,
    ) -> "RelationUsageGroundingDecision":
        decision = RelationUsageGroundingDecision._from_dict_structural(payload)
        RelationUsageGroundingDecision.validate_governed_semantics(
            decision, authority=authority, for_new_write=False
        )
        return decision

    @staticmethod
    def _from_dict_structural(
        payload: dict[str, Any],
    ) -> "RelationUsageGroundingDecision":
        if set(payload) != _FIELDS:
            raise ValueError("relation usage grounding payload must contain exact fields")
        return RelationUsageGroundingDecision(
            schema_version=str(payload.get("schema_version") or ""),
            grounding_decision_ref=str(payload.get("grounding_decision_ref") or ""),
            grounding_decision_hash=str(payload.get("grounding_decision_hash") or ""),
            organization_id=str(payload.get("organization_id") or ""),
            workspace_id=str(payload.get("workspace_id") or ""),
            source_binding=dict(payload.get("source_binding") or {}),
            extract_run_binding=dict(payload.get("extract_run_binding") or {}),
            schema_decision_binding=dict(payload.get("schema_decision_binding") or {}),
            finding_kind=str(payload.get("finding_kind") or ""),
            required_route=str(payload.get("required_route") or ""),
            subject_binding=dict(payload.get("subject_binding") or {}),
            subject_relation_binding=(
                dict(payload["subject_relation_binding"])
                if payload.get("subject_relation_binding") is not None
                else None
            ),
            corrected_relation_binding=(
                dict(payload["corrected_relation_binding"])
                if payload.get("corrected_relation_binding") is not None
                else None
            ),
            original_relation_instance=(
                dict(payload["original_relation_instance"])
                if payload.get("original_relation_instance") is not None
                else None
            ),
            corrected_relation_instance=(
                dict(payload["corrected_relation_instance"])
                if payload.get("corrected_relation_instance") is not None
                else None
            ),
            schema_learning_feedback_ref=str(
                payload.get("schema_learning_feedback_ref") or ""
            ),
            reviewed_by=str(payload.get("reviewed_by") or ""),
            occurred_at=str(payload.get("occurred_at") or ""),
        )


__all__ = ["RelationUsageGroundingDecision", "SCHEMA_VERSION"]
