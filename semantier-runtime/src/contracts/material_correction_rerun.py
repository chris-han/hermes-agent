"""Immutable evidence for one material-correction witness and extraction rerun."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any

from eos.artifact_hashing import content_hash


SCHEMA_VERSION = "material_correction_rerun.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_GRAPH_INTENT_REF = re.compile(r"^graph-intent-[0-9a-f]{24}$")
_CANDIDATE_REF = re.compile(r"^cgsc_[0-9a-f]{24}$")
_DECISION_REF = re.compile(r"^schema-decision-[0-9a-f]{24}$")
_FEEDBACK_REF = re.compile(r"^schema-feedback-[0-9a-f]{24}$")
_CORRECTION_MATERIAL_REF = re.compile(
    r"^artifacts/material-correction-material/[0-9a-f]{64}\.json$"
)
_MACHINE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_OUTCOMES = {"accepted", "pending_adjudication", "route_only"}
_REASONS_BY_OUTCOME = {
    "accepted": {
        "schema_only_successor_recompiled_cleanly",
        "graph_intent_successor_recompiled_cleanly",
    },
    "pending_adjudication": {
        "graph_intent_successor_requires_witness_adjudication"
    },
    "route_only": {
        "occurrence_only_no_recompile",
        "schema_successor_required",
        "profile_catalog_successor_required",
        "taxonomy_definition_successor_required",
    },
}
_FIELDS = {
    "schema_version",
    "rerun_ref",
    "rerun_hash",
    "organization_id",
    "workspace_id",
    "outcome",
    "reason",
    "feedback_ref",
    "feedback_hash",
    "correction_material_hash",
    "correction_material_ref",
    "before",
    "after",
    "comparison_ref",
    "comparison_hash",
    "created_at",
    "created_by",
}
_GRAPH_INTENT_MATERIAL_FIELDS = {
    "document_function",
    "represented_or_governed_subject",
    "primary_relationship",
    "preferred_representation",
    "authority_boundary",
    "material_dimensions",
    "source_role_interpretation",
    "unresolved_questions",
}
CORRECTION_MATERIAL_SCHEMA_VERSION = "graph_intent_correction_material.v1"
SCHEMA_CANDIDATE_CORRECTION_MATERIAL_VERSION = (
    "schema_candidate_correction_material.v1"
)
RELATION_USAGE_ROUTE_MATERIAL_VERSION = "relation_usage_route_material.v1"
_CANDIDATE_DELTA_FIELDS = {
    "extraction_plan",
    "unresolved_ambiguities",
    "assumptions",
    "proposed_node_types",
    "proposed_event_types",
    "proposed_frame_types",
    "representation_decisions",
}
_ROUTE_MATERIAL_FIELDS = {
    "schema_version",
    "feedback_ref",
    "feedback_hash",
    "grounding_decision_ref",
    "grounding_decision_hash",
    "finding_kind",
    "required_route",
    "subject_ref",
    "subject_hash",
}


def graph_intent_correction_material(changes: Any) -> dict[str, Any]:
    if (
        not isinstance(changes, dict)
        or not changes
        or set(changes) - _GRAPH_INTENT_MATERIAL_FIELDS
    ):
        raise ValueError("correction material must contain exact GraphIntent material fields")

    def require_ascii_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str) or not key.isascii() or not key:
                    raise ValueError("correction material machine keys must be ASCII")
                require_ascii_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                require_ascii_keys(nested)

    require_ascii_keys(changes)
    return {
        "schema_version": CORRECTION_MATERIAL_SCHEMA_VERSION,
        "changes": dict(changes),
    }


def validate_graph_intent_correction_material(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "changes"}:
        raise ValueError("correction material must contain exact fields")
    if payload["schema_version"] != CORRECTION_MATERIAL_SCHEMA_VERSION:
        raise ValueError("correction material schema version mismatch")
    return graph_intent_correction_material(payload["changes"])


def schema_candidate_correction_material(changes: Any) -> dict[str, Any]:
    if (
        not isinstance(changes, dict)
        or not changes
        or set(changes) - _CANDIDATE_DELTA_FIELDS
    ):
        raise ValueError(
            "schema candidate correction material contains unsupported fields"
        )
    # Values remain governed candidate content and may carry localized display text.
    def require_ascii_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str) or not key.isascii() or not key:
                    raise ValueError("correction material machine keys must be ASCII")
                require_ascii_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                require_ascii_keys(nested)

    require_ascii_keys(changes)
    return {
        "schema_version": SCHEMA_CANDIDATE_CORRECTION_MATERIAL_VERSION,
        "changes": dict(changes),
    }


def relation_usage_route_material(content: Any) -> dict[str, Any]:
    if not isinstance(content, dict):
        raise ValueError("route material must be an object")
    payload = {"schema_version": RELATION_USAGE_ROUTE_MATERIAL_VERSION, **content}
    if set(payload) != _ROUTE_MATERIAL_FIELDS:
        raise ValueError("route material must contain exact fields")
    _machine_ref(payload["feedback_ref"], "route feedback_ref", _FEEDBACK_REF)
    _hash_ref(payload["feedback_hash"], "route feedback_hash")
    _machine_ref(payload["grounding_decision_ref"], "grounding_decision_ref")
    _hash_ref(payload["grounding_decision_hash"], "grounding_decision_hash")
    for field in ("finding_kind", "required_route", "subject_ref"):
        _machine_ref(payload[field], f"route {field}")
    _hash_ref(payload["subject_hash"], "route subject_hash")
    return payload


def validate_material_correction_material(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("correction material must be an object")
    version = payload.get("schema_version")
    if version == CORRECTION_MATERIAL_SCHEMA_VERSION:
        return validate_graph_intent_correction_material(payload)
    if version == SCHEMA_CANDIDATE_CORRECTION_MATERIAL_VERSION:
        if set(payload) != {"schema_version", "changes"}:
            raise ValueError("correction material must contain exact fields")
        return schema_candidate_correction_material(payload["changes"])
    if version == RELATION_USAGE_ROUTE_MATERIAL_VERSION:
        return relation_usage_route_material(
            {key: value for key, value in payload.items() if key != "schema_version"}
        )
    raise ValueError("correction material schema version mismatch")


def material_correction_outcome_reason(
    payload: dict[str, Any],
) -> tuple[str, str]:
    """Resolve the only valid lifecycle outcome/reason for exact material."""

    version = payload.get("schema_version")
    if version == CORRECTION_MATERIAL_SCHEMA_VERSION:
        return (
            "pending_adjudication",
            "graph_intent_successor_requires_witness_adjudication",
        )
    if version == SCHEMA_CANDIDATE_CORRECTION_MATERIAL_VERSION:
        return "accepted", "schema_only_successor_recompiled_cleanly"
    if version == RELATION_USAGE_ROUTE_MATERIAL_VERSION:
        route_reasons = {
            "occurrence_supersession": "occurrence_only_no_recompile",
            "schema_successor_required": "schema_successor_required",
            "profile_catalog_proposal_required": (
                "profile_catalog_successor_required"
            ),
            "taxonomy_definition_successor_required": (
                "taxonomy_definition_successor_required"
            ),
        }
        reason = route_reasons.get(payload.get("required_route"))
        if reason is None:
            raise ValueError("material correction route is unsupported")
        return "route_only", reason
    raise ValueError("material correction outcome material is unsupported")


def build_schema_feedback_retest(
    *,
    baseline_decision: dict[str, Any],
    successor_decision: dict[str, Any],
    baseline_extract: dict[str, Any],
    successor_extract: dict[str, Any],
) -> dict[str, Any]:
    """Compare persisted Decision/Extract bodies, including historical v2 replay.

    This pure comparison grants no authority to execute a new extraction or
    successor; current writers validate Decision v3 before calling it.
    """

    output_kinds = ("nodes", "edges", "events", "frames")
    pinned_graph_intent = baseline_decision.get("schema_version") in {
        "schema_decision.v2",
        "schema_decision.v3",
    } and successor_decision.get("schema_version") == baseline_decision.get(
        "schema_version"
    )
    before_output = baseline_extract.get("output") or {}
    after_output = successor_extract.get("output") or {}
    payload = {
        "schema_version": (
            "schema_feedback_retest.v2"
            if pinned_graph_intent
            else "schema_feedback_retest.v1"
        ),
        "baseline_schema_decision_ref": baseline_decision["schema_decision_id"],
        "baseline_schema_decision_hash": baseline_decision[
            "schema_decision_hash"
        ],
        "successor_schema_decision_ref": successor_decision[
            "schema_decision_id"
        ],
        "successor_schema_decision_hash": successor_decision[
            "schema_decision_hash"
        ],
        "source_binding": dict(baseline_decision["source_binding"]),
        "baseline_extract_ref": baseline_extract["extract_run_id"],
        "baseline_extract_hash": baseline_extract["extract_result_hash"],
        "successor_extract_ref": successor_extract["extract_run_id"],
        "successor_extract_hash": successor_extract["extract_result_hash"],
        "metrics": {
            kind: {
                "before": len(before_output.get(kind) or []),
                "after": len(after_output.get(kind) or []),
            }
            for kind in output_kinds
        },
        "evidence_binding_count": {
            "before": len(baseline_extract.get("evidence_bindings") or []),
            "after": len(successor_extract.get("evidence_bindings") or []),
        },
        **(
            {
                "graph_intent_bindings": {
                    "before": dict(
                        baseline_decision.get("graph_intent_binding") or {}
                    ),
                    "after": dict(
                        successor_decision.get("graph_intent_binding") or {}
                    ),
                },
                "output_hashes": {
                    kind: {
                        "before": f"sha256:{content_hash(before_output.get(kind) or [])}",
                        "after": f"sha256:{content_hash(after_output.get(kind) or [])}",
                        "changed": content_hash(before_output.get(kind) or [])
                        != content_hash(after_output.get(kind) or []),
                    }
                    for kind in output_kinds
                },
            }
            if pinned_graph_intent
            else {}
        ),
    }
    digest = content_hash(payload)
    return {
        **payload,
        "comparison_id": f"schema-feedback-retest-{digest[:24]}",
        "comparison_hash": f"sha256:{digest}",
    }


_STATE_FIELDS = {
    "graph_intent_ref",
    "graph_intent_hash",
    "candidate_ref",
    "candidate_hash",
    "schema_decision_ref",
    "schema_decision_hash",
    "positive_manifest_ref",
    "positive_manifest_hash",
    "positive_count",
    "hard_negative_manifest_ref",
    "hard_negative_manifest_hash",
    "hard_negative_count",
    "material_mismatches",
}


def _hash_ref(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase sha256 ref")
    return value


def _optional_pair(payload: dict[str, Any], ref: str, digest: str, label: str) -> None:
    values = (payload.get(ref), payload.get(digest))
    if (values[0] is None) != (values[1] is None):
        raise ValueError(f"{label} ref/hash must be both null or non-null")
    if values[1] is not None:
        _hash_ref(values[1], f"{label}_hash")


def _machine_ref(value: Any, label: str, pattern: re.Pattern[str] = _MACHINE_REF) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or pattern.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be an exact ASCII machine ref")
    return value


def _state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
        raise ValueError(f"{label} must contain exact rerun state fields")
    payload = dict(value)
    _machine_ref(
        payload["graph_intent_ref"], f"{label}.graph_intent_ref", _GRAPH_INTENT_REF
    )
    _machine_ref(
        payload["positive_manifest_ref"], f"{label}.positive_manifest_ref"
    )
    _machine_ref(
        payload["hard_negative_manifest_ref"],
        f"{label}.hard_negative_manifest_ref",
    )
    for name in (
        "graph_intent_hash",
        "positive_manifest_hash",
        "hard_negative_manifest_hash",
    ):
        _hash_ref(payload[name], f"{label}.{name}")
    _optional_pair(payload, "candidate_ref", "candidate_hash", f"{label} candidate")
    _optional_pair(
        payload,
        "schema_decision_ref",
        "schema_decision_hash",
        f"{label} decision",
    )
    if (payload["candidate_ref"] is None) != (
        payload["schema_decision_ref"] is None
    ):
        raise ValueError(f"{label} candidate and decision pins must align")
    if payload["candidate_ref"] is not None:
        _machine_ref(
            payload["candidate_ref"], f"{label}.candidate_ref", _CANDIDATE_REF
        )
        _machine_ref(
            payload["schema_decision_ref"],
            f"{label}.schema_decision_ref",
            _DECISION_REF,
        )
    if payload["positive_count"] != 32 or payload["hard_negative_count"] != 6:
        raise ValueError(f"{label} must pin exact 32+6 witness counts")
    mismatches = payload["material_mismatches"]
    if not isinstance(mismatches, list) or any(
        not isinstance(item, dict)
        or set(item) != {"witness_ref", "affected_semantics"}
        or not isinstance(item["witness_ref"], str)
        or not item["witness_ref"].isascii()
        or _MACHINE_REF.fullmatch(item["witness_ref"]) is None
        or item["affected_semantics"] != "semantic_expectation"
        for item in mismatches
    ):
        raise ValueError(
            f"{label}.material_mismatches witness_ref/affected_semantics is invalid"
        )
    return payload


@dataclass(frozen=True)
class MaterialCorrectionRerun:
    rerun_ref: str
    rerun_hash: str
    organization_id: str
    workspace_id: str
    outcome: str
    reason: str
    feedback_ref: str
    feedback_hash: str
    correction_material_hash: str
    correction_material_ref: str
    before: dict[str, Any]
    after: dict[str, Any] | None
    comparison_ref: str | None
    comparison_hash: str | None
    created_at: str
    created_by: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if self.outcome not in _OUTCOMES:
            raise ValueError("rerun outcome is invalid")
        for field in ("organization_id", "workspace_id", "created_by"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value.isascii()
                or _SAFE_SEGMENT.fullmatch(value) is None
                or value in {".", ".."}
            ):
                raise ValueError(f"{field} must be a safe ASCII identifier")
        if self.reason not in _REASONS_BY_OUTCOME[self.outcome]:
            raise ValueError("reason is invalid for rerun outcome")
        _machine_ref(self.feedback_ref, "feedback_ref", _FEEDBACK_REF)
        for field in ("feedback_hash", "correction_material_hash", "rerun_hash"):
            _hash_ref(getattr(self, field), field)
        _machine_ref(
            self.correction_material_ref,
            "correction_material_ref",
            _CORRECTION_MATERIAL_REF,
        )
        if self.correction_material_ref != (
            "artifacts/material-correction-material/"
            + self.correction_material_hash.removeprefix("sha256:")
            + ".json"
        ):
            raise ValueError("correction material ref/hash mismatch")
        before = _state(self.before, "before")
        if before["candidate_ref"] is None:
            raise ValueError("before candidate and decision pins are required")
        if before["material_mismatches"]:
            raise ValueError("before accepted witness gate must be clean")
        if self.outcome == "route_only":
            if (
                self.after is not None
                or self.comparison_ref is not None
                or self.comparison_hash is not None
            ):
                raise ValueError("route-only rerun cannot bind successor or comparison")
        else:
            if self.after is None:
                raise ValueError("material rerun requires after state")
            if self.outcome == "pending_adjudication" and (
                self.after.get("candidate_ref") is not None
                or self.after.get("candidate_hash") is not None
                or self.after.get("schema_decision_ref") is not None
                or self.after.get("schema_decision_hash") is not None
                or self.comparison_ref is not None
                or self.comparison_hash is not None
            ):
                raise ValueError(
                    "pending adjudication cannot bind candidate, decision, or comparison"
                )
            after = _state(self.after, "after")
            if self.outcome == "pending_adjudication":
                pass
            else:
                if after["candidate_ref"] is None:
                    raise ValueError("accepted rerun requires successor candidate and decision")
                if after["material_mismatches"]:
                    raise ValueError("accepted rerun requires a clean witness gate")
                pins = {
                    "comparison_ref": self.comparison_ref,
                    "comparison_hash": self.comparison_hash,
                }
                _optional_pair(pins, "comparison_ref", "comparison_hash", "comparison")
                if self.comparison_ref is None:
                    raise ValueError("accepted rerun requires comparison")
        try:
            created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise ValueError("created_at must be timezone-aware UTC ISO-8601") from exc
        if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
            raise ValueError("created_at must be timezone-aware UTC ISO-8601")
        expected_hash = f"sha256:{content_hash(self._hash_payload())}"
        expected_ref = f"material-correction-rerun:{expected_hash[7:31]}"
        if self.rerun_hash != expected_hash or self.rerun_ref != expected_ref:
            raise ValueError("material correction rerun identity mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rerun_ref": self.rerun_ref,
            "rerun_hash": self.rerun_hash,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "feedback_ref": self.feedback_ref,
            "feedback_hash": self.feedback_hash,
            "correction_material_hash": self.correction_material_hash,
            "correction_material_ref": self.correction_material_ref,
            "before": dict(self.before),
            "after": dict(self.after) if self.after is not None else None,
            "comparison_ref": self.comparison_ref,
            "comparison_hash": self.comparison_hash,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("rerun_ref")
        payload.pop("rerun_hash")
        return payload

    @classmethod
    def create(cls, content: dict[str, Any]) -> "MaterialCorrectionRerun":
        payload = dict(content)
        payload["schema_version"] = SCHEMA_VERSION
        payload.pop("rerun_ref", None)
        payload.pop("rerun_hash", None)
        digest = content_hash(payload)
        return cls.from_dict(
            {
                **payload,
                "rerun_ref": f"material-correction-rerun:{digest[:24]}",
                "rerun_hash": f"sha256:{digest}",
            }
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaterialCorrectionRerun":
        if set(payload) != _FIELDS:
            raise ValueError("material correction rerun must contain exact fields")
        return cls(
            schema_version=str(payload["schema_version"]),
            rerun_ref=str(payload["rerun_ref"]),
            rerun_hash=str(payload["rerun_hash"]),
            organization_id=str(payload["organization_id"]),
            workspace_id=str(payload["workspace_id"]),
            outcome=str(payload["outcome"]),
            reason=str(payload["reason"]),
            feedback_ref=str(payload["feedback_ref"]),
            feedback_hash=str(payload["feedback_hash"]),
            correction_material_hash=str(payload["correction_material_hash"]),
            correction_material_ref=str(payload["correction_material_ref"]),
            before=dict(payload["before"]),
            after=(dict(payload["after"]) if payload["after"] is not None else None),
            comparison_ref=(
                str(payload["comparison_ref"])
                if payload["comparison_ref"] is not None
                else None
            ),
            comparison_hash=(
                str(payload["comparison_hash"])
                if payload["comparison_hash"] is not None
                else None
            ),
            created_at=str(payload["created_at"]),
            created_by=str(payload["created_by"]),
        )


__all__ = [
    "CORRECTION_MATERIAL_SCHEMA_VERSION",
    "MaterialCorrectionRerun",
    "graph_intent_correction_material",
    "build_schema_feedback_retest",
    "material_correction_outcome_reason",
    "relation_usage_route_material",
    "schema_candidate_correction_material",
    "validate_graph_intent_correction_material",
    "validate_material_correction_material",
]
