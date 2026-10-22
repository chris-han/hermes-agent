"""Governed bounded Extract service boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from eos.artifact_hashing import content_hash


def _required(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _utc(value: str, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601")
    return value


@dataclass(frozen=True)
class ExtractRequest:
    request_id: str
    organization_id: str
    workspace_id: str
    lineage_id: str
    canonical_source_ir_ref: str
    canonical_source_hash: str
    schema_decision_ref: str
    schema_decision_hash: str
    requested_by: str
    schema_version: str = "extract_request.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "extract_request.v1":
            raise ValueError("schema_version must be extract_request.v1")
        for field in (
            "request_id",
            "organization_id",
            "workspace_id",
            "lineage_id",
            "canonical_source_ir_ref",
            "schema_decision_ref",
            "requested_by",
        ):
            _required(getattr(self, field), field)
        for field in ("canonical_source_hash", "schema_decision_hash"):
            if not getattr(self, field).startswith("sha256:"):
                raise ValueError(f"{field} must be a sha256 ref")

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id,
            "lineage_id": self.lineage_id,
            "canonical_source_ir_ref": self.canonical_source_ir_ref,
            "canonical_source_hash": self.canonical_source_hash,
            "schema_decision_ref": self.schema_decision_ref,
            "schema_decision_hash": self.schema_decision_hash,
            "requested_by": self.requested_by,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractRequest":
        values = {
            key: str(payload.get(key) or "")
            for key in cls.__dataclass_fields__
            if key != "schema_version"
        }
        return cls(
            **values,
            schema_version=str(
                payload.get("schema_version") or "extract_request.v1"
            ),
        )


@dataclass(frozen=True)
class ExtractResult:
    extract_run_id: str
    extract_result_hash: str
    status: str
    source_binding: dict[str, Any]
    schema_binding: dict[str, Any]
    output: dict[str, Any]
    evidence_bindings: tuple[dict[str, Any], ...]
    validation: dict[str, Any]
    provenance: dict[str, Any]
    schema_version: str = "extract_result.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "extract_result.v1":
            raise ValueError("schema_version must be extract_result.v1")
        _required(self.extract_run_id, "extract_run_id")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError("status must be succeeded or failed")
        for field in ("source_binding", "schema_binding", "output", "validation", "provenance"):
            _dict(getattr(self, field), field)
        _utc(str(self.provenance.get("started_at") or ""), "started_at")
        _utc(str(self.provenance.get("completed_at") or ""), "completed_at")
        expected = f"sha256:{content_hash(self._hash_payload())}"
        if self.extract_result_hash != expected:
            raise ValueError("extract_result_hash does not match result content")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extract_run_id": self.extract_run_id,
            "extract_result_hash": self.extract_result_hash,
            "status": self.status,
            "source_binding": dict(self.source_binding),
            "schema_binding": dict(self.schema_binding),
            "output": dict(self.output),
            "evidence_bindings": [dict(item) for item in self.evidence_bindings],
            "validation": dict(self.validation),
            "provenance": dict(self.provenance),
        }

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload.pop("extract_result_hash")
        return payload

    @classmethod
    def create(cls, content: dict[str, Any]) -> "ExtractResult":
        payload = dict(content)
        payload["schema_version"] = "extract_result.v1"
        payload.pop("extract_result_hash", None)
        digest = content_hash(payload)
        return cls.from_dict({**payload, "extract_result_hash": f"sha256:{digest}"})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtractResult":
        return cls(
            schema_version=str(payload.get("schema_version") or ""),
            extract_run_id=str(payload.get("extract_run_id") or ""),
            extract_result_hash=str(payload.get("extract_result_hash") or ""),
            status=str(payload.get("status") or ""),
            source_binding=_dict(payload.get("source_binding"), "source_binding"),
            schema_binding=_dict(payload.get("schema_binding"), "schema_binding"),
            output=_dict(payload.get("output"), "output"),
            evidence_bindings=tuple(
                dict(item) for item in (payload.get("evidence_bindings") or [])
            ),
            validation=_dict(payload.get("validation"), "validation"),
            provenance=_dict(payload.get("provenance"), "provenance"),
        )


__all__ = ["ExtractRequest", "ExtractResult"]
