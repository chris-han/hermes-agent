from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from semantier.sensitive_governance_contracts.registry import (
    CURRENT_CONTRACT_VERSION,
    require_supported_contract_version,
)
from semantier.sensitive_governance_contracts.enums import DisclosureAction


@dataclass(frozen=True)
class SensitiveGovernanceRequest:
    contract_version: str
    idempotency_key: str
    content: str
    purpose: str
    destination: str
    source_kind: str = "pasted_text"
    source_ref: str | None = None
    display_preferences: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SensitiveGovernanceRequest":
        version = require_supported_contract_version(
            str(payload.get("contract_version") or CURRENT_CONTRACT_VERSION)
        )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        content = str(payload.get("content") or "")
        purpose = str(payload.get("purpose") or "soe_tender_review").strip()
        destination = str(payload.get("destination") or "local_simulated_agent").strip()
        source_kind = str(payload.get("source_kind") or "pasted_text").strip()
        source_ref = str(payload.get("source_ref") or "").strip() or None
        preferences = payload.get('display_preferences', {})
        if not isinstance(preferences, dict):
            raise ValueError('invalid display preference')
        try:
            if any(not isinstance(key, str) or not key.isascii() or not isinstance(value, str)
                   for key, value in preferences.items()):
                raise ValueError
            preferences = {key: DisclosureAction(value).value for key, value in preferences.items()}
        except ValueError:
            raise ValueError('invalid display preference') from None
        if not idempotency_key:
            raise ValueError("idempotency_key required")
        if source_kind not in {"pasted_text", "txt", "md", "docx", "pdf", "xlsx"}:
            raise ValueError("source_kind must be pasted_text, txt, md, docx, pdf, or xlsx")
        if source_kind == "pasted_text":
            if source_ref:
                raise ValueError("source_ref is not allowed for pasted_text")
            if not content.strip():
                raise ValueError("content required")
        else:
            if content.strip():
                raise ValueError("document sources must use source_ref")
            if not source_ref:
                raise ValueError("source_ref required for document sources")
        return cls(
            contract_version=version,
            idempotency_key=idempotency_key,
            content=content,
            purpose=purpose,
            destination=destination,
            source_kind=source_kind,
            source_ref=source_ref,
            display_preferences=preferences,
        )


@dataclass(frozen=True)
class ExecuteAssessmentRequest:
    contract_version: str
    idempotency_key: str
    assessment_hash: str
    confirmation: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecuteAssessmentRequest":
        version = require_supported_contract_version(
            str(payload.get("contract_version") or CURRENT_CONTRACT_VERSION)
        )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        assessment_hash = str(payload.get("assessment_hash") or "").strip()
        confirmation = payload.get("confirmation") is True
        if not idempotency_key:
            raise ValueError("idempotency_key required")
        if not assessment_hash:
            raise ValueError("assessment_hash required")
        if not confirmation:
            raise ValueError("confirmation required")
        return cls(
            contract_version=version,
            idempotency_key=idempotency_key,
            assessment_hash=assessment_hash,
            confirmation=confirmation,
        )


@dataclass(frozen=True)
class RevealSegmentRequest:
    contract_version: str
    idempotency_key: str
    purpose: str
    confirmation: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RevealSegmentRequest":
        version = require_supported_contract_version(
            str(payload.get("contract_version") or CURRENT_CONTRACT_VERSION)
        )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        purpose = str(payload.get("purpose") or "soe_tender_review").strip()
        confirmation = payload.get("confirmation") is True
        if not idempotency_key:
            raise ValueError("idempotency_key required")
        if not confirmation:
            raise ValueError("confirmation required")
        return cls(
            contract_version=version,
            idempotency_key=idempotency_key,
            purpose=purpose,
            confirmation=confirmation,
        )


@dataclass(frozen=True)
class CopySegmentRequest:
    contract_version: str
    idempotency_key: str
    purpose: str
    confirmation: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CopySegmentRequest":
        reveal = RevealSegmentRequest.from_payload(payload)
        return cls(
            contract_version=reveal.contract_version,
            idempotency_key=reveal.idempotency_key,
            purpose=reveal.purpose,
            confirmation=reveal.confirmation,
        )


@dataclass(frozen=True)
class ReviewEscalationRequest:
    contract_version: str
    idempotency_key: str
    decision: str
    reason: str
    confirmation: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReviewEscalationRequest":
        version = require_supported_contract_version(
            str(payload.get("contract_version") or CURRENT_CONTRACT_VERSION)
        )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        decision = str(payload.get("decision") or "").strip().upper()
        reason = str(payload.get("reason") or "").strip()
        confirmation = payload.get("confirmation") is True
        if not idempotency_key:
            raise ValueError("idempotency_key required")
        if decision not in {"APPROVE", "REJECT"}:
            raise ValueError("decision must be APPROVE or REJECT")
        if not reason:
            raise ValueError("reason required")
        if not confirmation:
            raise ValueError("confirmation required")
        return cls(
            contract_version=version,
            idempotency_key=idempotency_key,
            decision=decision,
            reason=reason,
            confirmation=confirmation,
        )


@dataclass(frozen=True)
class ReplayRunRequest:
    contract_version: str
    idempotency_key: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplayRunRequest":
        version = require_supported_contract_version(
            str(payload.get("contract_version") or CURRENT_CONTRACT_VERSION)
        )
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key:
            raise ValueError("idempotency_key required")
        return cls(contract_version=version, idempotency_key=idempotency_key)
