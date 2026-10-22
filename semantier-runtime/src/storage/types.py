"""Unified type system for Semantier storage layer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime


class ExecutionEventType(str, Enum):
    """Execution lifecycle events."""
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXECUTION_INTENT_CREATED = "EXECUTION_INTENT_CREATED"
    EXECUTION_DUPLICATE = "EXECUTION_DUPLICATE"
    EXECUTION_COMMITTED = "EXECUTION_COMMITTED"


class ValidationStatus(str, Enum):
    """Validation outcomes (unified)."""
    ALLOW = "ALLOW"
    EXCEPTION = "EXCEPTION"
    REJECT = "REJECT"


class SemanticDraftStatus(str, Enum):
    """Draft lifecycle states."""
    INCOMPLETE = "INCOMPLETE"
    PENDING_USER_INPUT = "PENDING_USER_INPUT"
    COMPLETE = "COMPLETE"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"


@dataclass
class EventRecord:
    """Unified event record structure."""
    event_id: str
    event_type: str
    occurred_at: str
    correlation_id: str
    payload: Dict[str, Any]
    
    # EOS extensions
    org_id: Optional[str] = None
    stream_id: Optional[str] = None
    ontology_version: Optional[str] = None
    projection_version: Optional[int] = None
    content_hash: Optional[str] = None
    prior_event_hash: Optional[str] = None
    signature: Optional[str] = None

    @property
    def prev_event_hash(self) -> Optional[str]:
        return self.prior_event_hash

    @prev_event_hash.setter
    def prev_event_hash(self, value: Optional[str]) -> None:
        self.prior_event_hash = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "org_id": self.org_id,
            "stream_id": self.stream_id,
            "ontology_version": self.ontology_version,
            "projection_version": self.projection_version,
            "content_hash": self.content_hash,
            "prior_event_hash": self.prior_event_hash,
            "signature": self.signature,
        }


@dataclass
class ValidationResult:
    """Unified validation result structure."""
    status: ValidationStatus
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_allow(self) -> bool:
        return self.status == ValidationStatus.ALLOW
    
    def is_exception(self) -> bool:
        return self.status == ValidationStatus.EXCEPTION
    
    def is_reject(self) -> bool:
        return self.status == ValidationStatus.REJECT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass
class IdempotencyKey:
    """Represents an idempotency key (content-addressable)."""
    org_id: str
    intent_hash: str
    
    def to_string(self) -> str:
        """Return string key for storage."""
        return f"{self.org_id}:{self.intent_hash}"
    
    @staticmethod
    def from_string(key_str: str) -> "IdempotencyKey":
        """Parse from string key."""
        parts = key_str.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid idempotency key format: {key_str}")
        return IdempotencyKey(org_id=parts[0], intent_hash=parts[1])


@dataclass
class ApprovalRecord:
    """Approval record for multi-sig."""
    idempotency_key: str
    approver: str
    approved_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticDraft:
    """In-flight semantic completion draft.
    
    Per PRD: maintains draft event per conversation until
    all required fields are complete.

    pins holds governance context versions required for replay determinism:
      O_v  — ontology version
      C_v  — COA/policy bundle version
      O_tag_v — governed tag ontology version (tiered gate)
      K_v  — knowledge governance layer version (tiered gate only)
      Pi_v — projection bundle version (tiered gate only)
      ConstraintBundle_v — semantic constraint bundle version (tiered gate)
      PrecedenceGraph_v — semantic precedence graph version (tiered gate)
      ResolutionRules_v — resolution ruleset version (tiered gate)

    projection_context_hash is persisted after commit for replay determinism.
    gate_mode selects the validation gate. v5 uses 'tiered' only.
    """
    conversation_id: str
    org_id: str
    tags: List[str] = field(default_factory=list)
    amount: Optional[str] = None
    currency: Optional[str] = None
    counterparty: Optional[str] = None
    invoice_ref: Optional[str] = None
    evidence_refs: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    status: SemanticDraftStatus = SemanticDraftStatus.INCOMPLETE
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    # FR7: Context pinning
    pins: Dict[str, Any] = field(default_factory=dict)
    # FR8: Gate selection (v5: 'tiered')
    gate_mode: str = "tiered"
    # FR9: Projection auditability
    projection_context_hash: Optional[str] = None
    projection_status: Optional[str] = None
    
    def mark_complete(self) -> None:
        """Mark draft as complete (all required fields present)."""
        self.status = SemanticDraftStatus.COMPLETE
        self.missing_fields = []
    
    def mark_incomplete(self, missing: List[str]) -> None:
        """Mark draft as incomplete with missing fields."""
        self.status = SemanticDraftStatus.INCOMPLETE
        self.missing_fields = missing
    
    def is_ready_for_validation(self) -> bool:
        """Check if draft is ready for validation."""
        return self.status == SemanticDraftStatus.COMPLETE


@dataclass
class JustificationRecord:
    """Internal representation of justification (from architecture)."""
    org_id: str
    intent_id: str
    claims: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    semantic_claim: Dict[str, Any]
    pins: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    signature: Optional[str] = None
