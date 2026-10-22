"""Unified Storage Layer for Semantier.

Consolidates all event stores, idempotency stores, and approval stores
into a single coherent interface with dependency injection.
"""

from storage.types import (
    EventRecord,
    ValidationResult,
    ValidationStatus,
    IdempotencyKey,
    ApprovalRecord,
    SemanticDraft,
    SemanticDraftStatus,
    JustificationRecord,
)
_STORE_MODULES = {
    'UnifiedEventStore': 'event_store', 'get_event_store': 'event_store',
    'UnifiedIdempotencyStore': 'idempotency', 'get_idempotency_store': 'idempotency',
    'UnifiedApprovalStore': 'approval', 'get_approval_store': 'approval',
    'DraftStore': 'draft_store', 'get_draft_store': 'draft_store',
    'KGLStore': 'kgl_store', 'get_kgl_store': 'kgl_store',
}


def __getattr__(name):
    # EOS owns schemas used by several stores. Import a concrete store only when
    # requested, so selecting the low-level backend does not initialize EOS.
    if name not in _STORE_MODULES:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module('storage.' + _STORE_MODULES[name]), name)
    globals()[name] = value
    return value

__all__ = [
    "EventRecord",
    "ValidationResult",
    "ValidationStatus",
    "IdempotencyKey",
    "ApprovalRecord",
    "SemanticDraft",
    "SemanticDraftStatus",
    "JustificationRecord",
    "UnifiedEventStore",
    "UnifiedIdempotencyStore",
    "UnifiedApprovalStore",
    "DraftStore",
    "get_event_store",
    "get_idempotency_store",
    "get_approval_store",
    "get_draft_store",
    "KGLStore",
    "get_kgl_store",
]
