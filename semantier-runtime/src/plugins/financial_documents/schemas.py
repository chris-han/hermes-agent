from __future__ import annotations

TOOLSET_NAME = "financial_documents"
INVOICE_SCHEMA_VERSION = "invoice_contract.v1"
ADAPTER_VERSION = "financial_documents.local.v1"
VALIDATION_RULESET_VERSION = "invoice_validation.local.v1"

EXTRACT_FINANCIAL_DOCUMENT_SCHEMA = {
    "description": "Extract invoice_contract.v1 from a governed canonical_document.v1 artifact.",
    "parameters": {
        "type": "object",
        "properties": {
            "workspace_id": {"type": "string"},
            "session_id": {"type": "string"},
            "canonical_document_artifact_ref": {"type": "string"},
            "document_type_hint": {"type": "string", "default": "invoice"},
            "jurisdiction_hint": {"type": ["string", "null"]},
            "preferred_currency": {"type": ["string", "null"]},
            "allow_model_fallback": {"type": "boolean", "default": False},
            "require_evidence": {"type": "boolean", "default": True},
            "mode": {"type": "string", "enum": ["live", "replay"], "default": "live"},
            "financial_extraction_artifact_ref": {"type": "string"},
            "model_artifact_refs": {"type": "array", "items": {"type": "string"}},
            "pins": {"type": "object", "additionalProperties": True},
        },
        "required": ["mode"],
    },
}

# One typed contract is shared by tool discovery and execution validation.
from contracts.expense_reimbursement import ExpenseReimbursementRequest

EVALUATE_EXPENSE_REIMBURSEMENT_SCHEMA = {
    'description': 'Evaluate a session invoice using available governed policy, replay a pinned result, or save a review draft. Missing policy returns insufficient_basis, never approval.',
    'parameters': ExpenseReimbursementRequest.model_json_schema(),
}
