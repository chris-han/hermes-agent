from __future__ import annotations

TOOLSET_NAME = "document_extraction"
CANONICAL_DOCUMENT_SCHEMA_VERSION = "canonical_document.v1"
CANONICAL_DOCUMENT_V2_SCHEMA_VERSION = "canonical_document.v2"
PARSER_VERSION = "document_extraction.local.v1"
DOCX_PARSER_VERSION = "document_extraction.docx_ooxml.v2"

EXTRACT_DOCUMENT_CONTENT_SCHEMA = {
    "description": "Extract a governed document and return bounded page text with source hashes. Complete structure remains in the pinned canonical artifact. Use mode=replay with normalized_document_artifact_ref and offset to read further text; do not read cache/spillover files.",
    "parameters": {
        "type": "object",
        "properties": {
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 6000, "default": 6000},
            "workspace_id": {"type": "string"},
            "session_id": {"type": "string"},
            "document_ref": {
                "type": "string",
                "description": (
                    "Governed upload/artifact reference such as uploads/invoice.pdf. "
                    "Gateway-provided Hermes document cache paths are imported into "
                    "the active session uploads root before parsing."
                ),
            },
            "media_type_hint": {"type": ["string", "null"]},
            "language_hints": {"type": "array", "items": {"type": "string"}},
            "features": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["text", "layout", "tables", "key_values", "qr_codes"],
                },
            },
            "allow_ocr": {"type": "boolean", "default": True},
            "mode": {"type": "string", "enum": ["live", "replay"], "default": "live"},
            "source_artifact_ref": {"type": "string"},
            "normalized_document_artifact_ref": {"type": "string"},
            "provider_artifact_refs": {"type": "array", "items": {"type": "string"}},
            "pins": {"type": "object", "additionalProperties": True},
        },
        "required": ["mode"],
    },
}
