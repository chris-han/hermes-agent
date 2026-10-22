from __future__ import annotations

TOOLSET_NAME = "tender_sensitive_term_detection_modeling"
DASHBOARD_CONTRACT_VERSION = "tender_sensitive_term_detection_modeling.dashboard.v1"
COMPILER_PROFILE_VERSION = "sensitive_lexicon_docx.v1"

_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

COMPILE_SENSITIVE_LEXICON_SCHEMA = {
    "description": "Compile a governed sensitive-lexicon DOCX canonical artifact into review candidates.",
    "parameters": {
        "type": "object",
        "properties": {
            "normalized_document_artifact_ref": {
                "type": "string",
                "description": "Relative governed artifact ref returned by extract_document_content.",
            },
            "knowledge_source_id": {"type": "string"},
            "source_ref": {"type": "string", "description": "Governed source ref, normally the upload ref."},
            "proposed_workspace_scope": {"type": "string"},
            "proposed_workspace_id": {"type": ["string", "null"]},
            "source_version": {"type": ["string", "null"]},
        },
        "required": [
            "normalized_document_artifact_ref",
            "knowledge_source_id",
            "source_ref",
        ],
        "additionalProperties": False,
    },
}
REVIEW_SENSITIVE_LEXICON_CANDIDATES_SCHEMA = {
    "description": "Review sensitive-lexicon candidates through governed core candidate lifecycle services.",
    "parameters": _OBJECT_SCHEMA,
}
LABEL_TENDER_DOCUMENT_SCHEMA = {
    "description": "Run governed tender-sensitive occurrence detection against an authority-resolved rule set.",
    "parameters": _OBJECT_SCHEMA,
}
GENERATE_SENSITIVE_LABELING_REPORT_SCHEMA = {
    "description": "Assemble a governed sensitive-labeling lineage report.",
    "parameters": _OBJECT_SCHEMA,
}
GENERATE_LABELED_DOCX_SCHEMA = {
    "description": "Generate a governed labeled DOCX derivative from persisted findings.",
    "parameters": _OBJECT_SCHEMA,
}
VERIFY_SENSITIVE_LABELING_REPLAY_SCHEMA = {
    "description": "Verify sensitive-labeling replay evidence through core replay services.",
    "parameters": _OBJECT_SCHEMA,
}

TOOLS = (
    {
        "name": "compile_sensitive_lexicon",
        "schema": COMPILE_SENSITIVE_LEXICON_SCHEMA,
        "handler": "compile_sensitive_lexicon",
    },
    {
        "name": "review_sensitive_lexicon_candidates",
        "schema": REVIEW_SENSITIVE_LEXICON_CANDIDATES_SCHEMA,
        "handler": "review_sensitive_lexicon_candidates",
    },
    {
        "name": "label_tender_document",
        "schema": LABEL_TENDER_DOCUMENT_SCHEMA,
        "handler": "label_tender_document",
    },
    {
        "name": "generate_sensitive_labeling_report",
        "schema": GENERATE_SENSITIVE_LABELING_REPORT_SCHEMA,
        "handler": "generate_sensitive_labeling_report",
    },
    {
        "name": "generate_labeled_docx",
        "schema": GENERATE_LABELED_DOCX_SCHEMA,
        "handler": "generate_labeled_docx",
    },
    {
        "name": "verify_sensitive_labeling_replay",
        "schema": VERIFY_SENSITIVE_LABELING_REPLAY_SCHEMA,
        "handler": "verify_sensitive_labeling_replay",
    },
)
