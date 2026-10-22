from __future__ import annotations

TOOLSET_NAME = "knowledge_parser"
DASHBOARD_CONTRACT_VERSION = "knowledge_parser.dashboard.v1"

SUPPORTED_PROFILES = {
    "structured_records.v1",
    "docx.v1",
    "pdf.v1",
    "html.v1",
    "email.v1",
    "text.v1",
}

_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

PARSE_KNOWLEDGE_SOURCE_SCHEMA = {
    "description": "Normalize a governed source payload into CanonicalClassificationInputV1 evidence.",
    "parameters": _OBJECT_SCHEMA,
}
GET_CANONICAL_CLASSIFICATION_INPUT_SCHEMA = {
    "description": "Read a canonical classification input through parser evidence services.",
    "parameters": _OBJECT_SCHEMA,
}
GET_KNOWLEDGE_PARSE_RUN_SCHEMA = {
    "description": "Read parser run evidence through parser evidence services.",
    "parameters": _OBJECT_SCHEMA,
}
VERIFY_KNOWLEDGE_PARSE_REPLAY_SCHEMA = {
    "description": "Verify parser evidence hashes for replay.",
    "parameters": _OBJECT_SCHEMA,
}

TOOLS = (
    {
        "name": "parse_knowledge_source",
        "schema": PARSE_KNOWLEDGE_SOURCE_SCHEMA,
        "handler": "parse_knowledge_source",
    },
    {
        "name": "get_canonical_classification_input",
        "schema": GET_CANONICAL_CLASSIFICATION_INPUT_SCHEMA,
        "handler": "get_canonical_classification_input",
    },
    {
        "name": "get_knowledge_parse_run",
        "schema": GET_KNOWLEDGE_PARSE_RUN_SCHEMA,
        "handler": "get_knowledge_parse_run",
    },
    {
        "name": "verify_knowledge_parse_replay",
        "schema": VERIFY_KNOWLEDGE_PARSE_REPLAY_SCHEMA,
        "handler": "verify_knowledge_parse_replay",
    },
)
