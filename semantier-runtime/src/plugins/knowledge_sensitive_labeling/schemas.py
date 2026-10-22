from __future__ import annotations

TOOLSET_NAME = "knowledge_sensitive_labeling"
DASHBOARD_CONTRACT_VERSION = "knowledge_sensitive_labeling.dashboard.v1"
CLASSIFIER_PROFILE_VERSION = "sensitivity_matching.v1"
ONTOLOGY_VERSION = "sensitive_data.ontology.boundary.v1"

_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

CLASSIFY_KNOWLEDGE_INPUT_SCHEMA = {
    "description": "Classify canonical knowledge evidence with an authority-resolved rule set.",
    "parameters": _OBJECT_SCHEMA,
}
GET_CLASSIFICATION_EVIDENCE_SCHEMA = {
    "description": "Read generalized classification evidence.",
    "parameters": _OBJECT_SCHEMA,
}
GENERATE_KNOWLEDGE_LABELING_REPORT_SCHEMA = {
    "description": "Build a redaction-safe classification report from persisted evidence.",
    "parameters": _OBJECT_SCHEMA,
}
GENERATE_KNOWLEDGE_LABELING_DERIVATIVE_SCHEMA = {
    "description": "Build a derivative output manifest from persisted evidence.",
    "parameters": _OBJECT_SCHEMA,
}
VERIFY_KNOWLEDGE_LABELING_REPLAY_SCHEMA = {
    "description": "Verify generalized sensitive-labeling replay evidence.",
    "parameters": _OBJECT_SCHEMA,
}

TOOLS = (
    {
        "name": "classify_knowledge_input",
        "schema": CLASSIFY_KNOWLEDGE_INPUT_SCHEMA,
        "handler": "classify_knowledge_input",
    },
    {
        "name": "get_classification_evidence",
        "schema": GET_CLASSIFICATION_EVIDENCE_SCHEMA,
        "handler": "get_classification_evidence",
    },
    {
        "name": "generate_knowledge_labeling_report",
        "schema": GENERATE_KNOWLEDGE_LABELING_REPORT_SCHEMA,
        "handler": "generate_knowledge_labeling_report",
    },
    {
        "name": "generate_knowledge_labeling_derivative",
        "schema": GENERATE_KNOWLEDGE_LABELING_DERIVATIVE_SCHEMA,
        "handler": "generate_knowledge_labeling_derivative",
    },
    {
        "name": "verify_knowledge_labeling_replay",
        "schema": VERIFY_KNOWLEDGE_LABELING_REPLAY_SCHEMA,
        "handler": "verify_knowledge_labeling_replay",
    },
)
