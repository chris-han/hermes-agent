from __future__ import annotations

TOOLSET_NAME = "tender_contextgraph_runtime"
TOOL_NAMES = [
    "tender_contextgraph_search",
    "tender_contextgraph_get_node",
    "tender_contextgraph_expand_neighborhood",
    "tender_contextgraph_get_path_evidence",
    "tender_contextgraph_review",
    "tender_contextgraph_emit_feedback",
]

_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "artifact_id": {"type": "string", "minLength": 1},
        "graph_version": {"type": "string", "minLength": 1},
        "content_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    },
    "required": ["artifact_id", "graph_version", "content_hash"],
}

SEARCH_SCHEMA = {**_BASE, "properties": {**_BASE["properties"], "query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "required": ["artifact_id", "graph_version", "content_hash", "query"]}
NODE_SCHEMA = {**_BASE, "properties": {**_BASE["properties"], "node_id": {"type": "string", "minLength": 1}}, "required": ["artifact_id", "graph_version", "content_hash", "node_id"]}
EXPAND_SCHEMA = {**NODE_SCHEMA, "properties": {**NODE_SCHEMA["properties"], "depth": {"type": "integer", "minimum": 1, "maximum": 3}}}
PATH_SCHEMA = {**_BASE, "properties": {**_BASE["properties"], "source_node_id": {"type": "string"}, "target_node_id": {"type": "string"}}, "required": ["artifact_id", "graph_version", "content_hash", "source_node_id", "target_node_id"]}
REVIEW_SCHEMA = {**SEARCH_SCHEMA, "properties": {**SEARCH_SCHEMA["properties"], "finding_type": {"type": "string"}}}
FEEDBACK_SCHEMA = {**NODE_SCHEMA, "properties": {**NODE_SCHEMA["properties"], "feedback_type": {"type": "string"}, "note": {"type": "string"}, "source_refs": {"type": "array", "items": {"type": "string"}}}, "required": ["artifact_id", "graph_version", "content_hash", "node_id", "feedback_type", "note"]}
SCHEMAS = dict(zip(TOOL_NAMES, [SEARCH_SCHEMA, NODE_SCHEMA, EXPAND_SCHEMA, PATH_SCHEMA, REVIEW_SCHEMA, FEEDBACK_SCHEMA]))
