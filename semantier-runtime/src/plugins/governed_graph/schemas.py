from __future__ import annotations

TOOLSET_NAME = "governed_graph"
KNOWLEDGE_GRAPH_QUERY_SCHEMA = {
    "description": (
        "Read the governed knowledge graph. Operations are neighbors, expand, lineage, impact, "
        "authority_path, rank_relationships, or explain_selection. This tool never mutates authority."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["neighbors", "expand", "lineage", "impact", "authority_path", "rank_relationships", "explain_selection"],
            },
            "node_id": {"type": "string"},
            "edge_ids": {"type": "array", "items": {"type": "string"}},
            "max_depth": {"type": "integer", "minimum": 1, "maximum": 5},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "predicates": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}
