from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.plugin_capability_contract import validate_registered_tools

from . import schemas, tools

_CAPABILITY_TOOLS = {
    "read_authority": set(schemas.TOOL_NAMES),
    "read_evaluation_baseline": set(schemas.TOOL_NAMES),
    "read_candidate": set(),
    "mutate_candidate": set(),
    "checkpoint": set(),
    "request_evaluation": set(),
    "governance_submit": set(),
    "approve": set(),
    "admit": set(),
    "change_authority_binding": set(),
}


def register(ctx: Any) -> None:
    registrations = [
        (schemas.TOOL_NAMES[0], schemas.SCHEMAS[schemas.TOOL_NAMES[0]], tools.tender_contextgraph_search),
        (schemas.TOOL_NAMES[1], schemas.SCHEMAS[schemas.TOOL_NAMES[1]], tools.tender_contextgraph_get_node),
        (schemas.TOOL_NAMES[2], schemas.SCHEMAS[schemas.TOOL_NAMES[2]], tools.tender_contextgraph_expand_neighborhood),
        (schemas.TOOL_NAMES[3], schemas.SCHEMAS[schemas.TOOL_NAMES[3]], tools.tender_contextgraph_get_path_evidence),
        (schemas.TOOL_NAMES[4], schemas.SCHEMAS[schemas.TOOL_NAMES[4]], tools.tender_contextgraph_review),
        (schemas.TOOL_NAMES[5], schemas.SCHEMAS[schemas.TOOL_NAMES[5]], tools.tender_contextgraph_emit_feedback),
    ]
    manifest = Path(__file__).with_name("plugin.yaml")
    validate_registered_tools(manifest, [name for name, _, _ in registrations], capability_to_tools=_CAPABILITY_TOOLS)
    for name, schema, handler in registrations:
        ctx.register_tool(name=name, toolset=schemas.TOOLSET_NAME, schema=schema, handler=handler)
