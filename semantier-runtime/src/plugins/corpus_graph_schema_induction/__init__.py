from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.plugin_capability_contract import validate_registered_tools

from . import schemas, tools


_CAPABILITY_TO_TOOLS = {
    "read_profiles": {"corpus_graph_schema_profiles"},
    "read_candidate": {
        "corpus_graph_schema_list",
        "corpus_graph_schema_get",
        "corpus_graph_schema_sample",
    },
    "read_graph_intent": {"corpus_graph_schema_intent_get"},
    "propose_candidate": {"corpus_graph_schema_induce"},
    "review_candidate": {"corpus_graph_schema_review"},
    "review_graph_intent": {"corpus_graph_schema_intent_review"},
    "execute_candidate_sample": {"corpus_graph_schema_sample"},
    "record_feedback_retest": {"corpus_graph_schema_feedback_retest"},
    "approve": set(),
    "admit": set(),
    "governance_submit": set(),
    "change_authority_binding": set(),
    "release_mutation": set(),
}


def register(ctx: Any) -> None:
    validate_registered_tools(
        Path(__file__).with_name("plugin.yaml"),
        schemas.TOOL_NAMES,
        capability_to_tools=_CAPABILITY_TO_TOOLS,
    )
    ctx.register_tool(name="corpus_graph_schema_profiles", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_profiles"],
                      handler=tools.corpus_graph_schema_profiles)
    ctx.register_tool(name="corpus_graph_schema_induce", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_induce"],
                      handler=tools.corpus_graph_schema_induce)
    ctx.register_tool(name="corpus_graph_schema_list", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_list"],
                      handler=tools.corpus_graph_schema_list)
    ctx.register_tool(name="corpus_graph_schema_get", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_get"],
                      handler=tools.corpus_graph_schema_get)
    ctx.register_tool(name="corpus_graph_schema_intent_get", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_intent_get"],
                      handler=tools.corpus_graph_schema_intent_get)
    ctx.register_tool(name="corpus_graph_schema_intent_review", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_intent_review"],
                      handler=tools.corpus_graph_schema_intent_review)
    ctx.register_tool(name="corpus_graph_schema_review", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_review"],
                      handler=tools.corpus_graph_schema_review)
    ctx.register_tool(name="corpus_graph_schema_sample", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_sample"],
                      handler=tools.corpus_graph_schema_sample)
    ctx.register_tool(name="corpus_graph_schema_feedback_retest", toolset=schemas.TOOLSET_NAME,
                      schema=schemas.SCHEMAS["corpus_graph_schema_feedback_retest"],
                      handler=tools.corpus_graph_schema_feedback_retest)
