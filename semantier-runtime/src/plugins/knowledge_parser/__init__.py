from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="parse_knowledge_source",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.PARSE_KNOWLEDGE_SOURCE_SCHEMA,
        handler=tools.parse_knowledge_source,
    )
    ctx.register_tool(
        name="get_canonical_classification_input",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GET_CANONICAL_CLASSIFICATION_INPUT_SCHEMA,
        handler=tools.get_canonical_classification_input,
    )
    ctx.register_tool(
        name="get_knowledge_parse_run",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GET_KNOWLEDGE_PARSE_RUN_SCHEMA,
        handler=tools.get_knowledge_parse_run,
    )
    ctx.register_tool(
        name="verify_knowledge_parse_replay",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.VERIFY_KNOWLEDGE_PARSE_REPLAY_SCHEMA,
        handler=tools.verify_knowledge_parse_replay,
    )
