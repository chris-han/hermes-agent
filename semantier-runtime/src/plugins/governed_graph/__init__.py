from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="knowledge_graph_query",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.KNOWLEDGE_GRAPH_QUERY_SCHEMA,
        handler=tools.knowledge_graph_query,
    )
