from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="extract_document_content",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.EXTRACT_DOCUMENT_CONTENT_SCHEMA,
        handler=tools.extract_document_content_tool,
        emoji="📄",
    )
