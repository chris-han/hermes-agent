from __future__ import annotations

from typing import Any

from . import schemas, tools


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="classify_knowledge_input",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.CLASSIFY_KNOWLEDGE_INPUT_SCHEMA,
        handler=tools.classify_knowledge_input,
    )
    ctx.register_tool(
        name="get_classification_evidence",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GET_CLASSIFICATION_EVIDENCE_SCHEMA,
        handler=tools.get_classification_evidence,
    )
    ctx.register_tool(
        name="generate_knowledge_labeling_report",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GENERATE_KNOWLEDGE_LABELING_REPORT_SCHEMA,
        handler=tools.generate_knowledge_labeling_report,
    )
    ctx.register_tool(
        name="generate_knowledge_labeling_derivative",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.GENERATE_KNOWLEDGE_LABELING_DERIVATIVE_SCHEMA,
        handler=tools.generate_knowledge_labeling_derivative,
    )
    ctx.register_tool(
        name="verify_knowledge_labeling_replay",
        toolset=schemas.TOOLSET_NAME,
        schema=schemas.VERIFY_KNOWLEDGE_LABELING_REPLAY_SCHEMA,
        handler=tools.verify_knowledge_labeling_replay,
    )
